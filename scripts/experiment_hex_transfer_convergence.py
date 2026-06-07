import argparse
import csv
import json
import os
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "games"),
    os.path.join(PROJECT_ROOT, "core"),
    os.path.join(PROJECT_ROOT, "scripts"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core.arena import Arena, MCTSPlayer, RandomPlayer
from core.coach import Coach
from game import get_game_by_id
from nnet.nnet import NNetWrapper
from scripts.weight_surger import surgery_expert_brain
from utils import dotdict


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_stages(value):
    stages = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not stages or stages[0] < 0:
        raise argparse.ArgumentTypeError("Stages must be non-negative integers.")
    if 0 not in stages:
        stages.insert(0, 0)
    return stages


def build_train_args(cli, checkpoint_dir, stages):
    return dotdict({
        "lr": cli.lr,
        "numIters": max(stages),
        "numEps": cli.episodes,
        "tempThreshold": cli.temp_threshold,
        "updateThreshold": 0.55,
        "maxlenOfQueue": 200000,
        "num_mcts_sims": cli.train_mcts_sims,
        "num_workers": cli.workers,
        "batch_size": cli.batch_size,
        "epochs": cli.epochs,
        "arenaCompare": 20,
        "cpuct": cli.cpuct,
        "checkpoint": checkpoint_dir,
        "checkpoint_stages": [stage for stage in stages if stage > 0],
        "seed": cli.seed,
        "cuda": torch.cuda.is_available(),
    })


def checkpoint_path(checkpoint_dir, stage):
    return os.path.join(checkpoint_dir, f"stage_{stage:03d}.pth.tar")


def train_scratch(cli, game, stages, output_dir):
    checkpoint_dir = os.path.join(output_dir, "scratch")
    os.makedirs(checkpoint_dir, exist_ok=True)
    args = build_train_args(cli, checkpoint_dir, stages)

    seed_everything(cli.seed)
    nnet = NNetWrapper(game, "hex", args=args)
    nnet.save_checkpoint(checkpoint_dir, "stage_000.pth.tar")
    result = Coach(game, nnet, args).learn() if max(stages) > 0 else {"iterations": []}
    return result.get("iterations", [])


def train_transfer(cli, game, stages, output_dir):
    checkpoint_dir = os.path.join(output_dir, "transfer")
    os.makedirs(checkpoint_dir, exist_ok=True)
    args = build_train_args(cli, checkpoint_dir, stages)
    seed_path = os.path.join(output_dir, "transfer_seed.pth.tar")

    seed_everything(cli.seed)
    surgery_expert_brain(
        cli.source_expert,
        target_game_id="hex",
        output_path=seed_path,
    )

    nnet = NNetWrapper(game, "hex", args=args)
    nnet.load_checkpoint(
        folder=os.path.dirname(seed_path),
        filename=os.path.basename(seed_path),
    )
    if cli.freeze_transfer:
        nnet.set_backbone_frozen(True)
    nnet.save_checkpoint(checkpoint_dir, "stage_000.pth.tar")
    result = Coach(game, nnet, args).learn() if max(stages) > 0 else {"iterations": []}
    return result.get("iterations", [])


def load_model(game, path):
    nnet = NNetWrapper(game, "hex", args=dotdict({}))
    nnet.load_checkpoint(
        folder=os.path.dirname(path),
        filename=os.path.basename(path),
    )
    return nnet


def matchup(game, first, second, games, mcts_sims, seed, second_is_random=False):
    seed_everything(seed)
    first_player = MCTSPlayer(game, first, num_sims=mcts_sims)
    if second_is_random:
        second_player = RandomPlayer(game)
    else:
        second_player = MCTSPlayer(game, second, num_sims=mcts_sims)
    wins, losses, draws = Arena(game, first_player, second_player).play_games(games)
    total = wins + losses + draws
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total if total else 0.0,
    }


def metrics_by_stage(metrics):
    return {int(row["iteration"]): row for row in metrics}


def evaluate_stages(cli, game, stages, output_dir, scratch_metrics, transfer_metrics):
    scratch_losses = metrics_by_stage(scratch_metrics)
    transfer_losses = metrics_by_stage(transfer_metrics)
    rows = []
    details = []

    for stage in stages:
        print(f"\n[EVAL] Stage {stage}")
        scratch_path = checkpoint_path(os.path.join(output_dir, "scratch"), stage)
        transfer_path = checkpoint_path(os.path.join(output_dir, "transfer"), stage)
        if not os.path.exists(scratch_path):
            raise FileNotFoundError(f"Missing scratch checkpoint: {scratch_path}")
        if not os.path.exists(transfer_path):
            raise FileNotFoundError(f"Missing transfer checkpoint: {transfer_path}")

        scratch = load_model(game, scratch_path)
        transfer = load_model(game, transfer_path)
        eval_seed = cli.seed + stage * 10000

        scratch_random = matchup(
            game, scratch, None, cli.eval_games, cli.eval_mcts_sims,
            eval_seed + 1, second_is_random=True,
        )
        transfer_random = matchup(
            game, transfer, None, cli.eval_games, cli.eval_mcts_sims,
            eval_seed + 2, second_is_random=True,
        )
        transfer_scratch = matchup(
            game, transfer, scratch, cli.eval_games, cli.eval_mcts_sims,
            eval_seed + 3,
        )

        scratch_loss = scratch_losses.get(stage, {})
        transfer_loss = transfer_losses.get(stage, {})
        row = {
            "training_stage": stage,
            "scratch_vs_random_win_rate": round(scratch_random["win_rate"], 4),
            "transfer_vs_random_win_rate": round(transfer_random["win_rate"], 4),
            "transfer_vs_scratch_win_rate": round(transfer_scratch["win_rate"], 4),
            "scratch_policy_loss": scratch_loss.get("policy_loss", ""),
            "scratch_value_loss": scratch_loss.get("value_loss", ""),
            "scratch_total_loss": scratch_loss.get("total_loss", ""),
            "transfer_policy_loss": transfer_loss.get("policy_loss", ""),
            "transfer_value_loss": transfer_loss.get("value_loss", ""),
            "transfer_total_loss": transfer_loss.get("total_loss", ""),
        }
        rows.append(row)
        details.append({
            "training_stage": stage,
            "scratch_checkpoint": scratch_path,
            "transfer_checkpoint": transfer_path,
            "scratch_vs_random": scratch_random,
            "transfer_vs_random": transfer_random,
            "transfer_vs_scratch": transfer_scratch,
        })
        print(
            f"  Scratch-Random={scratch_random['win_rate']:.1%}, "
            f"Transfer-Random={transfer_random['win_rate']:.1%}, "
            f"Transfer-Scratch={transfer_scratch['win_rate']:.1%}"
        )

    return rows, details


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_line_panel(draw, box, title, stages, series, y_min, y_max, y_label):
    x0, y0, x1, y1 = box
    title_font = get_font(28, bold=True)
    text_font = get_font(19)
    small_font = get_font(16)
    draw.text((x0, y0), title, font=title_font, fill=(30, 30, 30))

    left, top = x0 + 90, y0 + 65
    right, bottom = x1 - 30, y1 - 75
    draw.line((left, top, left, bottom), fill=(70, 70, 70), width=3)
    draw.line((left, bottom, right, bottom), fill=(70, 70, 70), width=3)

    for index in range(5):
        fraction = index / 4
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * (y_max - y_min)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=2)
        label = f"{value:.2f}" if y_max <= 10 else f"{value:.0f}"
        draw.text((left - 70, y - 10), label, font=small_font, fill=(80, 80, 80))

    stage_min, stage_max = min(stages), max(stages)
    stage_span = max(1, stage_max - stage_min)

    def point(stage, value):
        px = left + (stage - stage_min) / stage_span * (right - left)
        py = bottom - (value - y_min) / max(1e-9, y_max - y_min) * (bottom - top)
        return px, py

    for name, values, color in series:
        valid = [(stage, value) for stage, value in zip(stages, values) if value is not None]
        points = [point(stage, value) for stage, value in valid]
        if len(points) >= 2:
            draw.line(points, fill=color, width=5)
        for px, py in points:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)

    for stage in stages:
        px, _ = point(stage, y_min)
        draw.text((px - 10, bottom + 15), str(stage), font=small_font, fill=(70, 70, 70))

    draw.text(((left + right) / 2 - 55, bottom + 45), "Training stage", font=text_font, fill=(50, 50, 50))
    draw.text((x0 + 5, top - 35), y_label, font=text_font, fill=(50, 50, 50))

    legend_x = left + 15
    legend_y = top + 10
    for name, _, color in series:
        draw.line((legend_x, legend_y + 9, legend_x + 35, legend_y + 9), fill=color, width=5)
        draw.text((legend_x + 45, legend_y), name, font=small_font, fill=(50, 50, 50))
        legend_y += 28


def write_plot(rows, path):
    stages = [int(row["training_stage"]) for row in rows]

    def loss_values(key):
        values = []
        for row in rows:
            value = row[key]
            values.append(float(value) if value != "" else None)
        return values

    scratch_loss = loss_values("scratch_total_loss")
    transfer_loss = loss_values("transfer_total_loss")
    valid_losses = [value for value in scratch_loss + transfer_loss if value is not None]
    loss_max = max(valid_losses) * 1.1 if valid_losses else 1.0

    image = Image.new("RGB", (1500, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (365, 35),
        "Hex Training Loss Convergence",
        font=get_font(42, bold=True),
        fill=(25, 25, 25),
    )
    draw_line_panel(
        draw,
        (120, 125, 1400, 940),
        "Total Training Loss",
        stages,
        [
            ("Scratch total loss", scratch_loss, (208, 140, 63)),
            ("Transfer total loss", transfer_loss, (47, 111, 143)),
        ],
        0,
        loss_max,
        "Total loss",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, "PNG", dpi=(300, 300))


def main():
    parser = argparse.ArgumentParser(
        description="Compare Hex transfer and scratch convergence under aligned budgets."
    )
    parser.add_argument(
        "--source-expert",
        default=os.path.join("pretrained_models", "othello_expert_8x8.pth.tar"),
    )
    parser.add_argument("--stages", type=parse_stages, default=parse_stages("0,5,10,15,20"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--train-mcts-sims", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-mcts-sims", type=int, default=50)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cpuct", type=float, default=1.0)
    parser.add_argument("--temp-threshold", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--freeze-transfer",
        action="store_true",
        help="Freeze transfer backbone and value head. Default is full fine-tuning for a fair initialization comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("experiment_results", "hex_transfer_convergence"),
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if not os.path.exists(args.source_expert):
        raise FileNotFoundError(f"Othello expert not found: {args.source_expert}")

    stages = args.stages
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    game = get_game_by_id("hex")
    started = time.time()

    print("[TRAIN] Scratch model")
    scratch_metrics = train_scratch(args, game, stages, output_dir)
    print("\n[TRAIN] Transfer model")
    transfer_metrics = train_transfer(args, game, stages, output_dir)

    rows, evaluation = evaluate_stages(
        args,
        game,
        stages,
        output_dir,
        scratch_metrics,
        transfer_metrics,
    )

    csv_path = os.path.join(output_dir, "hex_transfer_convergence.csv")
    png_path = os.path.join(output_dir, "hex_transfer_loss_convergence.png")
    json_path = os.path.join(output_dir, "hex_transfer_convergence_details.json")
    write_csv(rows, csv_path)
    write_plot(rows, png_path)

    report = {
        "experiment": "hex_transfer_convergence",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - started, 1),
        "stages": stages,
        "seed": args.seed,
        "freeze_transfer": args.freeze_transfer,
        "settings": vars(args),
        "scratch_training": scratch_metrics,
        "transfer_training": transfer_metrics,
        "evaluation": evaluation,
        "csv": csv_path,
        "figure": png_path,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n[DONE]")
    print(f"  CSV: {csv_path}")
    print(f"  PNG: {png_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
