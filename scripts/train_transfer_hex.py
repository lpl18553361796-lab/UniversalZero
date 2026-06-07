import argparse
import json
import os
import sys
import time

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.append(os.path.join(PROJECT_ROOT, "games"))
    sys.path.append(os.path.join(PROJECT_ROOT, "core"))
    sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))

from core.coach import Coach
from game import get_game_by_id
from nnet.nnet import NNetWrapper
from scripts.weight_surger import surgery_expert_brain
from utils import dotdict


def build_args(cli):
    return dotdict({
        'lr': cli.lr,
        'numIters': cli.iters,
        'numEps': cli.episodes,
        'tempThreshold': cli.temp_threshold,
        'updateThreshold': 0.55,
        'maxlenOfQueue': 200000,
        'num_mcts_sims': cli.mcts_sims,
        'num_workers': cli.workers,
        'batch_size': cli.batch_size,
        'epochs': cli.epochs,
        'arenaCompare': 20,
        'cpuct': cli.cpuct,
        'checkpoint': cli.output_dir,
        'cuda': torch.cuda.is_available(),
    })


def ensure_seed(source_expert, seed_path):
    if os.path.exists(seed_path):
        return seed_path
    surgery_expert_brain(source_expert, target_game_id='hex')
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Expected transfer seed was not created: {seed_path}")
    return seed_path


def main():
    parser = argparse.ArgumentParser(description='Train Hex from an injected Othello expert seed.')
    parser.add_argument('--source-expert', default=os.path.join('pretrained_models', 'othello_expert_8x8.pth.tar'))
    parser.add_argument('--seed', default=os.path.join('experiment_results', 'expert_injected_hex.pth.tar'))
    parser.add_argument('--iters', type=int, default=60)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--mcts-sims', type=int, default=200)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cpuct', type=float, default=1.0)
    parser.add_argument('--temp-threshold', type=int, default=15)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--output-dir', default=os.path.join('experiment_results', 'hex_transfer'))
    args_cli = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    os.makedirs(args_cli.output_dir, exist_ok=True)
    seed_path = ensure_seed(args_cli.source_expert, args_cli.seed)

    game = get_game_by_id('hex')
    args = build_args(args_cli)
    nnet = NNetWrapper(game, 'hex', args=args)
    nnet.load_checkpoint(folder=os.path.dirname(seed_path), filename=os.path.basename(seed_path))
    if args_cli.freeze_backbone:
        nnet.set_backbone_frozen(True)

    started = time.time()
    result = Coach(game, nnet, args).learn()

    final_model = os.path.join(args_cli.output_dir, 'hex_transfer_final.pth.tar')
    nnet.save_checkpoint(folder=args_cli.output_dir, filename='hex_transfer_final.pth.tar')

    metrics = {
        'experiment': 'hex_transfer',
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started)),
        'duration_seconds': round(time.time() - started, 1),
        'source_expert': args_cli.source_expert,
        'seed_model': seed_path,
        'freeze_backbone': args_cli.freeze_backbone,
        'args': dict(args_cli.__dict__),
        'final_model': final_model,
        'training': result.get('iterations', []),
    }
    metrics_path = os.path.join(args_cli.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] Transfer Hex model: {final_model}")
    print(f"[DONE] Transfer Hex metrics: {metrics_path}")


if __name__ == '__main__':
    main()
