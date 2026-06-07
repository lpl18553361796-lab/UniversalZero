"""
Performance Evaluation Experiments for Thesis

Experiment 1: Pure MCTS Baseline (no neural network) vs Random
Experiment 2: MCTS Sims Ablation (trained model, varying search budget)
Experiment 3: Network Depth Ablation (ResNet blocks: 2/4/8)
Experiment 4: Data Efficiency Ablation (self-play episodes: 3/10/20)

Usage:
    python experiments_performance.py              # Full run (~1.5h)
    python experiments_performance.py --quick      # Smoke test (~5min)
    python experiments_performance.py --plot-only  # Plot existing results
    python experiments_performance.py --exp 1      # Run only experiment 1
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import get_game_by_id, GAME_REGISTRY
from nnet.nnet import NNetWrapper
from nnet.model import UniversalNet
from coach import Coach
from arena import Arena, RandomPlayer, MCTSPlayer, evaluate_vs_random
from mcts import MCTS
from utils import dotdict

RESULTS_DIR = './experiment_results/performance/'
TRANSFER_DIR = './experiment_results/transfer_experiment/'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ================================================================
#  Uniform NNet (pure MCTS baseline — no learned knowledge)
# ================================================================

class UniformNNet:
    """
    Returns uniform policy + zero value.
    Used as baseline: MCTS with no neural network guidance.
    """
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, board):
        pi = np.ones(self.action_size) / self.action_size
        v = np.array([0.0])
        return pi, v


# ================================================================
#  Experiment 1: Pure MCTS vs Random
# ================================================================

def exp1_pure_mcts(args, results_dir):
    """Pure MCTS (uniform prior, no NN) vs Random for all games."""
    print("\n" + "=" * 60)
    print("  Experiment 1: Pure MCTS (no NN) vs Random")
    print("=" * 60)

    game_ids = ['tictactoe', 'hex_json', 'breakthrough_json']
    sims_list = args.sims_list
    num_games = args.eval_games

    results = {}

    for game_id in game_ids:
        game = get_game_by_id(game_id)
        action_size = game.get_action_size()
        uniform_nnet = UniformNNet(action_size)

        results[game_id] = {}
        for sims in sims_list:
            print(f"  {game_id} | sims={sims} | {num_games} games...", end=" ")
            start = time.time()

            ai = MCTSPlayer(game, uniform_nnet, num_sims=sims)
            rnd = RandomPlayer(game)
            arena = Arena(game, ai, rnd)
            wins, losses, draws = arena.play_games(num_games)

            total = wins + losses + draws
            wr = wins / total if total > 0 else 0
            elo = Arena.compute_elo(wins, losses, draws, opponent_elo=800)
            elapsed = time.time() - start

            results[game_id][sims] = {
                'win_rate': round(wr, 4),
                'wins': wins, 'losses': losses, 'draws': draws,
                'elo': round(elo, 1),
                'time_s': round(elapsed, 1),
            }
            print(f"WR={wr:.0%}, Elo={elo:.0f}, {elapsed:.1f}s")

    path = os.path.join(results_dir, 'exp1_pure_mcts.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return results


# ================================================================
#  Experiment 2: MCTS Sims Ablation (trained model)
# ================================================================

def exp2_sims_ablation(args, results_dir):
    """Vary MCTS sims with a trained model to show search budget impact."""
    print("\n" + "=" * 60)
    print("  Experiment 2: MCTS Sims Ablation (Trained Models)")
    print("=" * 60)

    sims_list = args.sims_list
    num_games = args.eval_games

    # Test on all games with trained checkpoints
    test_configs = [
        ('hex_json', os.path.join(TRANSFER_DIR, 'stl_hex_checkpoints')),
        ('tictactoe', os.path.join(TRANSFER_DIR, 'ttt_scratch_checkpoints')),
        ('breakthrough_json', os.path.join(TRANSFER_DIR, 'stl_bt_checkpoints')),
    ]

    results = {}

    for game_id, ckpt_dir in test_configs:
        game = get_game_by_id(game_id)
        nnet = NNetWrapper(game, game_id)

        ckpt_path = os.path.join(ckpt_dir, 'best.pth.tar')
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {game_id}: no checkpoint at {ckpt_path}")
            continue

        nnet.load_checkpoint(folder=ckpt_dir, filename='best.pth.tar')
        print(f"  Loaded: {ckpt_path}")

        results[game_id] = {}
        for sims in sims_list:
            print(f"  {game_id} | sims={sims} | {num_games} games...", end=" ")
            start = time.time()

            result = evaluate_vs_random(game, nnet,
                                        num_games=num_games,
                                        num_sims=sims)
            elapsed = time.time() - start

            results[game_id][sims] = {
                'win_rate': round(result['win_rate'], 4),
                'wins': result['wins'],
                'losses': result['losses'],
                'draws': result['draws'],
                'elo': round(result['elo'], 1),
                'time_s': round(elapsed, 1),
            }
            print(f"WR={result['win_rate']:.0%}, Elo={result['elo']:.0f}, "
                  f"{elapsed:.1f}s")

    path = os.path.join(results_dir, 'exp2_sims_ablation.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return results


# ================================================================
#  Experiment 3: Network Depth Ablation
# ================================================================

def _create_custom_nnet(game, game_id, num_res_blocks):
    """Create NNetWrapper with custom backbone depth."""
    nnet = NNetWrapper(game, game_id)
    # Replace the default model with custom depth
    nnet.nnet = UniversalNet(GAME_REGISTRY, num_res_blocks=num_res_blocks)
    nnet.nnet.to(nnet.device)
    return nnet


def exp3_depth_ablation(args, results_dir):
    """Train TTT with different ResNet depths, compare convergence."""
    print("\n" + "=" * 60)
    print("  Experiment 3: Network Depth Ablation (TTT)")
    print("=" * 60)

    game_id = 'tictactoe'
    game = get_game_by_id(game_id)
    depths = args.depth_list
    num_iters = args.ablation_iters
    eval_interval = args.ablation_eval_interval

    results = {}

    for depth in depths:
        label = f"depth_{depth}"
        print(f"\n  --- {label} (num_res_blocks={depth}) ---")

        nnet = _create_custom_nnet(game, game_id, num_res_blocks=depth)
        ckpt_dir = os.path.join(results_dir, f'{label}_checkpoints')
        ensure_dir(ckpt_dir)

        train_args = dotdict({
            'numIters': 1,
            'numEps': args.ablation_eps,
            'tempThreshold': 10,
            'num_mcts_sims': args.ablation_sims,
            'cpuct': 1.0,
            'maxlenOfQueue': 10000,
            'checkpoint': ckpt_dir,
        })

        coach = Coach(game, nnet, train_args)
        metrics = {
            'label': label,
            'num_res_blocks': depth,
            'policy_loss': [],
            'value_loss': [],
            'win_rate_history': [],
            'eval_iterations': [],
        }

        for i in range(1, num_iters + 1):
            single_args = dotdict(dict(train_args))
            single_args.numIters = 1
            single_args.checkpoint = ckpt_dir
            coach.args = single_args
            m = coach.learn()

            if m['policy_loss']:
                metrics['policy_loss'].append(m['policy_loss'][-1])
                metrics['value_loss'].append(m['value_loss'][-1])

            if i % eval_interval == 0 or i == num_iters:
                r = evaluate_vs_random(game, nnet,
                                       num_games=args.eval_games,
                                       num_sims=args.ablation_sims)
                metrics['win_rate_history'].append(round(r['win_rate'], 4))
                metrics['eval_iterations'].append(i)
                print(f"    [{label} iter {i}/{num_iters}] "
                      f"pi={metrics['policy_loss'][-1]:.4f}, "
                      f"WR={r['win_rate']:.0%}")
            else:
                pi = metrics['policy_loss'][-1] if metrics['policy_loss'] else 0
                print(f"    [{label} iter {i}/{num_iters}] pi={pi:.4f}")

        results[label] = metrics

    path = os.path.join(results_dir, 'exp3_depth_ablation.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return results


# ================================================================
#  Experiment 4: Data Efficiency Ablation
# ================================================================

def exp4_data_ablation(args, results_dir):
    """Train TTT with different self-play episode counts."""
    print("\n" + "=" * 60)
    print("  Experiment 4: Data Efficiency Ablation (TTT)")
    print("=" * 60)

    game_id = 'tictactoe'
    game = get_game_by_id(game_id)
    eps_list = args.eps_list
    num_iters = args.ablation_iters
    eval_interval = args.ablation_eval_interval

    results = {}

    for num_eps in eps_list:
        label = f"eps_{num_eps}"
        print(f"\n  --- {label} (numEps={num_eps}) ---")

        nnet = NNetWrapper(game, game_id)
        ckpt_dir = os.path.join(results_dir, f'{label}_checkpoints')
        ensure_dir(ckpt_dir)

        train_args = dotdict({
            'numIters': 1,
            'numEps': num_eps,
            'tempThreshold': 10,
            'num_mcts_sims': args.ablation_sims,
            'cpuct': 1.0,
            'maxlenOfQueue': 10000,
            'checkpoint': ckpt_dir,
        })

        coach = Coach(game, nnet, train_args)
        metrics = {
            'label': label,
            'numEps': num_eps,
            'policy_loss': [],
            'value_loss': [],
            'win_rate_history': [],
            'eval_iterations': [],
            'data_size': [],
        }

        for i in range(1, num_iters + 1):
            single_args = dotdict(dict(train_args))
            single_args.numIters = 1
            single_args.checkpoint = ckpt_dir
            coach.args = single_args
            m = coach.learn()

            if m['policy_loss']:
                metrics['policy_loss'].append(m['policy_loss'][-1])
                metrics['value_loss'].append(m['value_loss'][-1])
            metrics['data_size'].append(
                m['data_size'][-1] if m['data_size'] else 0)

            if i % eval_interval == 0 or i == num_iters:
                r = evaluate_vs_random(game, nnet,
                                       num_games=args.eval_games,
                                       num_sims=args.ablation_sims)
                metrics['win_rate_history'].append(round(r['win_rate'], 4))
                metrics['eval_iterations'].append(i)
                print(f"    [{label} iter {i}/{num_iters}] "
                      f"pi={metrics['policy_loss'][-1]:.4f}, "
                      f"WR={r['win_rate']:.0%}")
            else:
                pi = metrics['policy_loss'][-1] if metrics['policy_loss'] else 0
                print(f"    [{label} iter {i}/{num_iters}] pi={pi:.4f}")

        results[label] = metrics

    path = os.path.join(results_dir, 'exp4_data_ablation.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return results


# ================================================================
#  Dashboard
# ================================================================

def _smooth(data, window=3):
    """Rolling average smoothing for noisy curves."""
    smoothed = []
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[lo:hi]))
    return smoothed


def plot_dashboard(results_dir=RESULTS_DIR):
    """Generate 6-panel performance dashboard (3x2)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n=== Generating Performance Dashboard ===")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Performance Evaluation: Ablation Studies',
                 fontsize=16, fontweight='bold', y=0.98)

    game_colors = {
        'tictactoe': '#4CAF50',
        'hex_json': '#F44336',
        'breakthrough_json': '#2196F3',
    }
    game_labels = {
        'tictactoe': 'TTT (3x3)',
        'hex_json': 'Hex (7x7)',
        'breakthrough_json': 'BT (8x8)',
    }
    depth_colors = {'depth_2': '#FF9800', 'depth_4': '#F44336', 'depth_8': '#9C27B0'}
    eps_colors = {'eps_3': '#2196F3', 'eps_10': '#4CAF50', 'eps_20': '#F44336'}

    # --- Row 1, Left: Pure MCTS vs Random ---
    ax = axes[0, 0]
    ax.set_title('Exp 1: Pure MCTS (no NN) vs Random')
    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Win Rate vs Random')

    path = os.path.join(results_dir, 'exp1_pure_mcts.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for gid, color in game_colors.items():
            if gid in data:
                sims = sorted([int(s) for s in data[gid].keys()])
                wrs = [data[gid][str(s)]['win_rate'] for s in sims]
                ax.plot(sims, wrs, '-o', color=color, linewidth=2,
                        markersize=8, label=game_labels.get(gid, gid))
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Row 1, Right: Trained NN + MCTS Sims ---
    ax = axes[0, 1]
    ax.set_title('Exp 2: Trained NN + Varying MCTS Sims')
    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Win Rate vs Random')

    path = os.path.join(results_dir, 'exp2_sims_ablation.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for gid, color in game_colors.items():
            if gid in data:
                sims = sorted([int(s) for s in data[gid].keys()])
                wrs = [data[gid][str(s)]['win_rate'] for s in sims]
                ax.plot(sims, wrs, '-s', color=color, linewidth=2,
                        markersize=8, label=f"NN+MCTS ({game_labels.get(gid, gid)})")
        exp1_path = os.path.join(results_dir, 'exp1_pure_mcts.json')
        if os.path.exists(exp1_path):
            with open(exp1_path) as f:
                exp1 = json.load(f)
            for gid in data:
                if gid in exp1:
                    sims = sorted([int(s) for s in exp1[gid].keys()])
                    wrs = [exp1[gid][str(s)]['win_rate'] for s in sims]
                    ax.plot(sims, wrs, '--', color=game_colors.get(gid, 'gray'),
                            linewidth=1, alpha=0.5,
                            label=f"Pure MCTS ({game_labels.get(gid, gid)})")
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Row 2, Left: Depth Ablation LOSS curves ---
    ax = axes[1, 0]
    ax.set_title('Exp 3: Policy Loss — Network Depth (TTT)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')

    path = os.path.join(results_dir, 'exp3_depth_ablation.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for label, color in depth_colors.items():
            if label in data and data[label].get('policy_loss'):
                m = data[label]
                iters = list(range(1, len(m['policy_loss']) + 1))
                ax.plot(iters, m['policy_loss'], '-', color=color, linewidth=2,
                        alpha=0.9, label=f"{m['num_res_blocks']} blocks")
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Row 2, Right: Data Efficiency LOSS curves ---
    ax = axes[1, 1]
    ax.set_title('Exp 4: Policy Loss — Data Efficiency (TTT)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')

    path = os.path.join(results_dir, 'exp4_data_ablation.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for label, color in eps_colors.items():
            if label in data and data[label].get('policy_loss'):
                m = data[label]
                iters = list(range(1, len(m['policy_loss']) + 1))
                ax.plot(iters, m['policy_loss'], '-', color=color, linewidth=2,
                        alpha=0.9, label=f"{m['numEps']} eps/iter")
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Row 3, Left: Depth Ablation WR (with smoothing) ---
    ax = axes[2, 0]
    ax.set_title('Exp 3: Win Rate — Network Depth (TTT)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate vs Random')

    path = os.path.join(results_dir, 'exp3_depth_ablation.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for label, color in depth_colors.items():
            if label in data and data[label].get('win_rate_history'):
                m = data[label]
                raw_wr = m['win_rate_history']
                smooth_wr = _smooth(raw_wr, window=3)
                ax.plot(m['eval_iterations'], raw_wr,
                        'o', color=color, markersize=5, alpha=0.35)
                ax.plot(m['eval_iterations'], smooth_wr,
                        '-', color=color, linewidth=2.5,
                        label=f"{m['num_res_blocks']} blocks")
        # Pure MCTS baseline
        exp1_path = os.path.join(results_dir, 'exp1_pure_mcts.json')
        if os.path.exists(exp1_path):
            with open(exp1_path) as f:
                exp1 = json.load(f)
            if 'tictactoe' in exp1 and '25' in exp1['tictactoe']:
                baseline_wr = exp1['tictactoe']['25']['win_rate']
                ax.axhline(y=baseline_wr, color='gray', linestyle='--', alpha=0.7,
                            label=f'Pure MCTS baseline ({baseline_wr:.0%})')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Row 3, Right: Data Efficiency WR (with smoothing) ---
    ax = axes[2, 1]
    ax.set_title('Exp 4: Win Rate — Data Efficiency (TTT)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate vs Random')

    path = os.path.join(results_dir, 'exp4_data_ablation.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for label, color in eps_colors.items():
            if label in data and data[label].get('win_rate_history'):
                m = data[label]
                raw_wr = m['win_rate_history']
                smooth_wr = _smooth(raw_wr, window=3)
                ax.plot(m['eval_iterations'], raw_wr,
                        'o', color=color, markersize=5, alpha=0.35)
                ax.plot(m['eval_iterations'], smooth_wr,
                        '-', color=color, linewidth=2.5,
                        label=f"{m['numEps']} eps/iter")
        # Pure MCTS baseline
        exp1_path = os.path.join(results_dir, 'exp1_pure_mcts.json')
        if os.path.exists(exp1_path):
            with open(exp1_path) as f:
                exp1 = json.load(f)
            if 'tictactoe' in exp1 and '25' in exp1['tictactoe']:
                baseline_wr = exp1['tictactoe']['25']['win_rate']
                ax.axhline(y=baseline_wr, color='gray', linestyle='--', alpha=0.7,
                            label=f'Pure MCTS baseline ({baseline_wr:.0%})')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'performance_dashboard.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Dashboard saved: {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Performance Summary")
    print("=" * 60)

    for exp_name in ['exp1_pure_mcts', 'exp2_sims_ablation',
                     'exp3_depth_ablation', 'exp4_data_ablation']:
        path = os.path.join(results_dir, f'{exp_name}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"\n  {exp_name}:")
            if exp_name in ('exp1_pure_mcts', 'exp2_sims_ablation'):
                for gid, sims_data in data.items():
                    if isinstance(sims_data, dict):
                        best_sims = max(sims_data.keys(),
                                        key=lambda s: sims_data[s].get('win_rate', 0))
                        best = sims_data[best_sims]
                        print(f"    {gid}: best WR={best['win_rate']:.0%} "
                              f"at sims={best_sims}")
            else:
                for label, m in data.items():
                    if isinstance(m, dict) and m.get('win_rate_history'):
                        final_wr = m['win_rate_history'][-1]
                        print(f"    {label}: final WR={final_wr:.0%}")


# ================================================================
#  Config
# ================================================================

def get_args(quick=False):
    if quick:
        return dotdict({
            'sims_list': [5, 25],
            'eval_games': 6,
            'depth_list': [2, 4],
            'eps_list': [3, 10],
            'ablation_iters': 5,
            'ablation_eps': 3,
            'ablation_sims': 10,
            'ablation_eval_interval': 2,
        })
    else:
        return dotdict({
            'sims_list': [10, 25, 50, 100],
            'eval_games': 40,
            'depth_list': [2, 4, 8],
            'eps_list': [3, 10, 20],
            'ablation_iters': 50,
            'ablation_eps': 10,
            'ablation_sims': 25,
            'ablation_eval_interval': 5,
        })


# ================================================================
#  Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Performance Experiments')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--exp', type=int, default=0,
                        help='Run only specific experiment (1-4), 0=all')
    cli = parser.parse_args()

    if cli.plot_only:
        plot_dashboard(RESULTS_DIR)
        return

    args = get_args(quick=cli.quick)
    ensure_dir(RESULTS_DIR)

    mode = "QUICK" if cli.quick else "FULL"
    print(f"\n{'#' * 60}")
    print(f"  Performance Experiments ({mode})")
    print(f"  Sims: {args.sims_list}, Eval games: {args.eval_games}")
    print(f"  Ablation iters: {args.ablation_iters}")
    print(f"{'#' * 60}")

    total_start = time.time()
    run = cli.exp

    if run in (0, 1):
        exp1_pure_mcts(args, RESULTS_DIR)
    if run in (0, 2):
        exp2_sims_ablation(args, RESULTS_DIR)
    if run in (0, 3):
        exp3_depth_ablation(args, RESULTS_DIR)
    if run in (0, 4):
        exp4_data_ablation(args, RESULTS_DIR)

    total = time.time() - total_start
    print(f"\n{'#' * 60}")
    print(f"  All experiments complete! {total:.1f}s ({total/60:.1f}min)")
    print(f"{'#' * 60}")

    plot_dashboard(RESULTS_DIR)


if __name__ == '__main__':
    main()
