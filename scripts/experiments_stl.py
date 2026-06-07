"""
STL Baseline Experiments (Single-Task Learning)

Train Breakthrough and Hex independently for 50 iterations each,
recording convergence speed (loss curves + periodic Elo evaluation).

These baselines will later be compared against MTL (Multi-Task Learning)
to quantify transfer learning benefits and catastrophic forgetting.

Usage:
    python experiments_stl.py              # Full 50-iteration run
    python experiments_stl.py --quick      # Quick 10-iteration smoke test
    python experiments_stl.py --plot-only  # Plot existing results
"""
import os
import sys
import json
import time
import numpy as np
import torch

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import get_game_by_id
from nnet.nnet import NNetWrapper
from coach import Coach
from arena import evaluate_vs_random
from utils import dotdict

RESULTS_DIR = './experiment_results/stl_baselines/'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def train_stl(game_id, args, results_dir, eval_interval=5):
    """
    Single-Task Learning baseline for one game.

    Trains for args.numIters iterations, evaluating vs Random
    every eval_interval iterations.

    Args:
        game_id: Game identifier (e.g. 'breakthrough_json', 'hex_json')
        args: Training hyperparameters
        results_dir: Where to save checkpoints and metrics
        eval_interval: Evaluate vs Random every N iterations

    Returns:
        dict: Complete metrics with loss curves and eval history
    """
    print(f"\n{'=' * 60}")
    print(f"  STL Baseline: {game_id}")
    print(f"  Iterations: {args.numIters}, Episodes/iter: {args.numEps}")
    print(f"  MCTS sims: {args.num_mcts_sims}, Eval interval: {eval_interval}")
    print(f"{'=' * 60}")

    game = get_game_by_id(game_id)
    nnet = NNetWrapper(game, game_id)

    checkpoint_dir = os.path.join(results_dir, f'{game_id}_checkpoints')
    ensure_dir(checkpoint_dir)

    # Per-iteration training with evaluation checkpoints
    all_metrics = {
        'game_id': game_id,
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'data_size': [],
        'elo_history': [],
        'win_rate_history': [],
        'eval_iterations': [],
        'wall_time_per_iter': [],
        'cumulative_wall_time': [],
        'args': dict(args),
    }

    # Use Coach iteratively (1 iter at a time) for fine-grained tracking
    coach = Coach(game, nnet, args)
    cumulative_time = 0.0

    for i in range(1, args.numIters + 1):
        iter_start = time.time()

        # Single iteration of self-play + train
        single_args = dotdict(dict(args))
        single_args.numIters = 1
        single_args.checkpoint = checkpoint_dir
        coach.args = single_args
        metrics = coach.learn()

        iter_time = time.time() - iter_start
        cumulative_time += iter_time

        # Record loss metrics
        if metrics['policy_loss']:
            all_metrics['policy_loss'].append(metrics['policy_loss'][-1])
            all_metrics['value_loss'].append(metrics['value_loss'][-1])
            all_metrics['total_loss'].append(metrics['total_loss'][-1])
        all_metrics['data_size'].append(metrics['data_size'][-1] if metrics['data_size'] else 0)
        all_metrics['wall_time_per_iter'].append(round(iter_time, 2))
        all_metrics['cumulative_wall_time'].append(round(cumulative_time, 2))

        # Periodic evaluation vs Random
        if i % eval_interval == 0 or i == args.numIters:
            print(f"  [{game_id} iter {i}/{args.numIters}] Evaluating vs Random...")
            eval_result = evaluate_vs_random(
                game, nnet,
                num_games=args.eval_games,
                num_sims=args.eval_sims
            )
            all_metrics['elo_history'].append(round(eval_result['elo'], 1))
            all_metrics['win_rate_history'].append(round(eval_result['win_rate'], 4))
            all_metrics['eval_iterations'].append(i)
            print(f"    WR: {eval_result['win_rate']:.1%}, "
                  f"Elo: {eval_result['elo']:.0f}, "
                  f"Time: {cumulative_time:.1f}s")
        else:
            # Print progress without eval
            pi_l = all_metrics['policy_loss'][-1] if all_metrics['policy_loss'] else 0
            v_l = all_metrics['value_loss'][-1] if all_metrics['value_loss'] else 0
            print(f"  [{game_id} iter {i}/{args.numIters}] "
                  f"pi={pi_l:.4f}, v={v_l:.4f}, "
                  f"time={iter_time:.1f}s")

    # Save final metrics
    metrics_path = os.path.join(results_dir, f'stl_{game_id}.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrics saved: {metrics_path}")

    return all_metrics


def plot_stl_baselines(results_dir=RESULTS_DIR):
    """Generate convergence comparison plot for STL baselines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n=== Generating STL Baseline Dashboard ===")

    # Load metrics
    bt_path = os.path.join(results_dir, 'stl_breakthrough_json.json')
    hex_path = os.path.join(results_dir, 'stl_hex_json.json')

    data = {}
    for name, path in [('BT', bt_path), ('Hex', hex_path)]:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data[name] = json.load(f)
            print(f"  Loaded: {os.path.basename(path)}")
        else:
            print(f"  Missing: {os.path.basename(path)}")

    if not data:
        print("  No data found. Run the experiment first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('STL Baselines: Breakthrough vs Hex (Independent Training)',
                 fontsize=14, fontweight='bold')

    colors = {'BT': '#2196F3', 'Hex': '#F44336'}

    # --- Plot 1: Policy Loss ---
    ax = axes[0, 0]
    ax.set_title('Policy Loss Convergence')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    for name, m in data.items():
        if m.get('policy_loss'):
            iters = range(1, len(m['policy_loss']) + 1)
            ax.plot(iters, m['policy_loss'], '-o', color=colors[name],
                    markersize=2, linewidth=1.5, label=name, alpha=0.8)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 2: Value Loss ---
    ax = axes[0, 1]
    ax.set_title('Value Loss Convergence')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value Loss')
    for name, m in data.items():
        if m.get('value_loss'):
            iters = range(1, len(m['value_loss']) + 1)
            ax.plot(iters, m['value_loss'], '-s', color=colors[name],
                    markersize=2, linewidth=1.5, label=name, alpha=0.8)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 3: Win Rate vs Random ---
    ax = axes[1, 0]
    ax.set_title('Win Rate vs Random Player')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate')
    for name, m in data.items():
        if m.get('win_rate_history') and m.get('eval_iterations'):
            ax.plot(m['eval_iterations'], m['win_rate_history'],
                    '-D', color=colors[name], markersize=5, linewidth=2,
                    label=name, alpha=0.9)
            ax.fill_between(m['eval_iterations'], 0, m['win_rate_history'],
                            alpha=0.1, color=colors[name])
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7,
               label='50% baseline')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 4: Elo Evolution ---
    ax = axes[1, 1]
    ax.set_title('Elo Rating vs Random')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Elo Rating')
    for name, m in data.items():
        if m.get('elo_history') and m.get('eval_iterations'):
            ax.plot(m['eval_iterations'], m['elo_history'],
                    '-^', color=colors[name], markersize=5, linewidth=2,
                    label=name, alpha=0.9)
    ax.axhline(y=800, color='gray', linestyle=':', linewidth=1, alpha=0.7,
               label='Random (Elo=800)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'stl_baselines_dashboard.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Dashboard saved: {save_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("  STL Baseline Summary")
    print("=" * 60)
    for name, m in data.items():
        n_iters = len(m.get('policy_loss', []))
        final_pi = m['policy_loss'][-1] if m.get('policy_loss') else 'N/A'
        final_v = m['value_loss'][-1] if m.get('value_loss') else 'N/A'
        final_wr = m['win_rate_history'][-1] if m.get('win_rate_history') else 'N/A'
        final_elo = m['elo_history'][-1] if m.get('elo_history') else 'N/A'
        total_time = m['cumulative_wall_time'][-1] if m.get('cumulative_wall_time') else 'N/A'

        print(f"\n  {name} ({m.get('game_id', '?')}):")
        print(f"    Iterations:     {n_iters}")
        print(f"    Final pi_loss:  {final_pi}")
        print(f"    Final v_loss:   {final_v}")
        print(f"    Final win_rate: {final_wr}")
        print(f"    Final Elo:      {final_elo}")
        print(f"    Total time:     {total_time}s")


def run_stl_baselines(quick=False):
    """Run STL baseline experiments for BT and Hex."""
    ensure_dir(RESULTS_DIR)

    if quick:
        args = dotdict({
            'numIters': 10,
            'numEps': 3,
            'tempThreshold': 10,
            'num_mcts_sims': 10,
            'cpuct': 1.0,
            'maxlenOfQueue': 5000,
            'eval_games': 6,
            'eval_sims': 8,
        })
        eval_interval = 2
    else:
        args = dotdict({
            'numIters': 50,
            'numEps': 5,
            'tempThreshold': 15,
            'num_mcts_sims': 15,
            'cpuct': 1.0,
            'maxlenOfQueue': 20000,
            'eval_games': 10,
            'eval_sims': 10,
        })
        eval_interval = 5

    start_time = time.time()

    # Phase 1: Breakthrough STL
    bt_metrics = train_stl(
        'breakthrough_json', dotdict(dict(args)),
        RESULTS_DIR, eval_interval=eval_interval
    )

    # Phase 2: Hex STL
    hex_metrics = train_stl(
        'hex_json', dotdict(dict(args)),
        RESULTS_DIR, eval_interval=eval_interval
    )

    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  STL Baselines Complete!")
    print(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 60}")

    # Generate dashboard
    plot_stl_baselines(RESULTS_DIR)


if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        plot_stl_baselines(RESULTS_DIR)
    elif '--quick' in sys.argv:
        run_stl_baselines(quick=True)
    else:
        run_stl_baselines(quick=False)
