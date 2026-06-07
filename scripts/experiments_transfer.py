"""
Transfer Learning Experiment: Strong Training + Zero-Shot Transfer Test

Experiment Design:
    Phase 1: Strong STL baselines (BT + Hex, independently)
    Phase 2: MTL joint training (BT + Hex, shared backbone)
    Phase 3: Transfer test (TicTacToe as held-out game)
        - Group A: Transfer — freeze MTL backbone, train only TTT policy head
        - Group B: From scratch — train TTT independently (STL baseline)
        - Compare convergence speed and final win rate

    If transfer works: Group A should converge faster than Group B.

Usage:
    python experiments_transfer.py                # Full experiment (~5-6h)
    python experiments_transfer.py --quick        # Quick smoke test (~15min)
    python experiments_transfer.py --phase 1      # Only Phase 1
    python experiments_transfer.py --phase 2      # Only Phase 2
    python experiments_transfer.py --phase 3      # Only Phase 3
    python experiments_transfer.py --plot-only    # Plot existing results
"""
import os
import sys
import json
import time
import copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import get_game_by_id, GAME_REGISTRY
from nnet.nnet import NNetWrapper
from coach import Coach, MultiTaskCoach
from arena import evaluate_vs_random
from utils import dotdict

RESULTS_DIR = './experiment_results/transfer_experiment/'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ================================================================
#  Shared: iterative training with periodic evaluation
# ================================================================

def train_iterative(game_id, nnet, args, results_dir, label,
                    eval_interval=5, eval_games=10, eval_sims=10):
    """
    Train a single game iteratively, recording loss + win rate at intervals.

    Args:
        game_id: Game to train on
        nnet: NNetWrapper (may have pre-loaded weights for transfer)
        args: Training hyperparams
        results_dir: Where to save checkpoints
        label: Identifier for this run (e.g. 'stl_bt', 'transfer_ttt')
        eval_interval: Evaluate vs Random every N iterations
        eval_games: Number of games per evaluation
        eval_sims: MCTS sims per evaluation move

    Returns:
        dict: Metrics with loss curves and eval history
    """
    game = get_game_by_id(game_id)
    checkpoint_dir = os.path.join(results_dir, f'{label}_checkpoints')
    ensure_dir(checkpoint_dir)

    metrics = {
        'label': label,
        'game_id': game_id,
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'data_size': [],
        'win_rate_history': [],
        'elo_history': [],
        'eval_iterations': [],
        'wall_time_per_iter': [],
        'cumulative_wall_time': [],
        'args': {k: v for k, v in dict(args).items()
                 if isinstance(v, (int, float, str, bool))},
    }

    coach = Coach(game, nnet, args)
    cumulative_time = 0.0

    for i in range(1, args.numIters + 1):
        iter_start = time.time()

        single_args = dotdict(dict(args))
        single_args.numIters = 1
        single_args.checkpoint = checkpoint_dir
        coach.args = single_args
        iter_metrics = coach.learn()

        iter_time = time.time() - iter_start
        cumulative_time += iter_time

        if iter_metrics['policy_loss']:
            metrics['policy_loss'].append(iter_metrics['policy_loss'][-1])
            metrics['value_loss'].append(iter_metrics['value_loss'][-1])
            metrics['total_loss'].append(iter_metrics['total_loss'][-1])
        metrics['data_size'].append(
            iter_metrics['data_size'][-1] if iter_metrics['data_size'] else 0)
        metrics['wall_time_per_iter'].append(round(iter_time, 2))
        metrics['cumulative_wall_time'].append(round(cumulative_time, 2))

        if i % eval_interval == 0 or i == args.numIters:
            print(f"  [{label} iter {i}/{args.numIters}] Evaluating vs Random...")
            result = evaluate_vs_random(game, nnet,
                                        num_games=eval_games,
                                        num_sims=eval_sims)
            metrics['win_rate_history'].append(round(result['win_rate'], 4))
            metrics['elo_history'].append(round(result['elo'], 1))
            metrics['eval_iterations'].append(i)
            print(f"    WR={result['win_rate']:.1%}, "
                  f"Elo={result['elo']:.0f}, "
                  f"Time={cumulative_time:.1f}s")
        else:
            pi_l = metrics['policy_loss'][-1] if metrics['policy_loss'] else 0
            v_l = metrics['value_loss'][-1] if metrics['value_loss'] else 0
            print(f"  [{label} iter {i}/{args.numIters}] "
                  f"pi={pi_l:.4f}, v={v_l:.4f}, "
                  f"time={iter_time:.1f}s")

    # Save metrics
    path = os.path.join(results_dir, f'{label}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved: {path}")
    return metrics


# ================================================================
#  Phase 1: Strong STL Baselines (BT + Hex)
# ================================================================

def phase1_stl(args, results_dir):
    """Train BT and Hex independently with stronger params."""
    print("\n" + "=" * 60)
    print("  PHASE 1: Strong STL Baselines")
    print("=" * 60)

    results = {}

    for game_id, label in [('breakthrough_json', 'stl_bt'),
                            ('hex_json', 'stl_hex')]:
        print(f"\n--- {label}: {game_id} ---")
        game = get_game_by_id(game_id)
        nnet = NNetWrapper(game, game_id)
        m = train_iterative(
            game_id, nnet, args, results_dir, label,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            eval_sims=args.eval_sims,
        )
        results[label] = m

    return results


# ================================================================
#  Phase 2: MTL Joint Training (BT + Hex shared backbone)
# ================================================================

def phase2_mtl(args, results_dir):
    """Train BT + Hex jointly with multi-task learning."""
    print("\n" + "=" * 60)
    print("  PHASE 2: MTL Joint Training (BT + Hex)")
    print("=" * 60)

    games = {
        'breakthrough_json': get_game_by_id('breakthrough_json'),
        'hex_json': get_game_by_id('hex_json'),
    }

    # Use first game to init NNetWrapper (it has all policy heads)
    first_id = list(games.keys())[0]
    nnet = NNetWrapper(games[first_id], first_id)

    checkpoint_dir = os.path.join(results_dir, 'mtl_checkpoints')
    ensure_dir(checkpoint_dir)

    # Double iters since round-robin alternates: BT, Hex, BT, Hex, ...
    total_iters = args.numIters * 2
    mtl_args = dotdict(dict(args))
    mtl_args.numIters = total_iters
    mtl_args.checkpoint = checkpoint_dir

    coach = MultiTaskCoach(games, nnet, mtl_args)

    # Let MultiTaskCoach.learn() handle the full loop (correct round-robin)
    start_time = time.time()
    raw_metrics = coach.learn()
    total_time = time.time() - start_time

    # Post-training evaluation at final state
    print(f"\n  MTL training complete ({total_time:.1f}s). Evaluating...")
    eval_results = {}
    for gid in games:
        game = games[gid]
        eval_nnet = NNetWrapper(game, gid)
        eval_nnet.nnet = nnet.nnet  # share the same model
        result = evaluate_vs_random(game, eval_nnet,
                                    num_games=args.eval_games,
                                    num_sims=args.eval_sims)
        eval_results[gid] = result
        print(f"    {gid}: WR={result['win_rate']:.1%}, Elo={result['elo']:.0f}")

    # Assemble metrics
    metrics = {
        'label': 'mtl_bt_hex',
        'policy_loss': raw_metrics.get('policy_loss', []),
        'value_loss': raw_metrics.get('value_loss', []),
        'total_loss': raw_metrics.get('total_loss', []),
        'data_size': raw_metrics.get('data_size', []),
        'task_schedule': raw_metrics.get('task_schedule', []),
        'total_iters': total_iters,
        'total_time_s': round(total_time, 1),
        'eval_results': {
            gid: {
                'win_rate': round(r['win_rate'], 4),
                'elo': round(r['elo'], 1),
            } for gid, r in eval_results.items()
        },
    }

    # Save MTL model for Phase 3 transfer
    nnet.save_checkpoint(folder=checkpoint_dir, filename='mtl_final.pth.tar')

    path = os.path.join(results_dir, 'mtl_bt_hex.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved: {path}")
    return metrics


# ================================================================
#  Phase 3: Transfer Test (TicTacToe held-out game)
# ================================================================

def phase3_transfer(args, results_dir):
    """
    Transfer test: compare TicTacToe trained with MTL backbone vs from scratch.

    Group A (transfer): Load MTL backbone from Phase 2, train TTT policy head
    Group B (scratch): Train TTT from random init
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: Transfer Test (TicTacToe)")
    print("=" * 60)

    ttt_id = 'tictactoe'
    ttt_game = get_game_by_id(ttt_id)

    # --- Group B: From Scratch (run first to avoid bias) ---
    print("\n--- Group B: TicTacToe from Scratch ---")
    nnet_scratch = NNetWrapper(ttt_game, ttt_id)
    ttt_args = dotdict(dict(args))
    ttt_args.numIters = args.ttt_iters

    scratch_metrics = train_iterative(
        ttt_id, nnet_scratch, ttt_args, results_dir, 'ttt_scratch',
        eval_interval=args.ttt_eval_interval,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
    )

    # --- Group A: Transfer from MTL backbone ---
    print("\n--- Group A: TicTacToe with MTL Transfer ---")
    mtl_checkpoint = os.path.join(results_dir, 'mtl_checkpoints',
                                  'mtl_final.pth.tar')
    nnet_transfer = NNetWrapper(ttt_game, ttt_id)

    if os.path.exists(mtl_checkpoint):
        print(f"  Loading MTL backbone from: {mtl_checkpoint}")
        nnet_transfer.load_checkpoint(
            folder=os.path.join(results_dir, 'mtl_checkpoints'),
            filename='mtl_final.pth.tar',
        )
        # Reset TTT policy head (backbone + value head keep MTL knowledge)
        print("  Resetting TicTacToe policy head to random...")
        for name, param in nnet_transfer.nnet.policy_heads[ttt_id].named_parameters():
            if param.dim() >= 2:
                torch.nn.init.kaiming_normal_(param)
            else:
                torch.nn.init.zeros_(param)
        print("  Backbone + ValueHead preserved, PolicyHead reset")
    else:
        print(f"  WARNING: MTL checkpoint not found at {mtl_checkpoint}")
        print("  Running Phase 2 first, or using random init.")

    transfer_metrics = train_iterative(
        ttt_id, nnet_transfer, ttt_args, results_dir, 'ttt_transfer',
        eval_interval=args.ttt_eval_interval,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
    )

    return {'scratch': scratch_metrics, 'transfer': transfer_metrics}


# ================================================================
#  Dashboard: Plot all results
# ================================================================

def plot_dashboard(results_dir=RESULTS_DIR):
    """Generate comprehensive dashboard with all 3 phases."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n=== Generating Transfer Experiment Dashboard ===")

    # Load all available metrics
    data = {}
    for fname in os.listdir(results_dir):
        if fname.endswith('.json'):
            path = os.path.join(results_dir, fname)
            with open(path, 'r') as f:
                d = json.load(f)
            label = d.get('label', fname.replace('.json', ''))
            data[label] = d
            print(f"  Loaded: {fname} ({label})")

    if not data:
        print("  No data found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Transfer Learning Experiment: STL vs MTL vs Transfer',
                 fontsize=15, fontweight='bold')

    colors = {
        'stl_bt': '#2196F3',
        'stl_hex': '#F44336',
        'mtl_bt_hex': '#9C27B0',
        'ttt_scratch': '#FF9800',
        'ttt_transfer': '#4CAF50',
    }

    # --- Plot 1: STL Policy Loss (Phase 1) ---
    ax = axes[0, 0]
    ax.set_title('Phase 1: STL Policy Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    for label in ['stl_bt', 'stl_hex']:
        if label in data and data[label].get('policy_loss'):
            m = data[label]
            iters = range(1, len(m['policy_loss']) + 1)
            ax.plot(iters, m['policy_loss'], '-o', color=colors.get(label, 'gray'),
                    markersize=2, linewidth=1.5, label=label, alpha=0.8)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 2: STL Win Rate (Phase 1) ---
    ax = axes[0, 1]
    ax.set_title('Phase 1: STL Win Rate vs Random')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate')
    for label in ['stl_bt', 'stl_hex']:
        if label in data and data[label].get('win_rate_history'):
            m = data[label]
            ax.plot(m['eval_iterations'], m['win_rate_history'],
                    '-D', color=colors.get(label, 'gray'),
                    markersize=5, linewidth=2, label=label, alpha=0.9)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 3: MTL Loss (Phase 2) ---
    ax = axes[0, 2]
    ax.set_title('Phase 2: MTL Joint Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    if 'mtl_bt_hex' in data and data['mtl_bt_hex'].get('policy_loss'):
        m = data['mtl_bt_hex']
        iters = range(1, len(m['policy_loss']) + 1)
        ax.plot(iters, m['policy_loss'], '-o', color=colors['mtl_bt_hex'],
                markersize=2, linewidth=1.5, label='MTL (BT+Hex)', alpha=0.8)

        # Color points by task
        if m.get('task_schedule'):
            for i, task in enumerate(m['task_schedule'][:len(m['policy_loss'])]):
                c = colors['stl_bt'] if 'breakthrough' in task else colors['stl_hex']
                ax.scatter(i + 1, m['policy_loss'][i], color=c, s=15, zorder=5)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 4: MTL Final Eval + STL comparison (Phase 2) ---
    ax = axes[1, 0]
    ax.set_title('Phase 2: Final Win Rate (STL vs MTL)')
    ax.set_ylabel('Win Rate vs Random')
    game_labels = ['BT', 'Hex']
    x_pos = np.arange(len(game_labels))
    bar_w = 0.3
    # STL final win rates
    stl_wrs = []
    for label in ['stl_bt', 'stl_hex']:
        if label in data and data[label].get('win_rate_history'):
            stl_wrs.append(data[label]['win_rate_history'][-1])
        else:
            stl_wrs.append(0)
    ax.bar(x_pos - bar_w/2, stl_wrs, bar_w, label='STL', color='#2196F3', alpha=0.8)
    # MTL final win rates
    mtl_wrs = [0, 0]
    if 'mtl_bt_hex' in data and data['mtl_bt_hex'].get('eval_results'):
        er = data['mtl_bt_hex']['eval_results']
        if 'breakthrough_json' in er:
            mtl_wrs[0] = er['breakthrough_json']['win_rate']
        if 'hex_json' in er:
            mtl_wrs[1] = er['hex_json']['win_rate']
    ax.bar(x_pos + bar_w/2, mtl_wrs, bar_w, label='MTL', color='#9C27B0', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(game_labels)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    # --- Plot 5: Transfer Test - Policy Loss (Phase 3) ---
    ax = axes[1, 1]
    ax.set_title('Phase 3: TicTacToe Transfer vs Scratch')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    for label in ['ttt_scratch', 'ttt_transfer']:
        if label in data and data[label].get('policy_loss'):
            m = data[label]
            iters = range(1, len(m['policy_loss']) + 1)
            name = 'Scratch' if 'scratch' in label else 'Transfer'
            ax.plot(iters, m['policy_loss'], '-o', color=colors.get(label, 'gray'),
                    markersize=2, linewidth=1.5, label=name, alpha=0.8)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 6: Transfer Test - Win Rate (Phase 3) ---
    ax = axes[1, 2]
    ax.set_title('Phase 3: TicTacToe Win Rate (Transfer vs Scratch)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate vs Random')
    for label in ['ttt_scratch', 'ttt_transfer']:
        if label in data and data[label].get('win_rate_history'):
            m = data[label]
            name = 'Scratch' if 'scratch' in label else 'Transfer'
            ax.plot(m['eval_iterations'], m['win_rate_history'],
                    '-D', color=colors.get(label, 'gray'),
                    markersize=5, linewidth=2, label=name, alpha=0.9)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'transfer_experiment_dashboard.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Dashboard saved: {save_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("  Experiment Summary")
    print("=" * 70)
    for label, m in data.items():
        n_iters = len(m.get('policy_loss', []))
        final_pi = f"{m['policy_loss'][-1]:.4f}" if m.get('policy_loss') else 'N/A'
        final_wr = f"{m['win_rate_history'][-1]:.1%}" if m.get('win_rate_history') else 'N/A'
        total_time = f"{m['cumulative_wall_time'][-1]:.0f}s" if m.get('cumulative_wall_time') else 'N/A'
        print(f"\n  {label}:")
        print(f"    Iterations: {n_iters}, Final pi_loss: {final_pi}, "
              f"Final WR: {final_wr}, Time: {total_time}")

    # Transfer speedup analysis
    if 'ttt_scratch' in data and 'ttt_transfer' in data:
        s = data['ttt_scratch']
        t = data['ttt_transfer']
        print("\n  --- Transfer Speedup Analysis ---")
        if s.get('win_rate_history') and t.get('win_rate_history'):
            # Find first iteration where WR > 50%
            def first_above_50(m):
                for i, wr in enumerate(m['win_rate_history']):
                    if wr > 0.5:
                        return m['eval_iterations'][i]
                return None

            s50 = first_above_50(s)
            t50 = first_above_50(t)
            print(f"    Scratch: first >50% WR at iter {s50 or 'never'}")
            print(f"    Transfer: first >50% WR at iter {t50 or 'never'}")
            if s50 and t50:
                print(f"    Speedup: {s50/t50:.1f}x faster convergence")
            elif t50 and not s50:
                print(f"    Transfer reached >50%, scratch did not!")
        # Compare final loss
        if s.get('policy_loss') and t.get('policy_loss'):
            print(f"    Final pi_loss — Scratch: {s['policy_loss'][-1]:.4f}, "
                  f"Transfer: {t['policy_loss'][-1]:.4f}")


# ================================================================
#  Config Presets
# ================================================================

def get_args(quick=False):
    """Return experiment hyperparameters."""
    if quick:
        return dotdict({
            # Phase 1 & 2: BT + Hex training
            'numIters': 8,
            'numEps': 3,
            'tempThreshold': 10,
            'num_mcts_sims': 10,
            'cpuct': 1.0,
            'maxlenOfQueue': 5000,
            'eval_interval': 4,
            'eval_games': 6,
            'eval_sims': 8,
            # Phase 3: TicTacToe
            'ttt_iters': 10,
            'ttt_eval_interval': 2,
        })
    else:
        return dotdict({
            # Phase 1 & 2: BT + Hex training
            'numIters': 50,
            'numEps': 10,
            'tempThreshold': 15,
            'num_mcts_sims': 25,
            'cpuct': 1.0,
            'maxlenOfQueue': 50000,
            'eval_interval': 5,
            'eval_games': 10,
            'eval_sims': 15,
            # Phase 3: TicTacToe
            'ttt_iters': 30,
            'ttt_eval_interval': 3,
        })


# ================================================================
#  Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Transfer Learning Experiment')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (~15 min)')
    parser.add_argument('--phase', type=int, default=0,
                        help='Run only specific phase (1/2/3), 0=all')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot existing results')
    cli_args = parser.parse_args()

    if cli_args.plot_only:
        plot_dashboard(RESULTS_DIR)
        return

    args = get_args(quick=cli_args.quick)
    ensure_dir(RESULTS_DIR)

    mode = "QUICK" if cli_args.quick else "FULL"
    print(f"\n{'#' * 60}")
    print(f"  Transfer Learning Experiment ({mode})")
    print(f"  BT+Hex iters: {args.numIters}, eps: {args.numEps}, "
          f"sims: {args.num_mcts_sims}")
    print(f"  TTT iters: {args.ttt_iters}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'#' * 60}")

    total_start = time.time()
    run_phase = cli_args.phase

    if run_phase in (0, 1):
        phase1_stl(args, RESULTS_DIR)

    if run_phase in (0, 2):
        phase2_mtl(args, RESULTS_DIR)

    if run_phase in (0, 3):
        phase3_transfer(args, RESULTS_DIR)

    total_time = time.time() - total_start
    print(f"\n{'#' * 60}")
    print(f"  Experiment Complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"{'#' * 60}")

    plot_dashboard(RESULTS_DIR)


if __name__ == '__main__':
    main()
