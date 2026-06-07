import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# --- Ensure project root paths are resolved ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    sys.path.append(os.path.join(_project_root, "games"))
    sys.path.append(os.path.join(_project_root, "core"))

from game import get_game_by_id
from nnet.nnet import NNetWrapper
from core.coach import Coach
from core.arena import Arena, MCTSPlayer
from utils import dotdict

# ────────────────────────────────────────────────────────
# 1. Tracking Coach: Hijacking Network training to log loss
# ────────────────────────────────────────────────────────

class TrackingCoach(Coach):
    """
    Extends standard Coach to log epoch-wise training loss.
    Hijacks the NNet train method to capture policy loss, value loss, and total loss
    without modifying the core multithreading self-play architecture.
    """
    def __init__(self, game, nnet, args):
        super().__init__(game, nnet, args)
        self.loss_history = []
        
        # Hijack NNetWrapper.train
        self.orig_train = self.nnet.train
        
        def tracked_train(examples):
            loss_hist = self.orig_train(examples)
            if loss_hist and 'policy_loss' in loss_hist and len(loss_hist['policy_loss']) > 0:
                avg_pi = float(np.mean(loss_hist['policy_loss']))
                avg_v = float(np.mean(loss_hist['value_loss']))
                avg_total = float(np.mean(loss_hist['total_loss']))
                self.loss_history.append({
                    'iteration': len(self.loss_history) + 1,
                    'policy_loss': avg_pi,
                    'value_loss': avg_v,
                    'total_loss': avg_total
                })
                print(f"    [Loss Track] Iter {len(self.loss_history)} -> Pi Loss: {avg_pi:.4f}, V Loss: {avg_v:.4f}, Total: {avg_total:.4f}")
            return loss_hist
            
        self.nnet.train = tracked_train

    def save_loss_history(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        print(f"[OK] Loss history saved to: {filepath}")

# ────────────────────────────────────────────────────────
# 2. Automated Training Routines
# ────────────────────────────────────────────────────────

def run_scratch(iters=60):
    print(f"\n{'='*60}\n🐣 STARTING HEX TRAINING FROM SCRATCH (Control Group)\n{'='*60}")
    game = get_game_by_id('hex')
    
    args = dotdict({
        'lr': 0.001,
        'numIters': iters,
        'numEps': 40,
        'tempThreshold': 15,
        'updateThreshold': 0.55,
        'maxlenOfQueue': 200000,
        'num_mcts_sims': 200,
        'num_workers': 3,
        'batch_size': 256,
        'epochs': 10,
        'arenaCompare': 20,
        'cpuct': 1.0,
        'checkpoint': './experiment_results/scratch_hex/',
        'cuda': torch.cuda.is_available(),
    })
    
    nnet = NNetWrapper(game, 'hex', args=args)
    coach = TrackingCoach(game, nnet, args)
    coach.learn()
    
    # Save final model
    os.makedirs('experiment_results', exist_ok=True)
    nnet.save_checkpoint(folder='experiment_results', filename='scratch_hex_final.pth.tar')
    coach.save_loss_history('experiment_results/scratch_hex_loss.json')
    print("✅ Scratch Hex training complete!")


def run_transfer(iters=60, source_path='pretrained_models/othello_expert_8x8.pth.tar'):
    print(f"\n{'='*60}\n🧬 STARTING HEX TRANSFER TRAINING (Othello -> Hex)\n{'='*60}")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source pretrained model not found: {source_path}")
        
    game = get_game_by_id('hex')
    
    args = dotdict({
        'lr': 0.001,
        'numIters': iters,
        'numEps': 40,
        'tempThreshold': 15,
        'updateThreshold': 0.55,
        'maxlenOfQueue': 200000,
        'num_mcts_sims': 200,
        'num_workers': 3,
        'batch_size': 256,
        'epochs': 10,
        'arenaCompare': 20,
        'cpuct': 1.0,
        'checkpoint': './experiment_results/transfer_hex/',
        'cuda': torch.cuda.is_available(),
    })
    
    # Load pretrained Othello model
    nnet = NNetWrapper(game, 'hex', args=args)
    print(f"Loading weights from {source_path}...")
    nnet.load_checkpoint(folder=os.path.dirname(source_path), filename=os.path.basename(source_path))
    
    # Freeze Backbone (Transfer learning constraint: only train Policy Head)
    nnet.set_backbone_frozen(True)
    
    coach = TrackingCoach(game, nnet, args)
    coach.learn()
    
    # Save final model
    os.makedirs('experiment_results', exist_ok=True)
    nnet.save_checkpoint(folder='experiment_results', filename='transfer_hex_final.pth.tar')
    coach.save_loss_history('experiment_results/transfer_hex_loss.json')
    print("✅ Transfer Hex training complete!")

# ────────────────────────────────────────────────────────
# 3. Model Arena Duel Evaluation
# ────────────────────────────────────────────────────────

def run_arena_duel(num_games=20, num_sims=50):
    print(f"\n{'='*60}\n🥊 STARTING ARENA DUEL: TRANSFER vs SCRATCH\n{'='*60}")
    game = get_game_by_id('hex')
    
    # Load Scratch final model
    scratch_path = 'experiment_results/scratch_hex_final.pth.tar'
    if not os.path.exists(scratch_path):
        scratch_path = 'experiment_results/scratch_hex/best.pth.tar'
        if not os.path.exists(scratch_path):
            raise FileNotFoundError("Scratch Hex model not found. Run Scratch training first.")
    
    nnet_scratch = NNetWrapper(game, 'hex')
    print(f"Loading Scratch model: {scratch_path}")
    nnet_scratch.load_checkpoint(folder=os.path.dirname(scratch_path), filename=os.path.basename(scratch_path))
    p_scratch = MCTSPlayer(game, nnet_scratch, num_sims=num_sims)
    
    # Load Transfer final model
    transfer_path = 'experiment_results/transfer_hex_final.pth.tar'
    if not os.path.exists(transfer_path):
        transfer_path = 'experiment_results/transfer_hex/best.pth.tar'
        if not os.path.exists(transfer_path):
            raise FileNotFoundError("Transfer Hex model not found. Run Transfer training first.")
            
    nnet_transfer = NNetWrapper(game, 'hex')
    print(f"Loading Transfer model: {transfer_path}")
    nnet_transfer.load_checkpoint(folder=os.path.dirname(transfer_path), filename=os.path.basename(transfer_path))
    p_transfer = MCTSPlayer(game, nnet_transfer, num_sims=num_sims)
    
    # Setup Arena
    # Player 1 is Transfer, Player 2 is Scratch
    arena = Arena(game, p_transfer, p_scratch)
    print(f"Running duel ({num_games} games)...")
    
    start_time = time.time()
    t_wins, s_wins, draws = arena.play_games(num_games)
    duration = time.time() - start_time
    
    total = t_wins + s_wins + draws
    t_wr = t_wins / total if total > 0 else 0.0
    s_wr = s_wins / total if total > 0 else 0.0
    
    results = {
        'transfer_wins': t_wins,
        'scratch_wins': s_wins,
        'draws': draws,
        'transfer_win_rate': round(t_wr, 4),
        'scratch_win_rate': round(s_wr, 4),
        'total_games': num_games,
        'simulations_per_move': num_sims,
        'duration_seconds': round(duration, 1)
    }
    
    # Save Arena Results
    os.makedirs('experiment_results', exist_ok=True)
    with open('experiment_results/arena_duel_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nDuel results:")
    print(f"  Transfer Wins : {t_wins} ({t_wr:.1%})")
    print(f"  Scratch Wins  : {s_wins} ({s_wr:.1%})")
    print(f"  Draws         : {draws}")
    print(f"  Saved report to: experiment_results/arena_duel_results.json")
    return results

# ────────────────────────────────────────────────────────
# 4. Generate Graphics
# ────────────────────────────────────────────────────────

def generate_plots():
    print(f"\n{'='*60}\n📊 GENERATING LOSS AND WIN RATE GRAPHICS\n{'='*60}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    scratch_loss_path = 'experiment_results/scratch_hex_loss.json'
    transfer_loss_path = 'experiment_results/transfer_hex_loss.json'
    arena_results_path = 'experiment_results/arena_duel_results.json'
    
    # Plot 1: Loss curves comparison
    has_loss_data = False
    plt.figure(figsize=(10, 6))
    
    if os.path.exists(scratch_loss_path):
        with open(scratch_loss_path, 'r') as f:
            scratch_data = json.load(f)
        iters = [x['iteration'] for x in scratch_data]
        losses = [x['total_loss'] for x in scratch_data]
        plt.plot(iters, losses, label='Scratch Hex (Control)', color='#F44336', linewidth=2, marker='o', markersize=4)
        has_loss_data = True
        
    if os.path.exists(transfer_loss_path):
        with open(transfer_loss_path, 'r') as f:
            transfer_data = json.load(f)
        iters = [x['iteration'] for x in transfer_data]
        losses = [x['total_loss'] for x in transfer_data]
        plt.plot(iters, losses, label='Transfer (Othello -> Hex)', color='#4CAF50', linewidth=2, marker='s', markersize=4)
        has_loss_data = True
        
    if has_loss_data:
        plt.title('Convergence Rate: Scratch vs Transfer learning', fontsize=14, fontweight='bold')
        plt.xlabel('Training Iterations', fontsize=12)
        plt.ylabel('Total Model Loss (Policy + Value)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        save_path = 'experiment_results/loss_comparison.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {save_path}")
    else:
        print("[!] No loss data found. Train first to produce loss comparison.")
        
    # Plot 2: Win rate bar chart
    if os.path.exists(arena_results_path):
        with open(arena_results_path, 'r') as f:
            arena_data = json.load(f)
            
        plt.figure(figsize=(6, 6))
        labels = ['Transfer AI', 'Scratch AI']
        rates = [arena_data['transfer_win_rate'] * 100, arena_data['scratch_win_rate'] * 100]
        colors = ['#4CAF50', '#F44336']
        
        bars = plt.bar(labels, rates, color=colors, width=0.5, edgecolor='black', linewidth=1)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.title(f"Arena Duel Performance ({arena_data['total_games']} Games)", fontsize=13, fontweight='bold')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f"{height:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.axhline(y=50, color='gray', linestyle=':', alpha=0.7)
        save_path = 'experiment_results/winrate_comparison.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {save_path}")
    else:
        print("[!] No arena duel results found. Run Arena Duel first to plot win rates.")

# ────────────────────────────────────────────────────────
# 5. CLI Execution Loop
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='UniversalZero Host Experiment Runner')
    parser.add_argument('--action', choices=['scratch', 'transfer', 'duel', 'plot', 'all'], default=None,
                        help='Specific action to perform. If omitted, triggers interactive CLI.')
    parser.add_argument('--iters', type=int, default=60, help='Number of iterations (default: 60)')
    parser.add_argument('--games', type=int, default=20, help='Number of arena games (default: 20)')
    parser.add_argument('--sims', type=int, default=50, help='MCTS simulations per move (default: 50)')
    parser.add_argument('--source', type=str, default='pretrained_models/othello_expert_8x8.pth.tar',
                        help='Pretrained source model path')
    
    args = parser.parse_args()
    
    if args.action is not None:
        if args.action == 'scratch':
            run_scratch(iters=args.iters)
        elif args.action == 'transfer':
            run_transfer(iters=args.iters, source_path=args.source)
        elif args.action == 'duel':
            run_arena_duel(num_games=args.games, num_sims=args.sims)
        elif args.action == 'plot':
            generate_plots()
        elif args.action == 'all':
            run_scratch(iters=args.iters)
            run_transfer(iters=args.iters, source_path=args.source)
            run_arena_duel(num_games=args.games, num_sims=args.sims)
            generate_plots()
        return
        
    # Interactive Console
    while True:
        print(f"\n=============================================")
        print(f"     UniversalZero Host Experiment CLI       ")
        print(f"=============================================")
        print(f"  1. Run Scratch Hex (Control Group)")
        print(f"  2. Run Transfer Hex (Othello -> Hex)")
        print(f"  3. Run Arena Duel (Scratch vs Transfer)")
        print(f"  4. Generate Comparative Plots")
        print(f"  5. Run All Pipeline Tasks (1 -> 2 -> 3 -> 4)")
        print(f"  6. Exit")
        print(f"=============================================")
        choice = input("Enter choice (1-6): ").strip()
        
        try:
            if choice == '1':
                iters = int(input("Enter iterations (default 60): ") or 60)
                run_scratch(iters=iters)
            elif choice == '2':
                iters = int(input("Enter iterations (default 60): ") or 60)
                src = input(f"Enter source path (default {args.source}): ") or args.source
                run_transfer(iters=iters, source_path=src)
            elif choice == '3':
                games = int(input("Enter games to play (default 20): ") or 20)
                sims = int(input("Enter MCTS sims per move (default 50): ") or 50)
                run_arena_duel(num_games=games, num_sims=sims)
            elif choice == '4':
                generate_plots()
            elif choice == '5':
                iters = int(input("Enter iterations (default 60): ") or 60)
                games = int(input("Enter arena duel games (default 20): ") or 20)
                sims = int(input("Enter MCTS sims (default 50): ") or 50)
                run_scratch(iters=iters)
                run_transfer(iters=iters, source_path=args.source)
                run_arena_duel(num_games=games, num_sims=sims)
                generate_plots()
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("[!] Invalid choice. Try again.")
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
