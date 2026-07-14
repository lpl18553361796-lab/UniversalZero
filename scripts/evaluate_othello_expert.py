import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Ensure project root paths are resolved ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    sys.path.append(os.path.join(_project_root, "games"))
    sys.path.append(os.path.join(_project_root, "core"))

from game import get_game_by_id
from core.arena import Arena, MCTSPlayer, RandomPlayer
from utils import dotdict

# ────────────────────────────────────────────────────────
# 1. Original 512-channel OthelloNNet Architecture
# ────────────────────────────────────────────────────────
class OthelloNNet(nn.Module):
    """
    Original 512-channel neural network architecture matching the pretrained
    expert checkpoint weights.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 512, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, 3, stride=1, padding=0)  # 8x8 -> 6x6
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=1, padding=0)  # 6x6 -> 4x4
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 65)  # 64 positions + 1 Pass action
        self.fc4 = nn.Linear(512, 1)   # State evaluation

    def forward(self, x):
        # Input shape: (batch, 1, 8, 8)
        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        
        pi = self.fc3(x)
        v = torch.tanh(self.fc4(x))
        return F.log_softmax(pi, dim=1), v

# ────────────────────────────────────────────────────────
# 2. Strict NNet Wrapper for Evaluation
# ────────────────────────────────────────────────────────
class OthelloExpertWrapper:
    """
    Wrapper around OthelloNNet to load checkpoint strictly, feed unpadded
    8x8 states, and map the 65-dim policy back to the 64-dim game action space.
    """
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OthelloNNet().to(self.device)
        
        print(f"Loading checkpoint strictly from: {model_path} ...")
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def predict(self, board):
        # Input board is canonical (8x8) from the game engine
        board_t = torch.FloatTensor(board.astype(np.float32)).to(self.device).view(1, 1, 8, 8)
        
        with torch.no_grad():
            pi_log, v = self.model(board_t)
            pi = torch.exp(pi_log).cpu().numpy()[0]
            
        # Map 65-dimensional policy back to 64-dimensional game actions
        pi_64 = pi[:64]
        sum_pi = np.sum(pi_64)
        if sum_pi > 0:
            pi_64 /= sum_pi
        else:
            pi_64 = np.ones(64) / 64.0
            
        return pi_64, v.cpu().numpy()[0]

# ────────────────────────────────────────────────────────
# 3. Baseline: Uniform Network (Pure MCTS without NN guidance)
# ────────────────────────────────────────────────────────
class UniformNNet:
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, board):
        pi = np.ones(self.action_size) / self.action_size
        v = np.array([0.0])
        return pi, v

# ────────────────────────────────────────────────────────
# 4. Main Evaluation Flow
# ────────────────────────────────────────────────────────
def evaluate_expert(model_path, num_games=20):
    print(f"\n=======================================================")
    print(f"   UniversalZero: Othello Expert Strength Evaluation   ")
    print(f"=======================================================")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Pretrained model weights not found at: {model_path}")

    # Load 8x8 game and wrap model strictly
    game = get_game_by_id('othello')
    action_size = game.get_action_size()
    
    expert_nnet = OthelloExpertWrapper(model_path)
    expert_player = MCTSPlayer(game, expert_nnet, num_sims=50)
    
    results = {
        'model_name': os.path.basename(model_path),
        'expert_mcts_sims': 50,
        'evaluation_games_per_test': num_games,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmarks': {}
    }
    
    # ────────────────────────────────────────────────────────
    # Test 1: Expert AI vs Random Player
    # ────────────────────────────────────────────────────────
    print(f"\n[Test 1] Othello Expert vs Random Player ({num_games} games)...")
    random_player = RandomPlayer(game)
    arena_rnd = Arena(game, expert_player, random_player)
    
    t0 = time.time()
    wins, losses, draws = arena_rnd.play_games(num_games)
    duration = time.time() - t0
    
    total = wins + losses + draws
    wr = wins / total if total > 0 else 0.0
    elo = Arena.compute_elo(wins, losses, draws, base_elo=1000, opponent_elo=800)
    
    results['benchmarks']['vs_random'] = {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': round(wr, 4),
        'elo_rating': round(elo, 1),
        'duration_seconds': round(duration, 1)
    }
    print(f"  Result: {wins}W - {losses}L - {draws}D (Win Rate: {wr:.1%}, Est. Elo: {elo:.0f})")
    
    # ────────────────────────────────────────────────────────
    # Test 2: Expert AI vs Pure MCTS (Budget = 50 sims)
    # ────────────────────────────────────────────────────────
    pure_sims_1 = 50
    print(f"\n[Test 2] Othello Expert vs Pure MCTS (Budget: {pure_sims_1} sims, {num_games} games)...")
    pure_nnet_1 = UniformNNet(action_size)
    pure_player_1 = MCTSPlayer(game, pure_nnet_1, num_sims=pure_sims_1)
    arena_pure_1 = Arena(game, expert_player, pure_player_1)
    
    t0 = time.time()
    wins_p1, losses_p1, draws_p1 = arena_pure_1.play_games(num_games)
    duration_p1 = time.time() - t0
    
    total_p1 = wins_p1 + losses_p1 + draws_p1
    wr_p1 = wins_p1 / total_p1 if total_p1 > 0 else 0.0
    elo_p1 = Arena.compute_elo(wins_p1, losses_p1, draws_p1, base_elo=1000, opponent_elo=1000)
    
    results['benchmarks']['vs_pure_mcts_50'] = {
        'wins': wins_p1,
        'losses': losses_p1,
        'draws': draws_p1,
        'win_rate': round(wr_p1, 4),
        'elo_rating': round(elo_p1, 1),
        'duration_seconds': round(duration_p1, 1)
    }
    print(f"  Result: {wins_p1}W - {losses_p1}L - {draws_p1}D (Win Rate: {wr_p1:.1%}, Est. Elo: {elo_p1:.0f})")

    # ────────────────────────────────────────────────────────
    # Test 3: Expert AI vs Pure MCTS (Budget = 100 sims)
    # ────────────────────────────────────────────────────────
    pure_sims_2 = 100
    print(f"\n[Test 3] Othello Expert vs Pure MCTS (Budget: {pure_sims_2} sims, {num_games} games)...")
    pure_nnet_2 = UniformNNet(action_size)
    pure_player_2 = MCTSPlayer(game, pure_nnet_2, num_sims=pure_sims_2)
    arena_pure_2 = Arena(game, expert_player, pure_player_2)
    
    t0 = time.time()
    wins_p2, losses_p2, draws_p2 = arena_pure_2.play_games(num_games)
    duration_p2 = time.time() - t0
    
    total_p2 = wins_p2 + losses_p2 + draws_p2
    wr_p2 = wins_p2 / total_p2 if total_p2 > 0 else 0.0
    elo_p2 = Arena.compute_elo(wins_p2, losses_p2, draws_p2, base_elo=1000, opponent_elo=1100)
    
    results['benchmarks']['vs_pure_mcts_100'] = {
        'wins': wins_p2,
        'losses': losses_p2,
        'draws': draws_p2,
        'win_rate': round(wr_p2, 4),
        'elo_rating': round(elo_p2, 1),
        'duration_seconds': round(duration_p2, 1)
    }
    print(f"  Result: {wins_p2}W - {losses_p2}L - {draws_p2}D (Win Rate: {wr_p2:.1%}, Est. Elo: {elo_p2:.0f})")

    # Save outputs to JSON
    save_dir = 'experiment_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'othello_expert_eval.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Output Academically Formatted Summary Report
    print(f"\n=======================================================")
    print(f"              ACADEMIC EVALUATION REPORT               ")
    print(f"=======================================================")
    print(f"Target Task         : Othello (8x8 Grid)")
    print(f"Model Under Test    : {results['model_name']}")
    print(f"Evaluation Protocol : {num_games} head-to-head match-ups per test suite")
    print(f"\nQuantitative Strength Results:")
    print(f"  1. vs Random Baseline:")
    print(f"     - Win Rate     : {results['benchmarks']['vs_random']['win_rate']:.1%}")
    print(f"     - Win-Loss-Draw: {wins}W - {losses}L - {draws}D")
    print(f"     - Estimated Elo: {results['benchmarks']['vs_random']['elo_rating']:.1f}")
    print(f"  2. vs Pure MCTS (50 sims):")
    print(f"     - Win Rate     : {results['benchmarks']['vs_pure_mcts_50']['win_rate']:.1%}")
    print(f"     - Win-Loss-Draw: {wins_p1}W - {losses_p1}L - {draws_p1}D")
    print(f"     - Estimated Elo: {results['benchmarks']['vs_pure_mcts_50']['elo_rating']:.1f}")
    print(f"  3. vs Pure MCTS (100 sims):")
    print(f"     - Win Rate     : {results['benchmarks']['vs_pure_mcts_100']['win_rate']:.1%}")
    print(f"     - Win-Loss-Draw: {wins_p2}W - {losses_p2}L - {draws_p2}D")
    print(f"     - Estimated Elo: {results['benchmarks']['vs_pure_mcts_100']['elo_rating']:.1f}")
    print(f"=======================================================")
    print(f"Metrics saved successfully to: {save_path}\n")

if __name__ == '__main__':
    default_model = 'pretrained_models/othello_expert_8x8.pth.tar'
    
    import argparse
    parser = argparse.ArgumentParser(description='Othello Expert Evaluation Baseline')
    parser.add_argument('--model', type=str, default=default_model, help='Path to Othello expert model weights')
    parser.add_argument('--games', type=int, default=20, help='Number of games per test suite (default: 20)')
    args = parser.parse_args()
    
    evaluate_expert(args.model, args.games)
