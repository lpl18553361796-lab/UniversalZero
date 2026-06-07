import argparse
import json
import os
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.append(os.path.join(PROJECT_ROOT, "games"))
    sys.path.append(os.path.join(PROJECT_ROOT, "core"))

from core.arena import Arena, MCTSPlayer
from game import get_game_by_id
from nnet.nnet import NNetWrapper
from utils import dotdict


def load_hex_model(path, game):
    nnet = NNetWrapper(game, 'hex', args=dotdict({}))
    nnet.load_checkpoint(folder=os.path.dirname(path), filename=os.path.basename(path))
    return nnet


def main():
    parser = argparse.ArgumentParser(description='Evaluate scratch Hex vs transfer Hex.')
    parser.add_argument('--scratch', default=os.path.join('experiment_results', 'hex_scratch', 'hex_scratch_final.pth.tar'))
    parser.add_argument('--transfer', default=os.path.join('experiment_results', 'hex_transfer', 'hex_transfer_final.pth.tar'))
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--mcts-sims', type=int, default=100)
    parser.add_argument('--output', default=os.path.join('experiment_results', 'hex_duel_eval.json'))
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if not os.path.exists(args.scratch):
        raise FileNotFoundError(f"Scratch model not found: {args.scratch}")
    if not os.path.exists(args.transfer):
        raise FileNotFoundError(f"Transfer model not found: {args.transfer}")

    game = get_game_by_id('hex')
    scratch = load_hex_model(args.scratch, game)
    transfer = load_hex_model(args.transfer, game)

    transfer_player = MCTSPlayer(game, transfer, num_sims=args.mcts_sims)
    scratch_player = MCTSPlayer(game, scratch, num_sims=args.mcts_sims)
    arena = Arena(game, transfer_player, scratch_player)

    started = time.time()
    wins, losses, draws = arena.play_games(args.games)
    total = wins + losses + draws
    win_rate = wins / total if total else 0.0

    report = {
        'game': 'hex',
        'model_1': 'transfer',
        'model_2': 'scratch',
        'transfer_model': args.transfer,
        'scratch_model': args.scratch,
        'games': args.games,
        'mcts_sims': args.mcts_sims,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'transfer_win_rate': round(win_rate, 4),
        'transfer_elo_vs_scratch': round(Arena.compute_elo(wins, losses, draws), 1),
        'duration_seconds': round(time.time() - started, 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
