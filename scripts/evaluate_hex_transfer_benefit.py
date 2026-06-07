import argparse
import csv
import json
import os
import sys
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.append(os.path.join(PROJECT_ROOT, "games"))
    sys.path.append(os.path.join(PROJECT_ROOT, "core"))

from core.arena import Arena, MCTSPlayer, RandomPlayer
from game import get_game_by_id
from nnet.nnet import NNetWrapper
from utils import dotdict


def load_hex_model(path):
    game = get_game_by_id('hex')
    nnet = NNetWrapper(game, 'hex', args=dotdict({}))
    nnet.load_checkpoint(folder=os.path.dirname(path), filename=os.path.basename(path))
    return game, nnet


def write_csv(report, path):
    rows = [
        {
            'benchmark': 'Transfer vs Random',
            'wins': report['transfer_vs_random']['wins'],
            'losses': report['transfer_vs_random']['losses'],
            'draws': report['transfer_vs_random']['draws'],
            'win_rate_percent': round(report['transfer_vs_random']['win_rate'] * 100, 1),
            'elo_rating': report['transfer_vs_random']['elo_rating'],
        },
        {
            'benchmark': 'Scratch vs Random',
            'wins': report['scratch_vs_random']['wins'],
            'losses': report['scratch_vs_random']['losses'],
            'draws': report['scratch_vs_random']['draws'],
            'win_rate_percent': round(report['scratch_vs_random']['win_rate'] * 100, 1),
            'elo_rating': report['scratch_vs_random']['elo_rating'],
        },
        {
            'benchmark': 'Transfer vs Scratch',
            'wins': report['transfer_vs_scratch']['wins'],
            'losses': report['transfer_vs_scratch']['losses'],
            'draws': report['transfer_vs_scratch']['draws'],
            'win_rate_percent': round(report['transfer_vs_scratch']['win_rate'] * 100, 1),
            'elo_rating': report['transfer_vs_scratch']['elo_rating'],
        },
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(report, path):
    labels = ['Transfer vs Random', 'Scratch vs Random', 'Transfer vs Scratch']
    win_rates = [
        report['transfer_vs_random']['win_rate'] * 100,
        report['scratch_vs_random']['win_rate'] * 100,
        report['transfer_vs_scratch']['win_rate'] * 100,
    ]
    elos = [
        report['transfer_vs_random']['elo_rating'],
        report['scratch_vs_random']['elo_rating'],
        report['transfer_vs_scratch']['elo_rating'],
    ]
    records = [
        f"{report['transfer_vs_random']['wins']}W-{report['transfer_vs_random']['losses']}L-{report['transfer_vs_random']['draws']}D",
        f"{report['scratch_vs_random']['wins']}W-{report['scratch_vs_random']['losses']}L-{report['scratch_vs_random']['draws']}D",
        f"{report['transfer_vs_scratch']['wins']}W-{report['transfer_vs_scratch']['losses']}L-{report['transfer_vs_scratch']['draws']}D",
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Win Rate', 'Estimated Elo'),
    )
    fig.add_trace(go.Bar(
        x=labels,
        y=win_rates,
        text=[f'{v:.1f}%' for v in win_rates],
        textposition='outside',
        customdata=records,
        hovertemplate='%{x}<br>Win rate: %{y:.1f}%<br>Record: %{customdata}<extra></extra>',
        marker_color=['#2f6f8f', '#d08c3f', '#3f8f5f'],
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=labels,
        y=elos,
        text=[f'{v:.1f}' for v in elos],
        textposition='outside',
        hovertemplate='%{x}<br>Elo: %{y:.1f}<extra></extra>',
        marker_color=['#5b6c8f', '#9a6b47', '#4f7f6b'],
    ), row=1, col=2)
    fig.update_yaxes(range=[0, 110], title='Win rate (%)', row=1, col=1)
    fig.update_yaxes(title='Elo rating', row=1, col=2)
    fig.update_layout(
        title='Hex Transfer Model Evaluation',
        template='plotly_white',
        width=1100,
        height=520,
        showlegend=False,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hex transfer model against random and scratch baselines.')
    parser.add_argument('--transfer', default=os.path.join('final_models', 'hex_transfer_60iters.pth.tar'))
    parser.add_argument('--scratch', default=os.path.join('final_models', 'hex_scratch_60iters.pth.tar'))
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--mcts-sims', type=int, default=50)
    parser.add_argument('--output', default=os.path.join('experiment_results', 'hex_transfer_benefit_eval.json'))
    parser.add_argument('--paper-dir', default=os.path.join('experiment_results', 'paper_figures'))
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if not os.path.exists(args.transfer):
        raise FileNotFoundError(f"Transfer model not found: {args.transfer}")
    if not os.path.exists(args.scratch):
        raise FileNotFoundError(f"Scratch model not found: {args.scratch}")

    game, transfer = load_hex_model(args.transfer)
    _, scratch = load_hex_model(args.scratch)

    transfer_player = MCTSPlayer(game, transfer, num_sims=args.mcts_sims)
    random_player = RandomPlayer(game)
    scratch_player = MCTSPlayer(game, scratch, num_sims=args.mcts_sims)

    started = time.time()

    random_arena = Arena(game, transfer_player, random_player)
    r_wins, r_losses, r_draws = random_arena.play_games(args.games)
    r_total = r_wins + r_losses + r_draws

    scratch_random_arena = Arena(game, scratch_player, random_player)
    s_wins, s_losses, s_draws = scratch_random_arena.play_games(args.games)
    s_total = s_wins + s_losses + s_draws

    duel_arena = Arena(game, transfer_player, scratch_player)
    d_wins, d_losses, d_draws = duel_arena.play_games(args.games)
    d_total = d_wins + d_losses + d_draws

    report = {
        'game': 'hex',
        'transfer_model': args.transfer,
        'scratch_model': args.scratch,
        'games_per_benchmark': args.games,
        'mcts_sims': args.mcts_sims,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(time.time() - started, 1),
        'transfer_vs_random': {
            'wins': r_wins,
            'losses': r_losses,
            'draws': r_draws,
            'win_rate': round(r_wins / r_total, 4) if r_total else 0.0,
            'elo_rating': round(Arena.compute_elo(r_wins, r_losses, r_draws, opponent_elo=800), 1),
        },
        'scratch_vs_random': {
            'wins': s_wins,
            'losses': s_losses,
            'draws': s_draws,
            'win_rate': round(s_wins / s_total, 4) if s_total else 0.0,
            'elo_rating': round(Arena.compute_elo(s_wins, s_losses, s_draws, opponent_elo=800), 1),
        },
        'transfer_vs_scratch': {
            'wins': d_wins,
            'losses': d_losses,
            'draws': d_draws,
            'win_rate': round(d_wins / d_total, 4) if d_total else 0.0,
            'elo_rating': round(Arena.compute_elo(d_wins, d_losses, d_draws, opponent_elo=1000), 1),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    write_csv(report, os.path.join(args.paper_dir, 'table_hex_transfer_benefit.csv'))
    write_plot(report, os.path.join(args.paper_dir, 'fig_hex_transfer_benefit.html'))

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
