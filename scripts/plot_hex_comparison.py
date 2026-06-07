import argparse
import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def series(metrics, key):
    rows = metrics.get('training', [])
    xs = [row.get('iteration') for row in rows]
    ys = [row.get(key) for row in rows]
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description='Plot Hex scratch vs transfer experiment results.')
    parser.add_argument('--scratch-metrics', default=os.path.join('experiment_results', 'hex_scratch', 'metrics.json'))
    parser.add_argument('--transfer-metrics', default=os.path.join('experiment_results', 'hex_transfer', 'metrics.json'))
    parser.add_argument('--duel', default=os.path.join('experiment_results', 'hex_duel_eval.json'))
    parser.add_argument('--output', default=os.path.join('experiment_results', 'hex_comparison.html'))
    args = parser.parse_args()

    scratch = load_json(args.scratch_metrics)
    transfer = load_json(args.transfer_metrics)
    duel = load_json(args.duel) if os.path.exists(args.duel) else None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Training Loss', 'Duel Result'),
        specs=[[{'type': 'xy'}, {'type': 'bar'}]],
    )

    for label, data in [('Scratch', scratch), ('Transfer', transfer)]:
        xs, ys = series(data, 'total_loss')
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=f'{label} total loss'), row=1, col=1)

    if duel:
        transfer_wr = duel.get('transfer_win_rate', 0)
        fig.add_trace(
            go.Bar(
                x=['Transfer', 'Scratch'],
                y=[transfer_wr, 1 - transfer_wr],
                name='Win rate',
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title='Hex Transfer Learning Comparison',
        yaxis_title='Loss',
        yaxis2_title='Win rate',
        template='plotly_white',
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.write_html(args.output)
    print(f"[DONE] Wrote plot: {args.output}")


if __name__ == '__main__':
    main()
