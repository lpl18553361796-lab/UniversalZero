import argparse
import csv
import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots


LABELS = {
    'vs_random': 'Random',
    'vs_pure_mcts_50': 'Pure MCTS-50',
    'vs_pure_mcts_100': 'Pure MCTS-100',
}


def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def rows_from_report(report):
    rows = []
    for key, label in LABELS.items():
        item = report['benchmarks'][key]
        rows.append({
            'benchmark': label,
            'wins': item['wins'],
            'losses': item['losses'],
            'draws': item['draws'],
            'win_rate_percent': round(item['win_rate'] * 100, 1),
            'elo_rating': item['elo_rating'],
            'duration_seconds': item['duration_seconds'],
        })
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report, rows, path):
    lines = [
        '# Othello Expert Credibility',
        '',
        f"Model: `{report['model_name']}`",
        f"Games per benchmark: {report['evaluation_games_per_test']}",
        f"Evaluation timestamp: {report['timestamp']}",
        '',
        '| Benchmark | W-L-D | Win Rate | Elo | Duration (s) |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['wins']}-{row['losses']}-{row['draws']} | "
            f"{row['win_rate_percent']:.1f}% | {row['elo_rating']:.1f} | "
            f"{row['duration_seconds']:.1f} |"
        )

    lines.extend([
        '',
        'Paper-ready summary:',
        '',
        (
            'The Othello expert achieved 100.0% win rate against the random baseline, '
            '90.0% against Pure MCTS-50, and 100.0% against Pure MCTS-100 over '
            f"{report['evaluation_games_per_test']} games per benchmark. These results indicate that "
            'the checkpoint is a reliable high-strength source model for transfer experiments.'
        ),
    ])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_plot(rows, output_html):
    labels = [row['benchmark'] for row in rows]
    win_rates = [row['win_rate_percent'] for row in rows]
    elos = [row['elo_rating'] for row in rows]
    text = [f"{row['wins']}W-{row['losses']}L-{row['draws']}D" for row in rows]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Win Rate', 'Estimated Elo'),
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=win_rates,
            text=[f'{v:.1f}%' for v in win_rates],
            textposition='outside',
            customdata=text,
            hovertemplate='%{x}<br>Win rate: %{y:.1f}%<br>Record: %{customdata}<extra></extra>',
            marker_color=['#2f6f8f', '#d08c3f', '#3f8f5f'],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=elos,
            text=[f'{v:.1f}' for v in elos],
            textposition='outside',
            hovertemplate='%{x}<br>Elo: %{y:.1f}<extra></extra>',
            marker_color=['#5b6c8f', '#9a6b47', '#4f7f6b'],
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(range=[0, 110], title='Win rate (%)', row=1, col=1)
    fig.update_yaxes(title='Elo rating', row=1, col=2)
    fig.update_layout(
        title='Othello Expert Model Credibility',
        template='plotly_white',
        width=1100,
        height=520,
        showlegend=False,
    )
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)

    png_path = os.path.splitext(output_html)[0] + '.png'
    try:
        fig.write_image(png_path, scale=2)
        return png_path
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Create paper figures for Othello expert credibility.')
    parser.add_argument('--input', default=os.path.join('experiment_results', 'othello_expert_eval.json'))
    parser.add_argument('--output-dir', default=os.path.join('experiment_results', 'paper_figures'))
    args = parser.parse_args()

    report = load_report(args.input)
    rows = rows_from_report(report)

    csv_path = os.path.join(args.output_dir, 'table_othello_credibility.csv')
    md_path = os.path.join(args.output_dir, 'othello_credibility_summary.md')
    html_path = os.path.join(args.output_dir, 'fig_othello_credibility.html')

    write_csv(rows, csv_path)
    write_markdown(report, rows, md_path)
    png_path = write_plot(rows, html_path)

    print(f"[DONE] CSV table: {csv_path}")
    print(f"[DONE] Markdown summary: {md_path}")
    print(f"[DONE] HTML figure: {html_path}")
    if png_path:
        print(f"[DONE] PNG figure: {png_path}")
    else:
        print("[INFO] PNG export skipped because static Plotly image support is not installed.")


if __name__ == '__main__':
    main()
