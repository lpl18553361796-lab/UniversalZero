# Othello Expert Credibility

Model: `othello_expert_8x8.pth.tar`
Games per benchmark: 100
Evaluation timestamp: 2026-06-07 18:58:48

| Benchmark | W-L-D | Win Rate | Elo | Duration (s) |
|---|---:|---:|---:|---:|
| Random | 100-0-0 | 100.0% | 1999.8 | 276.7 |
| Pure MCTS-50 | 98-2-0 | 98.0% | 1676.1 | 390.1 |
| Pure MCTS-100 | 97-3-0 | 97.0% | 1703.9 | 444.5 |

Paper-ready summary:

The Othello expert achieved 100.0% against the random baseline, 98.0% against Pure MCTS-50, and 97.0% against Pure MCTS-100 over 100 games per benchmark. These results indicate that the checkpoint is a reliable high-strength source model for transfer experiments.