# Experimental Setup

## 1. Games and State Representation

The experiments were conducted on two deterministic, two-player,
zero-sum board games: Othello and Hex. Othello used an 8 x 8 board with
64 board-position actions, while Hex used a 7 x 7 board with 49 actions.
For the UniversalZero network, game states were centered and zero-padded
to a common 9 x 9 spatial representation. This allowed both games to use
the same convolutional feature-extraction architecture while retaining
game-specific policy output dimensions.

The UniversalZero model consisted of a shared residual convolutional
backbone, game-specific policy heads, and a shared value head. The Hex
policy head produced a 49-dimensional action distribution. Training used
the AlphaZero-style combination of a policy loss and a value loss:

```text
total_loss = policy_loss + value_loss
```

Self-play action targets were obtained from MCTS visit distributions, and
value targets were derived from final game outcomes.

## 2. Othello Expert Model

The source expert was the pretrained model:

```text
pretrained_models/othello_expert_8x8.pth.tar
```

It used the original Othello architecture with four convolutional layers,
512 feature channels, two fully connected layers, a 65-dimensional policy
output, and a scalar value output. During evaluation, the additional pass
output was removed and the remaining 64 policy probabilities were
renormalized to match the current Othello environment.

For Othello-to-Hex initialization, compatible early convolutional and
batch-normalization parameters were mapped into the UniversalZero
backbone. Because the source model used 512 channels and the target model
used 64 channels, compatible tensors were transferred through channel
slicing. Parameters without compatible source counterparts, including
the Hex policy head, were initialized normally.

## 3. Othello Expert Credibility Evaluation

The Othello expert was evaluated against three baselines:

1. Random Agent.
2. Pure MCTS with 50 simulations per move.
3. Pure MCTS with 100 simulations per move.

The expert itself used neural-network-guided MCTS with 50 simulations per
move. Each benchmark contained 100 games, with the starting-player
assignment balanced across the two agents. A fresh MCTS search object was
created for every game to prevent search-tree statistics from leaking
between games.

The random seed was fixed to:

```text
20260607
```

Win-rate uncertainty was reported using a 95% Wilson confidence interval.
The formal evaluation results were stored in:

```text
experiment_results/othello_expert_eval_robust.json
```

The resulting expert-model performance was:

| Evaluation opponent | Wins | Losses | Draws | Win rate | 95% confidence interval | Estimated Elo |
|---|---:|---:|---:|---:|---:|---:|
| Random Agent | 100 | 0 | 0 | 100.0% | 96.3%-100.0% | 1999.8 |
| Pure MCTS, 50 simulations | 98 | 2 | 0 | 98.0% | 93.0%-99.5% | 1676.1 |
| Pure MCTS, 100 simulations | 97 | 3 | 0 | 97.0% | 91.6%-99.0% | 1703.9 |

The results indicate that the expert consistently outperformed all three
baselines. The small decrease from 98.0% against 50-simulation Pure MCTS
to 97.0% against 100-simulation Pure MCTS is consistent with the increase
in opponent search budget. However, the overlapping confidence intervals
do not support a claim of a statistically meaningful difference between
the two Pure MCTS conditions.

The earlier 10-game Othello evaluation was treated as preliminary and was
not used as the final credibility result.

## 4. Hex Transfer-versus-Scratch Strength Experiment

Two Hex agents were trained under the same training budget:

- **Scratch model:** UniversalZero initialized randomly.
- **Transfer model:** UniversalZero initialized using the Othello-to-Hex
  weight-transfer procedure.

The shared settings were:

| Parameter | Value |
|---|---:|
| Training iterations | 10 |
| Self-play episodes per iteration | 20 |
| MCTS simulations per move | 50 |
| Training workers | 3 |
| Batch size | 128 |
| Epochs per iteration | 5 |
| Learning rate | 0.001 |
| PUCT constant | 1.0 |
| Temperature threshold | 15 |

For this experiment, the transferred backbone and shared value head were
frozen, and the Hex policy head was optimized using Hex self-play data.
The Scratch model retained the same UniversalZero architecture but was
trained from random initialization.

The final agents were evaluated in three matchups:

1. Transfer versus Random.
2. Scratch versus Random.
3. Transfer versus Scratch.

Each matchup contained 100 games with alternating starting-player roles.
Both neural agents used 50 MCTS simulations per move. Results were stored
in:

```text
experiment_results/hex_transfer_benefit_fixed_quick_eval.json
```

The corresponding training logs were stored in:

```text
experiment_results/hex_transfer_fixed_quick/metrics.json
experiment_results/hex_scratch_fixed_quick/metrics.json
```

## 5. Hex Convergence Experiment

A separate experiment examined whether transfer initialization provided
faster early-stage optimization under an aligned Hex training budget.
Scratch and Transfer models used the same environment, UniversalZero
architecture, self-play budget, MCTS settings, optimizer settings, and
random seed.

The shared configuration was:

| Parameter | Value |
|---|---:|
| Training iterations | 20 |
| Recorded stages | 0, 5, 10, 15, 20 |
| Self-play episodes per iteration | 20 |
| Training MCTS simulations | 50 |
| Evaluation games per matchup and stage | 20 |
| Evaluation MCTS simulations | 50 |
| Training workers | 3 |
| Batch size | 128 |
| Epochs per iteration | 5 |
| Learning rate | 0.001 |
| PUCT constant | 1.0 |
| Temperature threshold | 15 |
| Random seed | 20260606 |

Unlike the frozen transfer-versus-scratch strength experiment, the
Transfer model used full fine-tuning in this convergence experiment.
Therefore, both the transferred parameters and the Hex-specific policy
head remained trainable. This setting was selected to compare the effect
of initialization while keeping the trainable architecture aligned.

Checkpoints were saved for both models at stages 0, 5, 10, 15, and 20.
At every stage, the following evaluations were performed:

1. Scratch versus Random.
2. Transfer versus Random.
3. Transfer versus Scratch.

Policy loss, value loss, and total loss were recorded at each training
iteration. The paper figure retains only the total-loss curves because
the Random baseline saturated rapidly and the stage-wise win rates
displayed substantial variance.

The convergence outputs were stored in:

```text
experiment_results/hex_transfer_convergence/hex_transfer_convergence.csv
experiment_results/hex_transfer_convergence/hex_transfer_convergence_details.json
experiment_results/hex_transfer_convergence/hex_transfer_loss_convergence.png
```

This experiment was designed to observe whether transfer initialization
provided faster early-stage improvement. It was not designed to establish
statistical significance, and the results should not be described as
conclusive evidence of faster convergence.

## 6. Raspberry Pi Deployment and Resource Monitoring

The inference application was deployed on a Raspberry Pi with
approximately 2 GB of system memory. PyTorch 2.6.0+debian was used on the
device. System utilization was sampled continuously during a five-minute
monitoring interval:

```text
2026-06-04 15:18:04 to 2026-06-04 15:22:59
```

The monitoring run contained 50 samples. Each sample recorded:

- CPU utilization.
- Memory utilization.
- Used, available, and total memory.
- CPU temperature.

The raw and summarized measurements were stored in:

```text
pi_tools/logs/system_usage.csv
pi_tools/logs/system_usage_latest.json
pi_tools/logs/system_usage_summary_table.md
pi_tools/logs/system_usage_summary_table.tex
```

This resource experiment characterizes observed system-level utilization
during the monitored application run. It does not measure training
performance on the Raspberry Pi, because model training was not intended
to be performed on the device.

## 7. Reproducibility and Reporting Policy

Where supported by the experiment, Python, NumPy, and PyTorch random seeds
were fixed. The convergence experiment also derived deterministic worker
seeds from the experiment seed and training iteration. Starting-player
roles were balanced during head-to-head evaluation.

Smoke tests were used only to verify code execution, checkpoint creation,
CSV generation, and figure export. Results under directories containing
`smoke` were excluded from the formal experimental evidence.

The formal experiments reported in the paper are:

1. Robust Othello expert credibility evaluation.
2. Frozen Othello-to-Hex transfer versus Scratch strength evaluation.
3. Full-fine-tuning Hex convergence comparison.
4. Raspberry Pi resource-utilization monitoring.
