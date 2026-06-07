"""
实验运行器: 训练收敛指标 + 迁移加速对比

运行方式:
    python experiments.py              # 运行完整实验 (慢)
    python experiments.py --quick      # 快速烟雾测试
    python experiments.py --plot-only  # 只绘制已有数据

实验流程:
    1. 训练 Breakthrough (收集 loss 曲线)
    2. 训练 Hex from scratch (收集 loss + 胜率)
    3. 迁移 Backbone → 训练 Hex with transfer (收集 loss + 胜率)
    4. 各阶段 Elo 评估 vs Random
    5. 生成 3 合 1 仪表盘图表
"""
import os
import sys
import json
import numpy as np
import torch

from breakthrough import BreakthroughGame
from hex_game import HexGame
from nnet.nnet import NNetWrapper
from nnet.model import UniversalNet
from game import GAME_REGISTRY
from coach import Coach
from arena import evaluate_vs_random
from utils import dotdict


RESULTS_DIR = './experiment_results/'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_breakthrough(args):
    """Phase 1: 训练 Breakthrough，收集 loss 和 Elo"""
    print("\n" + "=" * 60)
    print("  Phase 1: Training Breakthrough")
    print("=" * 60)

    game = BreakthroughGame()
    nnet = NNetWrapper(game, game_id='breakthrough')

    checkpoint_dir = os.path.join(RESULTS_DIR, 'bt_checkpoints')
    ensure_dir(checkpoint_dir)
    args.checkpoint = checkpoint_dir

    coach = Coach(game, nnet, args)
    metrics = coach.learn()

    # Elo 评估 vs Random
    print("评估 Breakthrough vs Random...")
    eval_result = evaluate_vs_random(game, nnet, num_games=args.eval_games, num_sims=args.eval_sims)
    metrics['final_elo'] = eval_result['elo']
    metrics['final_win_rate'] = eval_result['win_rate']
    print(f"  Win rate: {eval_result['win_rate']:.1%}, Elo: {eval_result['elo']:.0f}")

    # 保存指标
    metrics_path = os.path.join(RESULTS_DIR, 'metrics_breakthrough.json')
    coach.save_metrics(metrics_path)

    return nnet


def train_hex_scratch(args):
    """Phase 2: 从零开始训练 Hex"""
    print("\n" + "=" * 60)
    print("  Phase 2: Training Hex (From Scratch)")
    print("=" * 60)

    game = HexGame(n=7)
    nnet = NNetWrapper(game, game_id='hex')

    checkpoint_dir = os.path.join(RESULTS_DIR, 'hex_scratch_checkpoints')
    ensure_dir(checkpoint_dir)
    args.checkpoint = checkpoint_dir

    coach = Coach(game, nnet, args)

    # 逐迭代训练并评估
    elo_history = []
    win_rate_history = []

    for i in range(1, args.numIters + 1):
        args_single = dotdict(dict(args))
        args_single.numIters = 1
        args_single.checkpoint = checkpoint_dir

        # 复用 coach 的状态 (历史数据池)
        coach.args = args_single
        coach.learn()

        # 每迭代评估一次
        print(f"  [Scratch iter {i}] 评估 vs Random...")
        eval_result = evaluate_vs_random(game, nnet, num_games=args.eval_games, num_sims=args.eval_sims)
        elo_history.append(eval_result['elo'])
        win_rate_history.append(eval_result['win_rate'])
        print(f"    Win rate: {eval_result['win_rate']:.1%}, Elo: {eval_result['elo']:.0f}")

    # 合并指标
    coach.metrics['elo_history'] = elo_history
    coach.metrics['win_rate_history'] = win_rate_history

    metrics_path = os.path.join(RESULTS_DIR, 'metrics_hex_scratch.json')
    coach.save_metrics(metrics_path)


def train_hex_transfer(args, bt_nnet):
    """Phase 3: 迁移学习后训练 Hex"""
    print("\n" + "=" * 60)
    print("  Phase 3: Training Hex (With Backbone Transfer)")
    print("=" * 60)

    game = HexGame(n=7)
    nnet = NNetWrapper(game, game_id='hex')

    # 迁移: 从 Breakthrough 模型复制 Backbone + Value Head
    print("  执行 Backbone 迁移...")
    source_dict = bt_nnet.nnet.state_dict()
    target_dict = nnet.nnet.state_dict()

    transferred = 0
    for k, v in source_dict.items():
        # 迁移 backbone 和 value_head，跳过 policy_heads
        if k.startswith('backbone.') or k.startswith('value_head.'):
            target_dict[k] = v
            transferred += 1

    nnet.nnet.load_state_dict(target_dict)
    print(f"  已迁移 {transferred} 层权重 (Backbone + Value Head)")

    # 重置 hex policy head
    for name, param in nnet.nnet.policy_heads['hex'].named_parameters():
        if param.dim() >= 2:
            torch.nn.init.kaiming_normal_(param)
        else:
            torch.nn.init.zeros_(param)
    print("  Hex 策略头已重置为随机初始化")

    checkpoint_dir = os.path.join(RESULTS_DIR, 'hex_transfer_checkpoints')
    ensure_dir(checkpoint_dir)
    args.checkpoint = checkpoint_dir

    coach = Coach(game, nnet, args)

    # 逐迭代训练并评估
    elo_history = []
    win_rate_history = []

    for i in range(1, args.numIters + 1):
        args_single = dotdict(dict(args))
        args_single.numIters = 1
        args_single.checkpoint = checkpoint_dir

        coach.args = args_single
        coach.learn()

        print(f"  [Transfer iter {i}] 评估 vs Random...")
        eval_result = evaluate_vs_random(game, nnet, num_games=args.eval_games, num_sims=args.eval_sims)
        elo_history.append(eval_result['elo'])
        win_rate_history.append(eval_result['win_rate'])
        print(f"    Win rate: {eval_result['win_rate']:.1%}, Elo: {eval_result['elo']:.0f}")

    coach.metrics['elo_history'] = elo_history
    coach.metrics['win_rate_history'] = win_rate_history

    metrics_path = os.path.join(RESULTS_DIR, 'metrics_hex_transfer.json')
    coach.save_metrics(metrics_path)


def run_experiment(quick=False):
    """运行完整实验"""
    ensure_dir(RESULTS_DIR)

    if quick:
        # 快速烟雾测试配置
        args = dotdict({
            'numIters': 2,
            'numEps': 2,
            'tempThreshold': 5,
            'num_mcts_sims': 5,
            'cpuct': 1.0,
            'maxlenOfQueue': 2000,
            'eval_games': 6,
            'eval_sims': 5,
        })
    else:
        # 正式实验配置
        args = dotdict({
            'numIters': 5,
            'numEps': 5,
            'tempThreshold': 10,
            'num_mcts_sims': 25,
            'cpuct': 1.0,
            'maxlenOfQueue': 10000,
            'eval_games': 20,
            'eval_sims': 15,
        })

    # Phase 1: Breakthrough
    bt_args = dotdict(dict(args))
    bt_args.numIters = args.numIters
    bt_nnet = train_breakthrough(bt_args)

    # Phase 2: Hex from scratch
    hex_args = dotdict(dict(args))
    train_hex_scratch(hex_args)

    # Phase 3: Hex with transfer
    hex_t_args = dotdict(dict(args))
    train_hex_transfer(hex_t_args, bt_nnet)

    print("\n" + "=" * 60)
    print("  Experiment Complete!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("  Run 'python plot.py' to generate dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    if '--plot-only' in sys.argv:
        # 只绘图
        from plot import generate_dashboard
        generate_dashboard(RESULTS_DIR)
    elif '--quick' in sys.argv:
        run_experiment(quick=True)
    else:
        run_experiment(quick=False)
