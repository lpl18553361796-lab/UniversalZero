"""
train_hpc.py — UniversalZero 高性能训练启动器

目标: 在 GPU 机器 (Linux/Windows) 上以高 MCTS 模拟次数训练，
      让 AI 产生更高质量的自对弈数据（"大师级"教师信号）。

用法:
    python train_hpc.py              # 默认训练 hex
    python train_hpc.py tictactoe   # 训练 TicTacToe
"""

import os
import sys
import torch

# --- 确保项目根目录在 path 中 ---
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from coach import Coach
from game import get_game_by_id
from nnet.nnet import NNetWrapper
from utils import dotdict

# --- HPC 配置 ---
args = dotdict({
    'numIters':        100,     # 迭代轮数（每轮 = 自对弈 + 训练 + 评估）
    'numEps':          100,     # 每轮自对弈局数，覆盖更多局面
    'tempThreshold':   15,      # 前 15 步探索性采样，之后贪婪选择
    'updateThreshold': 0.55,    # 新模型需赢 55% 以上才替换旧模型
    'maxlenOfQueue':   200000,  # 经验回放池大小
    'num_mcts_sims':   400,     # 核心: 模拟次数 25→400，数据质量大幅提升
    'arenaCompare':    30,      # 评估对局数，防止弱模型偶然"上位"
    'cpuct':           1.0,
    'checkpoint':      './hpc_results/',
    'load_model':      False,
    'cuda':            torch.cuda.is_available(),  # 自动检测 NVIDIA GPU
})


def start_hpc_training(game_id: str):
    """
    启动指定游戏的 HPC 训练。

    Args:
        game_id: 'hex' 或 'tictactoe'
    """
    print(f"\n>>> 启动 HPC 训练任务: {game_id.upper()}")
    print(f">>> 计算设备: {'GPU (CUDA)' if args.cuda else 'CPU — 建议在 GPU 机器上运行!'}")
    print(f">>> MCTS 模拟次数: {args.num_mcts_sims}  |  迭代: {args.numIters}  |  每轮自对弈: {args.numEps} 局")
    print(f">>> 模型存储目录: {os.path.abspath(args.checkpoint)}\n")

    # 1. 创建存储目录
    os.makedirs(args.checkpoint, exist_ok=True)

    # 2. 初始化游戏和神经网络
    g = get_game_by_id(game_id)
    nnet = NNetWrapper(g, game_id)   # 注意: 本项目 NNetWrapper 需要 game_id 参数

    # 3. 运行 Coach 训练循环
    c = Coach(g, nnet, args)
    c.learn()

    print(f"\n✅ 训练完成！最优模型已保存至: {args.checkpoint}")


if __name__ == "__main__":
    # 支持命令行指定游戏，默认训练 hex
    target = sys.argv[1] if len(sys.argv) > 1 else 'hex'
    start_hpc_training(target)
