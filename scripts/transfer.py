import argparse
import os
import sys
import torch
import numpy as np

# --- 确保项目各层级路径正确 ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    sys.path.append(os.path.join(_project_root, "games"))
    sys.path.append(os.path.join(_project_root, "core"))
    sys.path.append(os.path.join(_project_root, "ui"))

from game import GAME_REGISTRY
from nnet.nnet import NNetWrapper
from coach import Coach
from utils import dotdict

# ────────────────────────────────────────────────────────
# 默认训练参数
# ────────────────────────────────────────────────────────

DEFAULT_ARGS = dotdict({
    'lr': 0.001,
    'numIters': 60,          # 增加到 60 轮，约 7 小时跑完
    'numEps': 40,            # 每轮40局，4个worker平分每人10局
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'num_mcts_sims': 200,    # 恢复200次模拟
    'num_workers': 3,       # 3进程：两组共6核心，完美契合 CPU
    'batch_size': 256,      # 更大的 batch，训练更稳
    'epochs': 10,
    'arenaCompare': 20,
    'cpuct': 1.0,
    'checkpoint': './experiment_results/transfer_hex/',
    'cuda': torch.cuda.is_available(),
})

# ────────────────────────────────────────────────────────
# 核心迁移函数
# ────────────────────────────────────────────────────────

def transfer(source_path, target_game_id, args, freeze_backbone=True):
    """
    将 source_path 的 backbone 迁移到 target_game_id 上训练。
    """

    # ── 步骤1：验证输入 ──────────────────────────────────
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"源模型不存在: {source_path}")

    if target_game_id not in GAME_REGISTRY:
        # 尝试触发注册
        from game import get_game_by_id
        try: get_game_by_id(target_game_id)
        except: pass

    if target_game_id not in GAME_REGISTRY:
        raise ValueError(
            f"未知游戏: {target_game_id}\n"
            f"可用游戏: {list(GAME_REGISTRY.keys())}"
        )

    print(f"\n{'='*50}")
    print(f"[START] UniversalZero Transfer Training")
    print(f"  Source     : {source_path}")
    print(f"  Target Game: {target_game_id}")
    print(f"  Freeze Backbone: {freeze_backbone}")
    print(f"{'='*50}\n")

    # ── 步骤2：初始化目标游戏 ────────────────────────────
    game_entry = GAME_REGISTRY[target_game_id]
    game = game_entry() if isinstance(game_entry, type) else game_entry

    # ── 步骤3：创建网络，加载源模型权重 ─────────────────
    # 使用之前修复过的 strict=False 加载逻辑
    nnet = NNetWrapper(game, target_game_id, args=args)
    print(f">>> 正在从 {source_path} 提取 Backbone DNA...")
    
    checkpoint_dir = os.path.dirname(source_path)
    checkpoint_file = os.path.basename(source_path)
    nnet.load_checkpoint(folder=checkpoint_dir, filename=checkpoint_file)

    print("[OK] Backbone & Value Head loaded successfully")
    print(f"[OK] {target_game_id} Policy Head reset, ready for training.")

    # ── 步骤4：冻结设置 ──────────────────────────────────
    if freeze_backbone:
        print("[FROZEN] Backbone locked - only Policy Head will train.")
        nnet.set_backbone_frozen(True)

    # ── 步骤5：保存迁移起点 ──────────────────────────────
    os.makedirs(args.checkpoint, exist_ok=True)
    init_filename = f'transfer_{target_game_id}_init.pth.tar'
    nnet.save_checkpoint(folder=args.checkpoint, filename=init_filename)
    print(f"[SAVED] Init checkpoint: {args.checkpoint}/{init_filename}")

    # ── 步骤6：在目标游戏上开始训练 ─────────────────────
    print(f"[TRAIN] Self-play on {target_game_id} starting...")
    coach = Coach(game, nnet, args)
    coach.learn()

    # ── 步骤7：保存最终模型 ──────────────────────────────
    final_filename = f'transfer_{target_game_id}_final.pth.tar'
    nnet.save_checkpoint(folder=args.checkpoint, filename=final_filename)
    print(f"[DONE] Transfer training complete!")
    print(f"[DONE] Final model: {args.checkpoint}/{final_filename}")

    return nnet

# ────────────────────────────────────────────────────────
# 命令行入口
# ────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniversalZero 经验迁移脚本')
    parser.add_argument('--source',      required=True,  help='源模型路径')
    parser.add_argument('--target_game', required=True,  help='目标游戏 ID (如 hex)')
    parser.add_argument('--iters',       type=int, default=10, help='训练轮数')
    parser.add_argument('--freeze',      action='store_true',  help='是否冻结 backbone')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    cli = parser.parse_args()

    # 组合参数，确保 copy 后依然是 dotdict
    train_args = dotdict(DEFAULT_ARGS.copy())
    train_args.numIters = cli.iters

    transfer(
        source_path=cli.source,
        target_game_id=cli.target_game,
        args=train_args,
        freeze_backbone=cli.freeze,
    )
