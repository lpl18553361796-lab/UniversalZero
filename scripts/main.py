import argparse
import os
import sys
import torch

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
# 默认训练参数 (从零开始训练建议轮数多一点)
# ────────────────────────────────────────────────────────

DEFAULT_ARGS = dotdict({
    'lr': 0.001,
    'numIters': 60,
    'numEps': 40,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'num_mcts_sims': 200,
    'num_workers': 3,
    'arenaCompare': 20,
    'cpuct': 1.0,
    'checkpoint': './experiment_results/scratch_hex/',
    'cuda': torch.cuda.is_available(),
})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniversalZero 从零训练脚本 (对照组)')
    parser.add_argument('--game',  required=True,       help='游戏 ID，如 hex')
    parser.add_argument('--iters', type=int, default=100, help='训练轮数')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    cli = parser.parse_args()

    # 1. 验证游戏是否存在
    if cli.game not in GAME_REGISTRY:
        from game import get_game_by_id
        try: get_game_by_id(cli.game)
        except: pass
        
    if cli.game not in GAME_REGISTRY:
        raise ValueError(f"未知游戏: {cli.game}\n可用: {list(GAME_REGISTRY.keys())}")

    # 2. 准备参数
    args = dotdict(DEFAULT_ARGS.copy())
    args.numIters = cli.iters
    args.checkpoint = os.path.join('experiment_results', f'scratch_{cli.game}')
    os.makedirs(args.checkpoint, exist_ok=True)

    # 3. 初始化游戏和网络 (从零初始化，不加载任何权重)
    game_entry = GAME_REGISTRY[cli.game]
    game = game_entry() if isinstance(game_entry, type) else game_entry
    
    nnet = NNetWrapper(game, cli.game, args=args)

    print(f"\n{'='*50}")
    print(f"🐣 UniversalZero 从零训练 (对照组) 启动")
    print(f"  游戏目标: {cli.game}")
    print(f"  计划轮数: {args.numIters}")
    print(f"  保存路径: {args.checkpoint}")
    print(f"{'='*50}\n")

    # 4. 启动训练
    # 动态设置保存路径，防止不同游戏互相覆盖
    args.checkpoint = os.path.join(args.checkpoint, f"scratch_{cli.game}")
    os.makedirs(args.checkpoint, exist_ok=True)
    
    print(f"--- Starting training for {cli.game} ---")
    print(f"--- Checkpoints will be saved to: {args.checkpoint} ---")
    
    coach = Coach(game, nnet, args)
    coach.learn()

    # 5. 保存最终模型到固定位置，方便 streamlit 加载
    final_filename = f'scratch_{cli.game}_final.pth.tar'
    nnet.save_checkpoint(folder='experiment_results', filename=final_filename)
    
    print(f"\n✅ 从零训练完成！")
    print(f"✅ 最终模型(对照组): experiment_results/{final_filename}")
