import os
import shutil
import time
from coach import Coach
from game import get_game_by_id
from nnet.nnet import NNetWrapper
from nnet.transfer_manager import TransferManager
from utils import dotdict

# --- 高性能实验配置 (针对 Mac 优化) ---
args = dotdict({
    'numIters': 50,          # 增加到 50 轮，足以观察到明显的收敛差异
    'numEps': 25,            # 每轮自对弈 25 局，提升经验池多样性
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'num_mcts_sims': 100,    # 增加搜索次数，使 MCTS 策略目标更精准
    'arenaCompare': 20,      # 增加评估对局数，降低胜率统计的波动噪声
    'cpuct': 1,
    'checkpoint': './experiment_results/',
    'load_model': False,
})

def run_controlled_trial(mode, target_game_id, source_path=None):
    """
    执行受控实验
    mode: 'scratch' (全随机) 或 'transfer' (加载 Backbone)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{target_game_id}_{mode}_{timestamp}"
    exp_dir = os.path.join(args.checkpoint, exp_name)
    
    print(f"\n🚀 模式: {mode.upper()} | 目标游戏: {target_game_id}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    # 1. 引擎初始化
    g = get_game_by_id(target_game_id)
    nnet = NNetWrapper(g, target_game_id)
    
    # 2. 核心变量控制：权重注入
    if mode == 'transfer':
        if not source_path or not os.path.exists(source_path):
            raise ValueError("迁移模式必须提供有效的 source_path")
        TransferManager.transfer_weights(nnet.nnet, source_path, target_game_id)
        print(f"[Info] 已注入先验知识，开始进行经验转移实验...")
    else:
        print(f"[Info] 采用白板初始化 (Tabula Rasa)，开始进行基准实验...")

    # 3. 记录实验配置以便溯源
    with open(os.path.join(exp_dir, "config.txt"), "w") as f:
        f.write(str(args))

    # 4. 启动自学习循环
    import copy
    run_args = copy.deepcopy(args)
    run_args.checkpoint = exp_dir
    coach = Coach(g, nnet, run_args)
    coach.learn()

if __name__ == "__main__":
    # --- 自动化实验流水线 ---
    
    # 步骤 1: 自动定位最新的 TTT 权重文件夹
    # (注：已将你代码里的 'ttt_scratch' 自动更正为之前实际生成的 'tictactoe_scratch' 以防止报错)
    all_results = sorted([d for d in os.listdir('./experiment_results/') if 'tictactoe_scratch' in d])
    if not all_results:
        print("请先取消 run_controlled_trial('scratch', 'tictactoe') 的注释运行一次 TTT 训练")
    else:
        latest_ttt_dir = all_results[-1]
        source_weight = os.path.join('./experiment_results/', latest_ttt_dir, "best.pth.tar")
        print(f">>> 检测到 TTT 种子权重: {source_weight}")

        # 步骤 2: 启动 Hex 基准组 (从零开始)
        # 建议在 Mac 上先跑这个，约需 1-2 小时
        run_controlled_trial('scratch', 'hex')

        # 步骤 3: 启动 Hex 迁移组 (加载 TTT Backbone)
        # 观察其初始 Loss 是否比 scratch 组低
        run_controlled_trial('transfer', 'hex', source_path=source_weight)
