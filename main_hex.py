from hex_game import HexGame
from nnet.nnet import NNetWrapper
from coach import Coach
from utils import dotdict
import os

def main():
    print("=== 通用性验证：启动 Hex 训练 ===")
    print("正在使用与 Breakthrough 完全相同的 AI 引擎...")
    
    g = HexGame(n=7) # 使用 7x7 棋盘
    nnet = NNetWrapper(g) # 神经网络自动适应新尺寸

    args = dotdict({
        'numIters': 2,
        'numEps': 2,
        'tempThreshold': 10,
        'num_mcts_sims': 25,
        'cpuct': 1.0,
        'checkpoint': './temp_hex/',
        'maxlenOfQueue': 5000,
    })

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    c = Coach(g, nnet, args)
    c.learn()

if __name__ == "__main__":
    main()
