from breakthrough import BreakthroughGame
from nnet.nnet import NNetWrapper
from coach import Coach
from utils import dotdict
import os

def main():
    print("=== AlphaZero 引擎启动 (Breakthrough) ===")
    
    # 1. 初始化
    g = BreakthroughGame()
    nnet = NNetWrapper(g)

    # 2. 配置参数 (轻量级测试配置)
    args = dotdict({
        'numIters': 2,          # 训练 2 轮
        'numEps': 2,            # 每轮自对弈 2 局
        'tempThreshold': 15,    # 前15步探索
        'num_mcts_sims': 25,    # 思考模拟 25 次
        'cpuct': 1.0,           # 探索系数
        'checkpoint': './temp/',# 保存路径
        'maxlenOfQueue': 10000, # 记忆池
    })

    # 3. 创建目录
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    # 4. 启动
    c = Coach(g, nnet, args)
    c.learn()

if __name__ == "__main__":
    main()
