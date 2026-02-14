import shutil
import os
from breakthrough import BreakthroughGame
from nnet.nnet import NNetWrapper
from coach import Coach
from utils import dotdict

def main():
    print("=== 开始修复模型兼容性 (Repairing Models) ===")
    
    # 1. 清理旧的不兼容数据
    # 这些旧模型是 8x8 的，现在我们要换成 9x9 的，必须删掉
    folders = ['./temp_strong/', './temp_transfer/', './temp_hex/']
    for f in folders:
        if os.path.exists(f):
            print(f"清理过期目录: {f}")
            shutil.rmtree(f)
        os.makedirs(f)

    print("\n=== 正在生成兼容 9x9 架构的临时模型 (Breakthrough) ===")
    g = BreakthroughGame()
    nnet = NNetWrapper(g, game_id='breakthrough')
    
    # 2. 使用极速配置 (只跑 1 轮，只为了生成文件)
    # 这样你就不用等几十分钟了，马上就能看 GUI
    args = dotdict({
        'numIters': 1, 
        'numEps': 1,            # 最少局数
        'tempThreshold': 5, 
        'num_mcts_sims': 5,     # 最少思考
        'cpuct': 1.0, 
        'checkpoint': './temp_strong/', # 存到 GUI 默认读取的位置
        'maxlenOfQueue': 1000,
    })

    # 3. 运行微型训练
    c = Coach(g, nnet, args)
    c.learn()
    
    print("\n>>> [SUCCESS] 模型修复完毕！")
    print(">>> 新的 best.pth.tar 已生成，兼容当前代码。")
    print(">>> 现在请立刻运行 'python gui.py' 查看成果。")

if __name__ == "__main__":
    main()
