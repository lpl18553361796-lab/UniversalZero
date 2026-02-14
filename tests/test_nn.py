import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakthrough import BreakthroughGame
from nnet.nnet import NNetWrapper

def main():
    print("=== 神经网络驱动测试 ===")
    game = BreakthroughGame()
    nnet = NNetWrapper(game, game_id='breakthrough')

    board = game.get_initial_board()
    print(f"正在测试推理 (Predict) ... 设备: {nnet.device}")

    pi, v = nnet.predict(board)
    print(f"策略输出长度: {len(pi)} (期望 192)")
    print(f"价值输出: {float(v[0]):.4f}")

    if len(pi) == 192 and -1 <= v[0] <= 1:
        print(">>> [SUCCESS] 神经网络运行正常！ <<<")
    else:
        print(">>> [FAIL] 输出异常 <<<")

if __name__ == "__main__":
    main()
