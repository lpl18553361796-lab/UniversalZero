import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from breakthrough import BreakthroughGame
from mcts import MCTS
from utils import dotdict

# --- 1. 定义假网络 (Dummy Neural Network) ---
class DummyNNet:
    """
    这是一个"笨"网络，用来测试 MCTS 逻辑是否通畅。
    """
    def __init__(self, game):
        self.action_size = game.get_action_size()

    def predict(self, board):
        """
        对于任何局面，返回：
        pi: 均匀概率 (所有动作概率一样)
        v:  0 (认为胜率是五五开)
        """
        return np.ones(self.action_size) / self.action_size, 0

# --- 2. 主测试流程 ---
def main():
    print("=== MCTS 集成测试 (MCTS vs Random) ===")

    # 初始化
    game = BreakthroughGame()
    nnet = DummyNNet(game)

    # 参数配置
    args = dotdict({
        'num_mcts_sims': 50,     # 每次思考模拟 50 步
        'cpuct': 1.0             # 探索系数
    })

    # 组装 MCTS
    mcts = MCTS(game, nnet, args)

    board = game.get_initial_board()
    player = 1 # MCTS 执白(先手)
    step = 0

    while True:
        step += 1
        print(f"\n--- 第 {step} 步 (当前玩家: {player}) ---")
        game.display(board)

        # 检查全局胜负
        # get_game_ended 返回 1(玩家赢) 或 -1(玩家输)
        game_result = game.get_game_ended(board, 1)
        if game_result != 0:
            winner = 1 if game_result == 1 else -1
            winner_name = "MCTS (白)" if winner == 1 else "Random (黑)"
            print(f"\n>>> 游戏结束! 获胜者: {winner_name} <<<")
            break

        # 准备数据：转成当前玩家视角 (Canonical Form)
        canonical_board = game.get_canonical_form(board, player)

        if player == 1:
            # === MCTS 的回合 ===
            print("MCTS 正在思考...", end="", flush=True)

            # 运行 MCTS 搜索 (temp=0 代表竞技模式，选最好的)
            probs = mcts.getActionProb(canonical_board, temp=0)
            action = np.argmax(probs)
            print(f" 决定走: {action}")

        else:
            # === 随机机器人的回合 ===
            print("Random 正在乱走...", end="", flush=True)
            valid_moves = game.get_valid_moves(canonical_board, 1)
            valid_indices = np.where(valid_moves == 1)[0]

            if len(valid_indices) == 0:
                print(" 无路可走，认输。")
                break
            action = np.random.choice(valid_indices)
            print(f" 随机走: {action}")

        # === 执行动作 ===
        # 1. 在当前视角下执行动作
        next_canonical, _ = game.get_next_state(canonical_board, action, 1)

        # 2. 还原回全局视角 (关键步骤)
        # 如果当前是对手(-1)走的，next_canonical 是对手视角，
        # 我们用 get_canonical_form(..., -1) 把它翻转回来
        board = game.get_canonical_form(next_canonical, player)

        player = -player

if __name__ == "__main__":
    main()
