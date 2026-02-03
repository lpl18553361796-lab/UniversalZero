import numpy as np
import time
from breakthrough import BreakthroughGame

def main():
    # 1. 初始化游戏
    game = BreakthroughGame()
    board = game.get_initial_board()
    player = 1  # 1是白方(先手), -1是黑方
    step_count = 0

    print("=== 开始 Breakthrough 自动化测试 (随机对战) ===")
    print("规则：W(白)往上攻，B(黑)往下攻。")
    print("只要有一方到达对面底线，游戏结束。")
    game.display(board)
    time.sleep(1)

    while True:
        step_count += 1
        print(f"\n----------- 第 {step_count} 步 -----------")
        print(f"当前玩家: {'白方 (W)' if player == 1 else '黑方 (B)'}")

        # --- 1. 视角转换 (关键步骤) ---
        # AI 和游戏逻辑永远只懂 "自己是1，往上攻"
        # 所以如果轮到黑方，我们要把棋盘翻转过来给他看，假装他是白方
        canonical_board = game.get_canonical_form(board, player)

        # --- 2. 获取合法动作 ---
        # 注意：这里传 1，因为在 canonical_board 上，当前玩家永远是 1
        valid_moves_mask = game.get_valid_moves(canonical_board, 1)
        
        # 检查是不是无路可走了 (被堵死)
        if np.sum(valid_moves_mask) == 0:
            print(f"玩家 {player} 无路可走，判负！")
            break

        # --- 3. 随机选择一个动作 (模拟 AI 思考) ---
        # np.where 找出所有是 1 (合法) 的动作索引
        valid_indices = np.where(valid_moves_mask == 1)[0]
        action = np.random.choice(valid_indices)

        # --- 4. 执行动作 ---
        # get_next_state 返回的是 "执行完后的 canonical 棋盘"
        next_canonical_board, _ = game.get_next_state(canonical_board, action, 1)

        # --- 5. 还原视角 (为了给人看) ---
        # 把 canonical 棋盘翻转回 全局视角 (Global View)
        # 如果是黑方走的，刚才我们翻了一次，现在要翻回来
        board = game.get_canonical_form(next_canonical_board, player)

        # --- 6. 打印全局棋盘 ---
        game.display(board)

        # --- 7. 判断胜负 ---
        # 注意：这里我们判断 canonical_board，看当前玩家有没有到达"顶端"
        game_over = game.get_game_ended(canonical_board, 1)
        
        if game_over != 0:
            # game_over=1 表示当前行动者赢了
            winner = player if game_over == 1 else -player
            print(f"\n=================================")
            print(f"GAME OVER! 获胜者: {'白方 (W)' if winner == 1 else '黑方 (B)'}")
            print(f"总步数: {step_count}")
            print(f"=================================")
            break

        # 切换玩家
        player = -player
        
        # 暂停 0.5 秒让我们看清楚
        time.sleep(0.5)

if __name__ == "__main__":
    main()
