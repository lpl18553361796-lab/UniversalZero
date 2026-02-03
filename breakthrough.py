import numpy as np
from game import Game

class BreakthroughGame(Game):
    def __init__(self):
        super().__init__()
        self.n = 8  # 8x8 棋盘
        # 动作空间设计：
        # 棋盘上 64 个格子，每个格子里的棋子有 3 种走法 (左斜, 直前, 右斜)
        # 动作总数 = 64 * 3 = 192
        self.action_size = self.n * self.n * 3 

    def get_initial_board(self):
        # 创建 8x8 空棋盘
        board = np.zeros((self.n, self.n), dtype=int)
        
        # 摆放棋子：
        # -1 (黑方) 在上面两行 (0, 1)
        board[0:2, :] = -1
        
        # 1 (白方) 在下面两行 (6, 7)
        board[6:8, :] = 1
        
        return board

    def get_board_size(self):
        return (self.n, self.n)
        
    def get_action_size(self):
        return self.action_size

    def get_valid_moves(self, board, player):
        # 返回一个长度为 192 的数组，1 表示合法，0 表示非法
        # 注意：这里我们假设 board 已经被转换过视角了，当前玩家永远是 1，永远往上走 (行号减小)。
        valids = [0] * self.action_size
        
        for r in range(self.n):
            for c in range(self.n):
                if board[r][c] == 1: # 找到我的棋子
                    # 检查 3 个方向：左斜(-1), 直前(0), 右斜(+1)
                    for dir_offset in [-1, 0, 1]:
                        target_r = r - 1        # 永远往上走
                        target_c = c + dir_offset
                        
                        # 检查是否越界
                        if 0 <= target_r < self.n and 0 <= target_c < self.n:
                            target_piece = board[target_r][target_c]
                            
                            # 规则 A: 直走 (偏移0) 只能走空位
                            if dir_offset == 0:
                                if target_piece == 0:
                                    # 计算动作ID: (当前位置 * 3) + (方向偏移映射: -1->0, 0->1, 1->2)
                                    action_idx = (r * self.n + c) * 3 + 1
                                    valids[action_idx] = 1
                            
                            # 规则 B: 斜走 (-1, 1) 可以是空位，也可以是吃子
                            else:
                                if target_piece != 1: # 只要不是自己人就能走
                                    action_idx = (r * self.n + c) * 3 + (dir_offset + 1)
                                    valids[action_idx] = 1
        return np.array(valids)

    def get_next_state(self, board, action, player):
        # 1. 解码动作
        move_dir = (action % 3) - 1     # 还原为 -1, 0, 1
        square_idx = action // 3
        src_r = square_idx // self.n
        src_c = square_idx % self.n
        
        # 2. 计算目标位置 (永远往上 -1)
        dst_r = src_r - 1
        dst_c = src_c + move_dir
        
        # 3. 执行移动
        new_board = np.copy(board)
        new_board[src_r][src_c] = 0           # 原地变空
        new_board[dst_r][dst_c] = 1           # 目标地变我 (Canonical 视角下我永远是1)
        
        # 返回后，外部框架会负责把视角翻转给对手
        return (new_board, -player)

    def get_game_ended(self, board, player):
        # 1. 检查有没有人到达顶端 (第0行)
        # 因为有视角转换，当前玩家(1)的目标永远是第0行
        if 1 in board[0, :]:
            return 1 
            
        # 2. 如果对手到达了我的底线 (第7行)，我就输了
        if -1 in board[self.n - 1, :]:
            return -1

        # 3. 检查是否被吃光
        if not np.any(board == 1): return -1
        if not np.any(board == -1): return 1

        return 0

    def get_canonical_form(self, board, player):
        # 核心黑科技：视角翻转
        if player == 1:
            return board
        else:
            # 翻转棋盘 (第0行变第7行) 并且 棋子变色 (-1变1)
            # 这样所有的逻辑都统一成了 "1 往上攻"
            return -np.flip(board, axis=0)

    def string_representation(self, board):
        return board.tobytes()

    def display(self, board):
        print("   0 1 2 3 4 5 6 7")
        print(" -----------------")
        for r in range(self.n):
            print(f"{r}|", end="")
            for c in range(self.n):
                char = "."
                if board[r][c] == 1: char = "W" # 白方
                if board[r][c] == -1: char = "B" # 黑方
                print(f" {char}", end="")
            print(" |")
        print(" -----------------")

# --- Register Game ---
from game import register_game
register_game('breakthrough', BreakthroughGame)
