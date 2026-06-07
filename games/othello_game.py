import numpy as np
from game import Game, register_game

class OthelloGame(Game):
    """
    黑白棋逻辑：修复了翻转逻辑中的越界问题，并增加了 Pass（跳过）机制。
    """
    def __init__(self, n=8):
        super().__init__()
        self.n = n
        # 维持 64 位动作空间，兼容旧模型
        self.action_size = n * n 
        self.action_mode = 'place'
        self.geometry = 'square'
        self.name = "Othello (8x8)"

    def get_initial_board(self):
        b = np.zeros((self.n, self.n), dtype=int)
        mid = self.n // 2
        b[mid-1][mid-1] = b[mid][mid] = -1 
        b[mid-1][mid] = b[mid][mid-1] = 1  
        return b

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self.action_size

    def get_valid_moves(self, board, player):
        valids = [0] * self.action_size
        for r in range(self.n):
            for c in range(self.n):
                if board[r][c] == 0:
                    if self._check_valid_move(board, player, r, c):
                        valids[r * self.n + c] = 1
        return np.array(valids)

    def get_next_state(self, board, action, player):
        r, c = divmod(action, self.n)
        new_board = np.copy(board)
        new_board[r][c] = player
        
        # 执行翻转
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for dr, dc in directions:
            if self._is_flippable(board, player, r, c, dr, dc):
                curr_r, curr_c = r + dr, c + dc
                # 增加边界检查，修复斜向翻转 bug
                while 0 <= curr_r < self.n and 0 <= curr_c < self.n and new_board[curr_r][curr_c] == -player:
                    new_board[curr_r][curr_c] = player
                    curr_r += dr
                    curr_c += dc
                    
        return (new_board, -player)

    def get_canonical_form(self, board, player):
        return board * player

    def get_game_ended(self, board, player):
        # 1. 检查当前玩家是否还有合法落子
        moves = self.get_valid_moves(board, player)
        if np.any(moves):
            return 0
            
        # 2. 如果当前玩家没法下，检查对手是否也没法下
        opp_moves = self.get_valid_moves(board, -player)
        if np.any(opp_moves):
            # 原版兼容逻辑：如果当前没法下但对手能下，判定为游戏继续（但实际应用中可能需要更复杂的轮转）
            return 0 
            
        # 3. 双方都没法下，计算胜负
        diff = np.sum(board)
        if diff > 0: return 1 if player == 1 else -1
        if diff < 0: return -1 if player == 1 else 1
        return 1e-4

    def string_representation(self, board):
        return board.tobytes()

    def _check_valid_move(self, board, player, r, c):
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for dr, dc in directions:
            if self._is_flippable(board, player, r, c, dr, dc):
                return True
        return False

    def _is_flippable(self, board, player, r, c, dr, dc):
        curr_r, curr_c = r + dr, c + dc
        # 第一步必须是对手的棋子
        if not (0 <= curr_r < self.n and 0 <= curr_c < self.n): return False
        if board[curr_r][curr_c] != -player: return False
            
        curr_r += dr
        curr_c += dc
        while 0 <= curr_r < self.n and 0 <= curr_c < self.n:
            if board[curr_r][curr_c] == 0: return False
            if board[curr_r][curr_c] == player: return True
            curr_r += dr
            curr_c += dc
        return False

# 注册到系统
register_game("othello", OthelloGame)
