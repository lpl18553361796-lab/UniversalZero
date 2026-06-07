import numpy as np
from game import Game, register_game

class TicTacToeGame(Game):
    """经典井字棋 (3x3)。连成三子即获胜，适合验证基础策略。"""
    def __init__(self, n=3):
        self.n = n
        self.action_mode = 'place'  # 告诉 UI 这是一个落子类游戏
        self.geometry = 'square'    # 告诉 UI 使用方形棋盘
        self.name = 'TicTacToe'     # 游戏显示名称

    def get_initial_board(self):
        return np.zeros((self.n, self.n))

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self.n * self.n

    def get_next_state(self, board, action, player):
        # 动作解包
        r, c = divmod(action, self.n)
        if board[r][c] != 0:
            # 非法落子保护，虽然 MCTS 应该避免，但逻辑上要严谨
            return board, player
        
        new_board = np.copy(board)
        new_board[r][c] = player
        return new_board, -player

    def get_valid_moves(self, board, player):
        board = np.array(board)
        valids = [0] * self.get_action_size()
        for r in range(self.n):
            for c in range(self.n):
                if board[r][c] == 0:
                    valids[r * self.n + c] = 1
        return np.array(valids)

    def get_game_ended(self, board, player):
        board = np.array(board)
        # 检查行、列、对角线
        for i in range(self.n):
            if abs(np.sum(board[i, :])) == self.n:
                return 1 if board[i, 0] == player else -1
            if abs(np.sum(board[:, i])) == self.n:
                return 1 if board[0, i] == player else -1
        
        # 主对角线
        if abs(np.sum(board.diagonal())) == self.n:
            return 1 if board[0, 0] == player else -1
        # 副对角线
        if abs(np.sum(np.fliplr(board).diagonal())) == self.n:
            return 1 if board[0, self.n-1] == player else -1

        # 平局检查 (无空位)
        if not np.any(board == 0):
            return 1e-4  # 小值表示平局
            
        return 0

    def get_canonical_form(self, board, player):
        # 通用方法：如果是后手(-1)，将棋盘翻转
        return player * board

    def string_representation(self, board):
        return board.tobytes()

# 注册到全局表
register_game('tictactoe', TicTacToeGame)
