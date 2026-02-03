import numpy as np
from game import Game

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

class HexGame(Game):
    def __init__(self, n=7):
        super().__init__()
        self.n = n
        self.action_size = n * n

    def get_initial_board(self):
        return np.zeros((self.n, self.n), dtype=int)

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self.action_size

    def get_valid_moves(self, board, player):
        valids = [0] * self.action_size
        b = board.flatten()
        for i in range(self.action_size):
            if b[i] == 0:
                valids[i] = 1
        return np.array(valids)

    def get_next_state(self, board, action, player):
        r, c = divmod(action, self.n)
        new_board = np.copy(board)
        new_board[r][c] = player
        return (new_board, -player)

    def get_canonical_form(self, board, player):
        if player == 1:
            return board
        else:
            return -board.T

    def get_game_ended(self, board, player):
        # 使用并查集 (Union-Find) 快速判断
        # 针对当前 board 构建 UF
        # 1. 检查白方 (1) 是否连通 Top(虚拟点 N*N) -> Bottom(虚拟点 N*N+1)
        if self._check_connection_uf(board, 1):
            return 1 if player == 1 else -1

        # 2. 检查黑方 (-1) 是否连通 Left -> Right
        # 技巧：转置后变成了连通 Top -> Bottom
        if self._check_connection_uf(board.T, -1):
            return -1 if player == 1 else 1
            
        if not np.any(board == 0): return 1e-4
        return 0

    def _check_connection_uf(self, board, color):
        """
        使用并查集判断 color 是否连通上下边
        """
        N = self.n
        # 节点 0 ~ N*N-1 是棋盘点
        # 节点 N*N 是 Top 虚拟点
        # 节点 N*N+1 是 Bottom 虚拟点
        top_node = N * N
        bottom_node = N * N + 1
        
        uf = UnionFind(N * N + 2)
        
        # 1. 遍历棋盘，连接相邻的同色子
        # 并且连接边缘子到虚拟节点
        
        # 优化：只遍历有子的地方
        # board is NxN
        
        # 6 个 Hex 方向
        directions = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
        
        for r in range(N):
            for c in range(N):
                if board[r][c] == color:
                    idx = r * N + c
                    
                    # 连虚拟边
                    if r == 0: uf.union(idx, top_node)
                    if r == N - 1: uf.union(idx, bottom_node)
                    
                    # 连邻居 (只连右下方的邻居即可避免重复，但为了简单全连也可以)
                    # 为了效率，我们只看 (0,1), (1,0), (1,-1) 这三个“正向”邻居足够覆盖
                    # 不过全向也不慢
                    neighbors = [(0, 1), (1, 0), (1, -1)] 
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N:
                            if board[nr][nc] == color:
                                n_idx = nr * N + nc
                                uf.union(idx, n_idx)
                                
        return uf.find(top_node) == uf.find(bottom_node)

    def string_representation(self, board):
        return board.tobytes()

    def display(self, board):
        print("Hex Board (W:Top-Bottom, B:Left-Right):")
        print("  " + " ".join(map(str, range(self.n))))
        for r in range(self.n):
            print(" " * r + f"{r} ", end="")
            for c in range(self.n):
                ch = "."
                if board[r][c] == 1: ch = "W"
                if board[r][c] == -1: ch = "B"
                print(ch + " ", end="")
            print("")

# --- Register Game ---
from game import register_game
register_game('hex', HexGame)
