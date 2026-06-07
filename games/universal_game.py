import os
import glob
import json
import numpy as np
from game import Game, register_game


class UnionFind:
    """并查集 (Union-Find) — 用于拓扑连通性判定"""
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


class UniversalGame(Game):
    """
    通用规则解释器 (Universal Rule Interpreter)

    通过解析 JSON 规则文件，动态初始化游戏元数据（棋盘几何、尺寸、动作空间），
    使其能作为标准 Game 对象接入 AlphaZero 引擎，
    并自动触发 UniversalNet 的策略头生成逻辑。

    用法:
        game = UniversalGame('rules/breakthrough_v2.json')
        print(game.get_action_size())  # 192
    """

    def __init__(self, json_path=None):
        super().__init__()
        if json_path is None:
            raise ValueError("json_path is required")

        # --- 加载规则文件 ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, json_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

        # --- 解析棋盘元数据 ---
        self.game_id = self.rules.get('id', None)
        self.name = self.rules['name']
        self.n = self.rules['board']['size']
        self.geometry = self.rules['board']['geometry']
        # 统一 geometry 别名
        if self.geometry in ('hex', 'hexagonal', 'hexagonal_axial'):
            self.geometry = 'hex'

        # --- 解析 Canonical 变换模式 ---
        # 优先使用 canonical_transform；向后兼容 canonical_flip
        self.canonical_transform = self.rules['board'].get('canonical_transform', None)
        if self.canonical_transform is None:
            if self.rules['board'].get('canonical_flip', False):
                self.canonical_transform = 'flip_vertical'
            else:
                self.canonical_transform = 'negate'

        # --- 解析邻接关系 (用于连通性判定) ---
        self.adjacency = self.rules['board'].get('adjacency', None)
        if self.adjacency is None and self.geometry == 'hex':
            self.adjacency = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]]

        # --- 解析动作模式并自动计算 action_size ---
        action_cfg = self.rules['actions']
        self.action_mode = action_cfg['mode']

        if self.action_mode == 'place':
            # 落子类 (如 Hex): action_size = n * n
            self._action_size = self.n * self.n
        elif self.action_mode == 'move_from_to':
            # 移动类 (如 Breakthrough): action_size = (n * n) * len(move_vectors)
            self.move_vectors = action_cfg['move_vectors']
            self._action_size = (self.n * self.n) * len(self.move_vectors)
        else:
            raise ValueError(f"Unknown action mode: '{self.action_mode}'")

        # --- 解析结束条件 ---
        self.end_condition = self.rules.get('end_condition', {})

        # --- 解析初始棋子布局 ---
        self.initial_setup = self.rules.get('pieces', {}).get('initial_rows', {})

    # ================================================================
    #  Game 接口实现 — 元数据查询 (已完成)
    # ================================================================

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self._action_size

    # ================================================================
    #  动作 ID 转换矩阵 (Action ID Codec)
    # ================================================================

    def _id_to_move(self, action_id):
        """
        将一维动作 ID 解码为 (source_r, source_c, vec_idx)。

        编码公式: action_id = (r * n + c) * num_dirs + vec_idx
        """
        num_dirs = len(self.move_vectors)
        vec_idx = action_id % num_dirs
        source_idx = action_id // num_dirs
        source_r = source_idx // self.n
        source_c = source_idx % self.n
        return source_r, source_c, vec_idx

    def _move_to_id(self, r, c, vec_idx):
        """
        将 (source_r, source_c, vec_idx) 编码为一维动作 ID。
        """
        num_dirs = len(self.move_vectors)
        return (r * self.n + c) * num_dirs + vec_idx

    # ================================================================
    #  Game 接口实现 — 规则引擎
    # ================================================================

    def get_initial_board(self):
        board = np.zeros((self.n, self.n), dtype=int)
        for row in self.initial_setup.get('player_1', []):
            board[row, :] = 1
        for row in self.initial_setup.get('player_neg1', []):
            board[row, :] = -1
        return board

    def get_valid_moves(self, board, player):
        """
        根据 JSON 规则中的 move_vectors 和 capture_rules，
        在 Canonical Form 下生成所有合法动作的二进制向量。
        """
        valids = np.zeros(self._action_size, dtype=int)

        if self.action_mode == 'place':
            # 落子类: 所有空位合法
            flat = board.flatten()
            for i in range(self._action_size):
                if flat[i] == 0:
                    valids[i] = 1

        elif self.action_mode == 'move_from_to':
            capture_cfg = self.rules['actions'].get('capture_rules', {})
            straight_capture = capture_cfg.get('straight_capture', True)
            diagonal_capture = capture_cfg.get('diagonal_capture_enemy_only', False)

            for r in range(self.n):
                for c in range(self.n):
                    if board[r][c] != 1:
                        continue  # canonical: 当前玩家永远是 1

                    for vec_idx, (dr, dc) in enumerate(self.move_vectors):
                        tr, tc = r + dr, c + dc

                        # 边界检查
                        if not (0 <= tr < self.n and 0 <= tc < self.n):
                            continue

                        target = board[tr][tc]
                        is_straight = (dr == 0 or dc == 0)

                        if is_straight:
                            # 直线移动: 根据 straight_capture 决定能否吃子
                            if not straight_capture and target != 0:
                                continue  # 不允许吃子，目标必须为空
                            if target == 1:
                                continue  # 不能踩自己人
                        else:
                            # 斜线移动: 根据 diagonal_capture 决定规则
                            if diagonal_capture:
                                if target == 1:
                                    continue  # 不能吃自己人
                            else:
                                if target != 0:
                                    continue  # 不允许吃子，目标必须为空

                        valids[self._move_to_id(r, c, vec_idx)] = 1

        return valids

    def get_next_state(self, board, action, player):
        """
        在副本棋盘上执行动作，返回 (new_board, -player)。
        """
        new_board = np.copy(board)

        if self.action_mode == 'place':
            r, c = divmod(action, self.n)
            new_board[r][c] = player

        elif self.action_mode == 'move_from_to':
            src_r, src_c, vec_idx = self._id_to_move(action)
            dr, dc = self.move_vectors[vec_idx]
            dst_r, dst_c = src_r + dr, src_c + dc
            new_board[src_r][src_c] = 0
            new_board[dst_r][dst_c] = 1  # canonical: 当前玩家永远是 1

        return (new_board, -player)

    # ================================================================
    #  胜负判定引擎 (End Condition Evaluator)
    # ================================================================

    def _get_neighbors(self, r, c):
        """
        返回 (r, c) 的所有合法相邻坐标列表。

        根据 geometry 决定邻接关系:
            square — 8 方向 (King moves)
            hex    — 6 方向 (由 self.adjacency 定义)
        """
        neighbors = []
        if self.geometry == 'hex' and self.adjacency:
            dirs = self.adjacency
        else:
            # square: 8 方向
            dirs = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                neighbors.append((nr, nc))
        return neighbors

    def _check_n_in_a_row(self, board, target_n):
        """
        检查是否有玩家达成 target_n 连子。

        检查方向: 水平、垂直、主对角线、副对角线。

        Returns:
            1  — Player 1 达成 n 连
           -1  — Player -1 达成 n 连
            0  — 无人达成
        """
        N = self.n
        # 四个方向: (dr, dc)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(N):
            for c in range(N):
                if board[r][c] == 0:
                    continue
                color = board[r][c]
                for dr, dc in directions:
                    # 检查从 (r,c) 出发沿 (dr,dc) 是否有 target_n 个连续同色
                    end_r = r + (target_n - 1) * dr
                    end_c = c + (target_n - 1) * dc
                    if not (0 <= end_r < N and 0 <= end_c < N):
                        continue
                    count = 0
                    for step in range(target_n):
                        if board[r + step * dr][c + step * dc] == color:
                            count += 1
                        else:
                            break
                    if count == target_n:
                        return int(color)

        return 0

    def _check_reach_rank(self, board):
        """
        达阵判定: 检查是否有玩家到达目标行。

        Returns:
            1  — Player 1 (piece value 1) 到达 row 0 (攻方底线)
           -1  — Player -1 (piece value -1) 到达 row n-1
            0  — 无人达阵
        """
        if 1 in board[0, :]:
            return 1
        if -1 in board[self.n - 1, :]:
            return -1
        return 0

    def _check_connectivity(self, board, color):
        """
        使用并查集判断 color 是否连通上下边界 (Top ↔ Bottom)。

        虚拟节点:
            top_node    = n * n
            bottom_node = n * n + 1
        """
        N = self.n
        top_node = N * N
        bottom_node = N * N + 1
        uf = UnionFind(N * N + 2)

        adjacency = self.adjacency

        for r in range(N):
            for c in range(N):
                if board[r][c] == color:
                    idx = r * N + c
                    # 连接虚拟边界节点
                    if r == 0:
                        uf.union(idx, top_node)
                    if r == N - 1:
                        uf.union(idx, bottom_node)
                    # 连接相邻同色棋子
                    for dr, dc in adjacency:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N:
                            if board[nr][nc] == color:
                                uf.union(idx, nr * N + nc)

        return uf.find(top_node) == uf.find(bottom_node)

    def get_game_ended(self, board, player):
        """
        通用终局判定路由器。

        根据 JSON 中的 end_condition.type 分发到对应的检测子模块，
        返回值遵循 AlphaZero 规范:
            1    — player 参数所代表的玩家获胜
           -1    — player 参数所代表的玩家失败
            1e-4 — 平局
            0    — 游戏未结束
        """
        end_type = self.end_condition.get('type', None)

        if end_type == 'reach_rank':
            # --- 达阵判定 ---
            winner = self._check_reach_rank(board)
            if winner != 0:
                return winner * player  # 转换为 player 视角的奖励

            # --- 歼灭判定 ---
            if self.end_condition.get('elimination', False):
                if not np.any(board == 1):
                    return -1 * player   # Player 1 被歼灭
                if not np.any(board == -1):
                    return 1 * player    # Player -1 被歼灭

            return 0

        elif end_type == 'n_in_a_row':
            # --- N 连子判定 ---
            target_n = self.end_condition.get('n', 3)
            winner = self._check_n_in_a_row(board, target_n)
            if winner != 0:
                return winner * player

            # --- 平局: 棋盘满且无胜者 ---
            if not np.any(board == 0):
                return 1e-4

            return 0

        elif end_type == 'connectivity':
            # --- Player 1: 垂直连通 (Top ↔ Bottom) ---
            if self._check_connectivity(board, 1):
                return 1 * player

            # --- Player -1: 水平连通 (Left ↔ Right) ---
            # 转置棋盘后检查垂直连通 = 原棋盘的水平连通
            if self._check_connectivity(board.T, -1):
                return -1 * player

            # --- 平局: 棋盘满且无胜者 ---
            if not np.any(board == 0):
                return 1e-4

            return 0

        return 0

    def get_canonical_form(self, board, player):
        """
        视角变换: 将棋盘转换为当前玩家的标准视角。

        支持的变换模式 (由 JSON board.canonical_transform 指定):
            flip_vertical — 取反 + 垂直翻转 (Breakthrough)
            transpose     — 取反 + 转置 (Hex)
            negate        — 仅取反 (默认)
        """
        if player == 1:
            return board
        if self.canonical_transform == 'flip_vertical':
            return -np.flip(board, axis=0)
        elif self.canonical_transform == 'transpose':
            return -board.T
        return -board

    def string_representation(self, board):
        return board.tobytes()

    def display(self, board):
        n = self.n
        print(f"=== {self.name} ({n}x{n}) ===")
        header = "  " + " ".join(str(c) for c in range(n))
        print(header)
        print("  " + "-" * (2 * n - 1))
        for r in range(n):
            row_str = f"{r}|"
            for c in range(n):
                ch = "."
                if board[r][c] == 1:
                    ch = "W"
                elif board[r][c] == -1:
                    ch = "B"
                row_str += f" {ch}"
            row_str += " |"
            print(row_str)
        print("  " + "-" * (2 * n - 1))


# --- 自动发现并注册 rules/ 目录下的所有 JSON 游戏 ---
_base = os.path.dirname(os.path.abspath(__file__))
_rules_dir = os.path.join(_base, 'rules')

if os.path.isdir(_rules_dir):
    for _json_path in sorted(glob.glob(os.path.join(_rules_dir, '*.json'))):
        try:
            _rel_path = os.path.relpath(_json_path, _base)
            _game = UniversalGame(_rel_path)
            # 优先使用 JSON 中的 id 字段，否则用文件名 (去扩展名)
            _game_id = _game.game_id or os.path.splitext(os.path.basename(_json_path))[0]
            register_game(_game_id, _game)
        except Exception as e:
            print(f"[UniversalGame] Warning: failed to load {_json_path}: {e}")
