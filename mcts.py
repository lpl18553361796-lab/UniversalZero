import math
import numpy as np

class MCTS:
    """
    蒙特卡洛树搜索 (Monte Carlo Tree Search)
    这是 AI 的核心思考引擎。
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet  # 神经网络 (一会儿我们造个假的先测试)
        self.args = args  # 参数 (比如思考多少次)
        self.max_depth = args.get('max_depth', 100) # 默认为 100
        
        # --- 记忆库 ---
        self.Qsa = {}  # 动作价值 (胜率)
        self.Nsa = {}  # 动作访问次数
        self.Ns = {}   # 状态访问次数
        self.Ps = {}   # 策略概率 (神经网络给的)
        
        self.Es = {}   # 游戏结束标志
        self.Vs = {}   # 合法动作列表

    def getActionProb(self, canonicalBoard, temp=1):
        """
        输入：当前棋盘 (标准视角)
        输入：temp (温度参数) - temp=1 表示探索，temp=0 表示竞技(只选最好的)
        输出：动作概率分布
        """
        # 1. 疯狂思考 (Simulations)
        for i in range(self.args.num_mcts_sims):
            self.search(canonicalBoard)

        s = self.game.string_representation(canonicalBoard)
        
        # 2. 统计思考结果
        # 查看在这个局面 s 下，每一个动作 a 被模拟了多少次 (Nsa)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        # 3. 根据温度 temp 处理
        if temp == 0:
            # 竞技模式：只选访问次数最多的那个动作 (概率=1)，其他为0
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # 训练模式：根据次数归一化成概率
        # counts ** (1./temp) 用来调整分布的尖锐程度
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, depth=0):
        """
        递归搜索：Selection -> Expansion -> Simulation -> Backprop
        With depth limit to prevent stack overflow.
        """
        s = self.game.string_representation(canonicalBoard)

        # --- 0. 深度熔断保护 ---
        if depth >= self.max_depth:
            # 达到最大深度，直接返回神经网络的评估值 v
            # 这里我们不扩展节点，而是直接把当前局面交给 “直觉” 评估
            # 这是一个近似处理，防止程序崩溃
            # 注意：nnet.predict 返回 (pi, v)，我们只需要 v
            _, v = self.nnet.predict(canonicalBoard)
            return -v

        # --- 1. 检查是否结束 ---
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonicalBoard, 1)
        if self.Es[s] != 0:
            # 如果游戏结束，直接返回结果 (1赢, -1输)
            # 注意：我们的 get_game_ended 返回的是相对视角的结果
            # 1 表示当前玩家赢，-1 表示当前玩家输。所以直接返回 Es[s] 即可。
            # (在递归中，这一层返回的值会变成上一层的负数)
            return self.Es[s]

        # --- 2. 扩展新节点 (Expansion) ---
        if s not in self.Ps:
            # 这是一个新局面，问问神经网络怎么看
            # [注意] self.nnet.predict 还没实现，我们一会儿用假的代替
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            
            # 过滤非法动作
            valids = self.game.get_valid_moves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # 屏蔽非法动作
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # 归一化
            else:
                # 极罕见情况：神经网络把所有合法动作都预测为0
                # print("警报：所有合法动作概率为0，启用随机回落。")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # --- 3. 选择最佳路径 (Selection) ---
        # 使用 UCT 公式：Q + U (胜率 + 探索欲望)
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        
        # --- 4. 递归推演下一层 ---
        next_s, next_player = self.game.get_next_state(canonicalBoard, a, 1)
        # 翻转视角：我要看下一步对我来说是多少分
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s, depth + 1) # 递归调用！注意增加了 depth

        # --- 5. 回溯更新 (Backprop) ---
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
