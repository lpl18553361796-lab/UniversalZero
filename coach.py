import numpy as np
from collections import deque
from random import shuffle
from mcts import MCTS
import time
import os

class Coach:
    """
    总教练 (Coach)
    负责指挥：自我对弈 -> 收集数据 -> 训练 AI
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []  # 历史数据池

    def executeEpisode(self):
        """
        执行一局完整的自我对弈 (Self-Play)
        返回：本局的训练数据 [(board, pi, v), ...]
        """
        trainExamples = []
        board = self.game.get_initial_board()
        self.curPlayer = 1
        episodeStep = 0
        
        # 每一局都重置 MCTS，保证思维干净
        mcts = MCTS(self.game, self.nnet, self.args)

        while True:
            episodeStep += 1
            # 1. 获取当前视角棋盘 (Canonical Form)
            # AI 永远认为自己是 1，往上攻
            canonicalBoard = self.game.get_canonical_form(board, self.curPlayer)
            
            # 2. MCTS 思考获取策略
            # 前 15 步探索 (temp=1)，后面竞技 (temp=0)
            temp = int(episodeStep < self.args.tempThreshold)
            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            
            # 3. 收集数据：[棋盘, 当前玩家, 策略, 占位符]
            trainExamples.append([canonicalBoard, self.curPlayer, pi, None])

            # 4. 选择动作
            action = np.random.choice(len(pi), p=pi)
            
            # 5. 执行动作 (CRITICAL: 必须在 Canonical 视角下执行)
            # 因为我们的游戏逻辑 get_next_state 默认是"往上攻"
            next_state_canonical, _ = self.game.get_next_state(canonicalBoard, action, 1)
            
            # 6. 还原回全局视角
            board = self.game.get_canonical_form(next_state_canonical, self.curPlayer)
            
            # 7. 切换玩家
            self.curPlayer = -self.curPlayer

            # 8. 检查游戏是否结束 (检查绝对胜负)
            r = self.game.get_game_ended(board, 1) 

            if r != 0:
                # 游戏结束！
                # r 是绝对胜负 (1:白赢, -1:黑赢)
                # v = r * player_who_moved
                # 如果我是白方(1)且白赢(1): 1*1 = 1 (赢)
                # 如果我是白方(1)且黑赢(-1): -1*1 = -1 (输)
                return [(x[0], x[2], r * x[1]) for x in trainExamples]

    def learn(self):
        """
        核心训练循环
        """
        for i in range(1, self.args.numIters + 1):
            print(f'------ 迭代 Iteration {i}/{self.args.numIters} ------')
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            # 1. 自我对弈 (Self-Play)
            print(f"自我对弈 {self.args.numEps} 局中...")
            for eps in range(self.args.numEps):
                iterationTrainExamples += self.executeEpisode()
                print(f"\r  进度: {eps+1}/{self.args.numEps}", end="")
            print("")

            # 保存历史数据
            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > 20:
                self.trainExamplesHistory.pop(0)
            
            # 2. 准备数据
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # 3. 训练神经网络
            print(f"开始训练 (数据量: {len(trainExamples)})...")
            self.nnet.train(trainExamples)
            
            # 4. 保存模型
            print("保存模型...")
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'checkpoint_{i}.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
