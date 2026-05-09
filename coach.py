import numpy as np
from collections import deque
from random import shuffle
from mcts import MCTS
from arena import MultiTaskArena
import time
import os
import json


class Coach:
    """
    总教练 (Coach)
    负责指挥：自我对弈 -> 收集数据 -> 训练 AI
    支持训练指标收集 (loss 曲线、胜率评估)
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.game_id = getattr(nnet, 'game_id', 'unknown')
        self.trainExamplesHistory = []  # 历史数据池

        # 训练指标
        self.metrics = {
            'policy_loss': [],     # 每迭代的平均 policy loss
            'value_loss': [],      # 每迭代的平均 value loss
            'total_loss': [],      # 每迭代的平均 total loss
            'data_size': [],       # 每迭代的训练数据量
            'game_id': self.game_id,
        }

    def executeEpisode(self):
        """
        执行一局完整的自我对弈 (Self-Play)
        返回：本局的训练数据 [(board, pi, v, game_id), ...]
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

            # 3. 收集数据：[棋盘, 当前玩家, 策略, 占位符, 游戏ID]
            trainExamples.append([canonicalBoard, self.curPlayer, pi, None, self.game_id])

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
                # v = r * player_who_moved, 第五位 game_id 保持不变
                return [(x[0], x[2], r * x[1], x[4]) for x in trainExamples]

    def learn(self):
        """
        核心训练循环

        Returns:
            dict: 训练指标 (metrics)
        """
        for i in range(1, self.args.numIters + 1):
            print(f'------ Iteration {i}/{self.args.numIters} ------')
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            # 1. 自我对弈 (Self-Play)
            print(f"Self-play {self.args.numEps} episodes...")
            for eps in range(self.args.numEps):
                iterationTrainExamples += self.executeEpisode()
                print(f"\r  Progress: {eps+1}/{self.args.numEps}", end="")
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

            # 3. 训练神经网络 (收集 loss)
            print(f"Training (data size: {len(trainExamples)})...")
            loss_data = self.nnet.train(trainExamples)

            # 记录指标
            if loss_data['total_loss']:
                self.metrics['policy_loss'].append(np.mean(loss_data['policy_loss']))
                self.metrics['value_loss'].append(np.mean(loss_data['value_loss']))
                self.metrics['total_loss'].append(np.mean(loss_data['total_loss']))
            self.metrics['data_size'].append(len(trainExamples))

            if self.metrics['total_loss']:
                avg_loss = self.metrics['total_loss'][-1]
                print(f"  Loss: {avg_loss:.4f} "
                      f"(pi: {self.metrics['policy_loss'][-1]:.4f}, "
                      f"v: {self.metrics['value_loss'][-1]:.4f})")
                
                # 追加保存 loss 到 loss.txt
                loss_path = os.path.join(self.args.checkpoint, "loss.txt")
                with open(loss_path, "a") as f:
                    f.write(f"{self.metrics['policy_loss'][-1]:.6f}\n")
            else:
                print(f"  Loss: N/A (insufficient data)")

            # 4. 保存模型
            print("Saving model...")
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'checkpoint_{i}.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

        return self.metrics

    def save_metrics(self, filepath):
        """将训练指标保存为 JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved: {filepath}")

    @staticmethod
    def load_metrics(filepath):
        """从 JSON 加载训练指标"""
        with open(filepath, 'r') as f:
            return json.load(f)


class MultiTaskCoach:
    """
    多任务总教练 (Multi-Task Coach)

    管理多个游戏的轮换自对弈，将所有经验汇入同一个缓冲区，
    并以 4-tuple (board, pi, v, game_id) 格式喂给 NNetWrapper 的
    Task-Aware Training 逻辑。

    轮换策略: 每个迭代按顺序切换游戏 —
        Iteration 1 -> Game A 自对弈
        Iteration 2 -> Game B 自对弈
        Iteration 3 -> Game A 自对弈 ...

    所有迭代共享同一个 trainExamplesHistory 和 NNetWrapper，
    梯度同时流经 Backbone + 各 PolicyHead + ValueHead。
    """

    def __init__(self, games, nnet, args):
        """
        Args:
            games: dict {game_id: game_instance}
            nnet:  NNetWrapper (共享，包含所有游戏的策略头)
            args:  训练参数
        """
        self.games = games
        self.game_ids = list(games.keys())
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []

        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'data_size': [],
            'task_schedule': [],   # 每迭代使用的 game_id
        }

    def executeEpisode(self, game, game_id):
        """对指定游戏执行一局自对弈，返回 4-tuple 列表"""
        trainExamples = []
        board = game.get_initial_board()
        curPlayer = 1
        episodeStep = 0

        # Switch nnet to target game context for correct predict routing
        orig_id = self.nnet.game_id
        orig_gx, orig_gy = self.nnet.game_x, self.nnet.game_y
        self.nnet.game_id = game_id
        gx, gy = game.get_board_size()
        self.nnet.game_x, self.nnet.game_y = gx, gy

        mcts = MCTS(game, self.nnet, self.args)

        while True:
            episodeStep += 1
            canonicalBoard = game.get_canonical_form(board, curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            pi = mcts.getActionProb(canonicalBoard, temp=temp)

            trainExamples.append([canonicalBoard, curPlayer, pi, None, game_id])

            action = np.random.choice(len(pi), p=pi)
            next_state_canonical, _ = game.get_next_state(canonicalBoard, action, 1)
            board = game.get_canonical_form(next_state_canonical, curPlayer)
            curPlayer = -curPlayer

            r = game.get_game_ended(board, 1)
            if r != 0:
                # Restore original nnet context
                self.nnet.game_id = orig_id
                self.nnet.game_x, self.nnet.game_y = orig_gx, orig_gy
                return [(x[0], x[2], r * x[1], x[4]) for x in trainExamples]

    def learn(self):
        """
        多任务训练主循环

        每个迭代:
            1. 按轮换顺序选择一个游戏
            2. 用该游戏执行 numEps 局自对弈
            3. 将带 game_id 标签的数据加入共享缓冲区
            4. 用混合缓冲区训练 (Task-Aware Training)

        Returns:
            dict: 训练指标
        """
        for i in range(1, self.args.numIters + 1):
            # 轮换选择游戏
            task_idx = (i - 1) % len(self.game_ids)
            current_game_id = self.game_ids[task_idx]
            current_game = self.games[current_game_id]

            print(f"------ Iteration {i}/{self.args.numIters} "
                  f"[Task: {current_game_id}] ------")

            iterationExamples = deque([], maxlen=self.args.maxlenOfQueue)

            # 1. 自对弈 (Self-Play)
            print(f"  Self-play {self.args.numEps} episodes ({current_game_id})...")
            for eps in range(self.args.numEps):
                iterationExamples += self.executeEpisode(
                    current_game, current_game_id)
                print(f"\r    Progress: {eps+1}/{self.args.numEps}", end="")
            print("")

            # 2. 加入共享历史缓冲区
            self.trainExamplesHistory.append(iterationExamples)
            if len(self.trainExamplesHistory) > 20:
                self.trainExamplesHistory.pop(0)

            # 3. 汇总所有历史数据 (混合多个游戏)
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # 统计各游戏样本数
            task_counts = {}
            for ex in trainExamples:
                gid = ex[3]
                task_counts[gid] = task_counts.get(gid, 0) + 1
            print(f"  Training data: {len(trainExamples)} samples "
                  f"(mix: {task_counts})")

            # 4. 任务感知训练
            loss_data = self.nnet.train(trainExamples)

            # 记录指标
            self.metrics['task_schedule'].append(current_game_id)
            if loss_data['total_loss']:
                self.metrics['policy_loss'].append(
                    np.mean(loss_data['policy_loss']))
                self.metrics['value_loss'].append(
                    np.mean(loss_data['value_loss']))
                self.metrics['total_loss'].append(
                    np.mean(loss_data['total_loss']))
            self.metrics['data_size'].append(len(trainExamples))

            if self.metrics['total_loss']:
                avg_loss = self.metrics['total_loss'][-1]
                print(f"  Loss: {avg_loss:.4f} "
                      f"(pi: {self.metrics['policy_loss'][-1]:.4f}, "
                      f"v: {self.metrics['value_loss'][-1]:.4f})")

                # 追加保存 loss 到 loss.txt
                loss_path = os.path.join(self.args.checkpoint, "loss.txt")
                with open(loss_path, "a") as f:
                    f.write(f"{self.metrics['policy_loss'][-1]:.6f}\n")
            else:
                print(f"  Loss: N/A (insufficient data)")

            # 5. 保存候选模型
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint,
                filename=f'checkpoint_{i}.pth.tar')

            # 6. 多任务竞技场评估 (每 arenaInterval 迭代执行一次)
            arena_interval = getattr(self.args, 'arenaInterval', 2)
            if i % arena_interval == 0 and hasattr(self.args, 'arenaCompare'):
                print(f"  Arena evaluation (every {arena_interval} iters)...")
                arena = MultiTaskArena(
                    self.games,
                    num_games_per_task=getattr(
                        self.args, 'arenaGamesPerTask', 6),
                    num_sims=getattr(self.args, 'arenaSims', 15),
                )
                # 加载 best 模型作为 old_nnet
                from nnet.nnet import NNetWrapper
                first_game_id = self.game_ids[0]
                first_game = self.games[first_game_id]
                old_nnet = NNetWrapper(first_game, first_game_id)
                best_path = os.path.join(
                    self.args.checkpoint, 'best.pth.tar')
                if os.path.exists(best_path):
                    old_nnet.load_checkpoint(
                        folder=self.args.checkpoint, filename='best.pth.tar')

                report = arena.evaluate(self.nnet, old_nnet)

                if report['should_update']:
                    print(f"  {report['reason']}")
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, filename='best.pth.tar')
                    print("  Best model updated.")
                else:
                    print(f"  {report['reason']}")
                    print("  Best model NOT updated.")

                # 保存报告
                arena.save_report(report)
                self.metrics.setdefault('arena_reports', []).append(report)
            else:
                # 无竞技场时直接保存为 best
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename='best.pth.tar')

        return self.metrics
