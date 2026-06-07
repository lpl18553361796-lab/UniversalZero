"""
Arena: 模型对战评估系统

支持两个 AI (MCTS+NNet) 之间的自动对战，
或 AI vs 随机玩家的评估，并计算 Elo 分数。

多任务竞技场 (MultiTaskArena):
    跨游戏评估 + 多维胜率判定 + 进化决策 + JSON 报告
"""
import json
import os
import time
import numpy as np
import torch
from mcts import MCTS
from utils import dotdict


class RandomPlayer:
    """随机玩家 - 作为 baseline"""
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        valid_indices = np.where(valids == 1)[0]
        if len(valid_indices) == 0:
            return -1
        return np.random.choice(valid_indices)


class MCTSPlayer:
    """MCTS 驱动的 AI 玩家"""
    def __init__(self, game, nnet, num_sims=25):
        self.game = game
        args = dotdict({'num_mcts_sims': num_sims, 'cpuct': 1.0})
        self.mcts = MCTS(game, nnet, args)

    def play(self, board):
        probs = self.mcts.getActionProb(board, temp=0)
        return np.argmax(probs)


class Arena:
    """
    对战擂台

    让两个玩家进行多局对战，统计胜负，计算 Elo。
    player1 执先手 (白)，player2 执后手 (黑)。
    为公平起见，play_games 会交换先后手各打一半。
    """
    def __init__(self, game, player1, player2):
        self.game = game
        self.player1 = player1
        self.player2 = player2

    def play_one_game(self, p1_starts=True):
        """
        进行一局对战。

        Args:
            p1_starts: True = player1 先手 (白), False = player2 先手

        Returns:
            1 if player1 wins, -1 if player2 wins, 0 for draw
        """
        board = self.game.get_initial_board()
        player = 1  # 当前全局玩家 (1=白先手)

        if p1_starts:
            players = {1: self.player1, -1: self.player2}
        else:
            players = {1: self.player2, -1: self.player1}

        while True:
            # 检查游戏是否结束
            result = self.game.get_game_ended(board, player)
            if result != 0:
                # result 是相对于 player 的结果
                # 需要映射回 player1/player2 的结果
                if p1_starts:
                    return result if player == 1 else -result
                else:
                    return -result if player == 1 else result

            # 当前玩家行动
            canonical = self.game.get_canonical_form(board, player)
            current_player_obj = players[player]
            action = current_player_obj.play(canonical)

            if action < 0:
                # 无合法动作
                if p1_starts:
                    return -1 if player == 1 else 1
                else:
                    return 1 if player == 1 else -1

            # 执行动作
            next_canonical, _ = self.game.get_next_state(canonical, action, 1)
            board = self.game.get_canonical_form(next_canonical, player)
            player = -player

    def play_games(self, num_games):
        """
        进行多局对战 (自动交换先后手)。

        Returns:
            (p1_wins, p2_wins, draws): 三元组
        """
        p1_wins = 0
        p2_wins = 0
        draws = 0
        half = num_games // 2

        for i in range(num_games):
            p1_starts = (i < half)
            result = self.play_one_game(p1_starts=p1_starts)

            if result > 0:
                p1_wins += 1
            elif result < 0:
                p2_wins += 1
            else:
                draws += 1

        return p1_wins, p2_wins, draws

    @staticmethod
    def compute_elo(wins, losses, draws=0, base_elo=1000, opponent_elo=1000):
        """
        根据对战结果计算 Elo 分数。

        Args:
            wins: 胜场数
            losses: 败场数
            draws: 平局数
            base_elo: 基础 Elo (起始分)
            opponent_elo: 对手 Elo

        Returns:
            estimated Elo rating
        """
        total = wins + losses + draws
        if total == 0:
            return base_elo

        score = (wins + 0.5 * draws) / total
        # 防止 log(0) 或 log(inf)
        score = max(0.001, min(0.999, score))

        # Elo 公式: E = 1 / (1 + 10^((Rb-Ra)/400))
        # 反推: Ra = Rb - 400 * log10(1/E - 1)
        elo = opponent_elo - 400 * np.log10(1.0 / score - 1.0)
        return elo


def evaluate_vs_random(game, nnet, num_games=20, num_sims=15):
    """
    便捷函数：评估 AI 对随机玩家的胜率。

    Returns:
        dict: {'win_rate': float, 'wins': int, 'losses': int, 'draws': int, 'elo': float}
    """
    ai_player = MCTSPlayer(game, nnet, num_sims=num_sims)
    random_player = RandomPlayer(game)

    arena = Arena(game, ai_player, random_player)
    wins, losses, draws = arena.play_games(num_games)

    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0
    elo = Arena.compute_elo(wins, losses, draws, opponent_elo=800)

    return {
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'elo': elo,
    }


# ================================================================
#  Task-Aware 推理代理
# ================================================================

class _TaskProxy:
    """
    轻量代理: 让一个 NNetWrapper 以指定 game_id 进行推理。

    MCTS 调用 nnet.predict(board) 时，代理会临时切换
    NNetWrapper 的 game_id 和棋盘尺寸，确保路由到正确的策略头。
    """
    def __init__(self, nnet_wrapper, game_id, game):
        self._nnet = nnet_wrapper
        self._game_id = game_id
        gx, gy = game.get_board_size()
        self._gx = gx
        self._gy = gy

    def predict(self, board):
        # 暂存原始状态
        orig_id = self._nnet.game_id
        orig_gx = self._nnet.game_x
        orig_gy = self._nnet.game_y

        # 切换到目标任务
        self._nnet.game_id = self._game_id
        self._nnet.game_x = self._gx
        self._nnet.game_y = self._gy
        try:
            return self._nnet.predict(board)
        finally:
            # 恢复原始状态
            self._nnet.game_id = orig_id
            self._nnet.game_x = orig_gx
            self._nnet.game_y = orig_gy


# ================================================================
#  多任务竞技场 (Multi-Task Arena)
# ================================================================

class MultiTaskArena:
    """
    多任务竞技场

    在多个游戏环境下分别对弈，汇总各游戏胜率，
    并基于多维判定准则决定是否更新全局最优模型。

    判定准则 (Anti-Catastrophic-Forgetting):
        1. 新模型在所有游戏中胜率 >= min_win_rate (默认 45%)
        2. 新模型在至少一个游戏中胜率 >= upgrade_win_rate (默认 55%)
        满足 1 AND 2 才视为有效进化。
    """

    def __init__(self, games, num_games_per_task=10, num_sims=25,
                 min_win_rate=0.45, upgrade_win_rate=0.55):
        """
        Args:
            games: dict {game_id: game_instance}
            num_games_per_task: 每个游戏环境的对弈局数
            num_sims: MCTS 模拟次数
            min_win_rate: 防遗忘下限 (任何游戏不低于此值)
            upgrade_win_rate: 进化上限 (至少一个游戏达到此值)
        """
        self.games = games
        self.game_ids = list(games.keys())
        self.num_games_per_task = num_games_per_task
        self.num_sims = num_sims
        self.min_win_rate = min_win_rate
        self.upgrade_win_rate = upgrade_win_rate

    def evaluate(self, new_nnet, old_nnet, verbose=True):
        """
        跨游戏评估: 新模型 vs 旧模型。

        Args:
            new_nnet: 新模型 (NNetWrapper)
            old_nnet: 旧模型 (NNetWrapper)
            verbose: 是否打印过程

        Returns:
            dict: {
                'results': {game_id: {wins, losses, draws, win_rate, elo}},
                'should_update': bool,
                'reason': str,
            }
        """
        results = {}

        for gid in self.game_ids:
            game = self.games[gid]

            # 创建 task-aware 代理
            new_proxy = _TaskProxy(new_nnet, gid, game)
            old_proxy = _TaskProxy(old_nnet, gid, game)

            # 构建 MCTS 驱动的玩家
            mcts_args = dotdict({
                'num_mcts_sims': self.num_sims,
                'cpuct': 1.0,
            })
            new_player = MCTSPlayer(game, new_proxy, num_sims=self.num_sims)
            old_player = MCTSPlayer(game, old_proxy, num_sims=self.num_sims)

            # 对弈
            arena = Arena(game, new_player, old_player)
            wins, losses, draws = arena.play_games(self.num_games_per_task)

            total = wins + losses + draws
            win_rate = wins / total if total > 0 else 0.0
            elo = Arena.compute_elo(wins, losses, draws)

            results[gid] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': round(win_rate, 4),
                'elo': round(elo, 1),
            }

            if verbose:
                print(f"  [{gid}] New {wins}W / {losses}L / {draws}D "
                      f"(WR: {win_rate:.1%}, Elo: {elo:.0f})")

        # 判定是否更新
        should_update, reason = self.check_replacement_threshold(results)

        return {
            'results': results,
            'should_update': should_update,
            'reason': reason,
        }

    def check_replacement_threshold(self, results):
        """
        多任务进化判定准则。

        规则:
            1. 所有游戏胜率 >= min_win_rate (防灾难性遗忘)
            2. 至少一个游戏胜率 >= upgrade_win_rate (证明有效进化)

        Returns:
            (should_update: bool, reason: str)
        """
        all_above_min = True
        any_above_upgrade = False
        failed_games = []
        strong_games = []

        for gid, res in results.items():
            wr = res['win_rate']
            if wr < self.min_win_rate:
                all_above_min = False
                failed_games.append(f"{gid}({wr:.1%})")
            if wr >= self.upgrade_win_rate:
                any_above_upgrade = True
                strong_games.append(f"{gid}({wr:.1%})")

        if not all_above_min:
            return False, (
                f"REJECT: Catastrophic forgetting detected. "
                f"Games below {self.min_win_rate:.0%}: {', '.join(failed_games)}"
            )

        if not any_above_upgrade:
            return False, (
                f"REJECT: No game reached {self.upgrade_win_rate:.0%} threshold. "
                f"Model did not demonstrate sufficient improvement."
            )

        return True, (
            f"ACCEPT: All games >= {self.min_win_rate:.0%}, "
            f"strong in: {', '.join(strong_games)}"
        )

    def save_report(self, report, folder='assets/arena_results',
                    filename=None):
        """将评估报告保存为 JSON"""
        os.makedirs(folder, exist_ok=True)
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'arena_report_{timestamp}.json'
        filepath = os.path.join(folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return filepath
