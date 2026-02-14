import os
import sys
import numpy as np
import torch
import torch.optim as optim

# 确保项目根目录在 sys.path 中，使游戏模块可被导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入所有游戏模块，触发它们的 register_game() 调用
import breakthrough    # noqa: F401 -> 注册 'breakthrough'
import hex_game        # noqa: F401 -> 注册 'hex'
import universal_game  # noqa: F401 -> 注册 JSON 驱动的游戏 (如 'bt_json')

from game import GAME_REGISTRY
from .model import UniversalNet


class NNetWrapper:
    """
    神经网络包装器 (MTL 版本)

    内部持有一个 UniversalNet 实例 (包含所有已注册游戏的策略头)，
    通过 game_id 路由到对应的策略分支进行训练和推理。
    """
    def __init__(self, game, game_id):
        """
        Args:
            game: 游戏实例 (用于获取棋盘尺寸)
            game_id: 游戏标识符 (如 'breakthrough', 'hex')
        """
        self.game_id = game_id
        self.nnet = UniversalNet(GAME_REGISTRY)
        self.game_x, self.game_y = game.get_board_size()
        self.MAX_SIZE = 9

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.nnet.to(self.device)

    def _pad_board(self, board):
        """
        【适配器】将任意小于 9x9 的棋盘补零到 9x9
        """
        if len(board.shape) == 2:
            padded = np.zeros((self.MAX_SIZE, self.MAX_SIZE))
            padded[:self.game_x, :self.game_y] = board
            return padded
        else:
            batch_size = board.shape[0]
            padded = np.zeros((batch_size, self.MAX_SIZE, self.MAX_SIZE))
            padded[:, :self.game_x, :self.game_y] = board
            return padded

    def train(self, examples):
        """
        任务感知训练 (Task-Aware Training)

        支持两种样本格式:
            - 3-tuple: (board, pi, v)          — 单任务模式，使用 self.game_id
            - 4-tuple: (board, pi, v, game_id)  — 多任务模式，按 game_id 路由

        对于混合任务批次:
            1. 共享主干 (Backbone) 处理所有样本
            2. 按 game_id 分组，路由到对应的策略头计算 loss_pi
            3. 共享价值头 (Value Head) 统一计算 loss_v

        Returns:
            dict: {'policy_loss': [...], 'value_loss': [...], 'total_loss': [...]}
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)
        batch_size = 64
        num_epochs = 10

        loss_history = {'policy_loss': [], 'value_loss': [], 'total_loss': []}

        # 检测样本格式
        multi_task = len(examples[0]) == 4

        self.nnet.train()
        for epoch in range(num_epochs):
            batch_count = int(len(examples) / batch_size)
            epoch_pi_loss = 0.0
            epoch_v_loss = 0.0

            for _ in range(batch_count):
                ids = np.random.randint(len(examples), size=batch_size)
                batch = [examples[i] for i in ids]

                # --- 解包样本 (兼容 3-tuple / 4-tuple) ---
                if multi_task:
                    boards, pis, vs, game_ids = list(zip(*batch))
                else:
                    boards, pis, vs = list(zip(*batch))
                    game_ids = [self.game_id] * len(batch)

                # --- 通用补零 (支持混合棋盘尺寸) ---
                padded = np.zeros((len(boards), self.MAX_SIZE, self.MAX_SIZE))
                for i, b in enumerate(boards):
                    h, w = b.shape
                    padded[i, :h, :w] = b

                boards_t = torch.FloatTensor(padded.astype(np.float64)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)

                # === 共享主干: 提取特征 (全部样本) ===
                features = self.nnet.backbone(
                    boards_t.view(-1, 1, self.MAX_SIZE, self.MAX_SIZE)
                )

                # === 共享价值头: 统一计算 loss_v ===
                out_v = self.nnet.value_head(features)
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / len(batch)

                # === 策略头: 按 game_id 路由计算 loss_pi ===
                pi_losses = []
                unique_games = set(game_ids)
                for gid in unique_games:
                    mask = [i for i, g in enumerate(game_ids) if g == gid]
                    game_features = features[mask]
                    game_pi_out = self.nnet.policy_heads[gid](game_features)
                    game_target = torch.FloatTensor(
                        np.array([pis[i] for i in mask])
                    ).to(self.device)
                    pi_losses.append(-torch.sum(game_target * game_pi_out))

                l_pi = sum(pi_losses) / len(batch)

                total_loss = l_pi + l_v

                # 反向传播 (梯度同时流经 backbone + 各策略头 + 价值头)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_pi_loss += l_pi.item()
                epoch_v_loss += l_v.item()

            if batch_count > 0:
                loss_history['policy_loss'].append(epoch_pi_loss / batch_count)
                loss_history['value_loss'].append(epoch_v_loss / batch_count)
                loss_history['total_loss'].append((epoch_pi_loss + epoch_v_loss) / batch_count)

        return loss_history

    def predict(self, board):
        # --- 关键步骤：补零 ---
        board = self._pad_board(board)

        board = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board = board.view(1, 1, self.MAX_SIZE, self.MAX_SIZE)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board, self.game_id)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        if not os.path.exists(folder):
            os.mkdir(folder)
        filepath = os.path.join(folder, filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path {filepath}")
        map_location = None if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
