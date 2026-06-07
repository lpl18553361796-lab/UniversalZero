import os
import torch
import numpy as np
from .model import UniversalNet

def pad_board(board, target_size=9):
    """居中填充算法：将小棋盘对齐到标准尺寸 (9x9)"""
    if len(board.shape) != 2: return board
    n = board.shape[0]
    if n == target_size: return board
    pad_before = (target_size - n) // 2
    pad_after = target_size - n - pad_before
    return np.pad(board, ((pad_before, pad_after), (pad_before, pad_after)), 
                  mode='constant', constant_values=0)

class NNetWrapper:
    def __init__(self, game, game_id, args=None):
        from game import GAME_REGISTRY
        # 传入全局注册表以动态生成所有游戏的策略头
        self.nnet = UniversalNet(GAME_REGISTRY)
        self.game_id = game_id
        self.args = args if args else {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)
        self.MAX_SIZE = 9

    def set_backbone_frozen(self, frozen=True):
        """
        实验验证接口：冻结/解锁主干权重
        frozen=True: 锁定主干和价值头，仅训练特定游戏的 Policy Head
        """
        # 1. 锁定共享主干
        for param in self.nnet.backbone.parameters():
            param.requires_grad = not frozen
            
        # 2. 锁定共享价值头
        for param in self.nnet.value_head.parameters():
            param.requires_grad = not frozen
            
        # 3. 确保当前及所有 Policy Heads 是可训练的
        for param in self.nnet.policy_heads.parameters():
            param.requires_grad = True
            
        status = "FROZEN (Locked)" if frozen else "TRAINABLE (Full-Tuning)"
        print(f">>> [实验配置] Backbone 状态: {status}")

    def train(self, examples):
        import torch.optim as optim
        
        # 核心逻辑：只为 requires_grad=True 的参数创建优化器
        trainable_params = [p for p in self.nnet.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.args.get('lr', 0.001))
        
        self.nnet.train()
        loss_history = {'policy_loss': [], 'value_loss': [], 'total_loss': []}

        processed_examples = []
        for board, pi, v, gid in examples:
            processed_examples.append((pad_board(board, 9), pi, v, gid))

        batch_size = self.args.get('batch_size', 64)
        epochs = self.args.get('epochs', 1)
        for _ in range(epochs):
            np.random.shuffle(processed_examples)
            for i in range(0, len(processed_examples), batch_size):
                batch = processed_examples[i : i + batch_size]
                boards, pis, vs, gids = list(zip(*batch))

                boards_t = torch.FloatTensor(np.array(boards)).to(self.device).view(-1, 1, 9, 9)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)

                optimizer.zero_grad()
                out_pi, out_v = self.nnet(boards_t, self.game_id)
                
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = l_pi + l_v

                total_loss.backward()
                optimizer.step()

                loss_history['policy_loss'].append(l_pi.item())
                loss_history['value_loss'].append(l_v.item())
                loss_history['total_loss'].append(total_loss.item())

        return loss_history

    def predict(self, board):
        padded = pad_board(board, 9)
        board_t = torch.FloatTensor(padded.astype(np.float32)).to(self.device).view(1, 1, 9, 9)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_t, self.game_id)
        return torch.exp(pi).detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder): os.mkdir(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        # 极致兼容性：强制重定向所有存储到 CPU，避开所有设备校验
        checkpoint = torch.load(
            filepath,
            map_location=lambda storage, loc: storage
        )
        # 使用 strict=False 允许加载只有主干权重的“专家种子”
        self.nnet.load_state_dict(checkpoint['state_dict'], strict=False)
