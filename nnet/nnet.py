import os
import numpy as np
import torch
import torch.optim as optim
from .model import ResNet

class NNetWrapper:
    def __init__(self, game):
        self.nnet = ResNet(game)
        self.game_x, self.game_y = game.get_board_size()
        self.MAX_SIZE = 9 # 与 model.py 保持一致

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.nnet.to(self.device)

    def _pad_board(self, board):
        """
        【适配器】将任意小于 9x9 的棋盘补零到 9x9
        """
        # board shape: (batch_size, x, y) 或 (x, y)
        if len(board.shape) == 2:
            # 单个棋盘
            padded = np.zeros((self.MAX_SIZE, self.MAX_SIZE))
            padded[:self.game_x, :self.game_y] = board
            return padded
        else:
            # 批量棋盘
            batch_size = board.shape[0]
            padded = np.zeros((batch_size, self.MAX_SIZE, self.MAX_SIZE))
            padded[:, :self.game_x, :self.game_y] = board
            return padded

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)
        batch_size = 64
        
        self.nnet.train()
        for epoch in range(10): 
            batch_count = int(len(examples) / batch_size)
            
            for _ in range(batch_count):
                ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in ids]))
                
                # --- 关键步骤：补零 ---
                boards = self._pad_board(np.array(boards))
                
                boards = torch.FloatTensor(boards.astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)

                # 前向传播
                out_pi, out_v = self.nnet(boards)
                
                # 计算 Loss
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = l_pi + l_v

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        # --- 关键步骤：补零 ---
        board = self._pad_board(board)
        
        board = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board = board.view(1, 1, self.MAX_SIZE, self.MAX_SIZE) # 变成 (1, 1, 9, 9)
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

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
