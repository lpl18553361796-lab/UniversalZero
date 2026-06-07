"""
train_tictactoe.py
专为 TicTacToe 设计的独立训练脚本
网络输入：3x3（不做 padding）
完全独立于 UniversalNet，只用于展示
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from random import shuffle
from tqdm import tqdm

# 路径设置
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'games'))
sys.path.insert(0, os.path.join(_root, 'core'))

from games.tictactoe import TicTacToeGame
from core.mcts import MCTS

# ────────────────────────────────────────────────────────
# 专用小网络：输入 3x3，输出 9个动作 + 1个价值
# ────────────────────────────────────────────────────────

class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：直接处理 3x3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 4, kernel_size=1)
        self.policy_bn   = nn.BatchNorm2d(4)
        self.policy_fc   = nn.Linear(4 * 3 * 3, 9)

        # Value head
        self.value_conv  = nn.Conv2d(64, 2, kernel_size=1)
        self.value_bn    = nn.BatchNorm2d(2)
        self.value_fc1   = nn.Linear(2 * 3 * 3, 32)
        self.value_fc2   = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, 1, 3, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 4 * 3 * 3)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 2 * 3 * 3)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

class TicTacToeWrapper:
    """让 MCTS 能调用的包装器"""
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predict(self, board):
        # board: (3, 3)
        t = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        t = t.view(1, 1, 3, 3)
        self.net.eval()
        with torch.no_grad():
            log_pi, v = self.net(t)
        pi = torch.exp(log_pi).detach().cpu().numpy()[0]
        return pi, v.detach().cpu().numpy()[0]

# ────────────────────────────────────────────────────────
# 训练逻辑
# ────────────────────────────────────────────────────────

def play_episode(game, wrapper, args):
    examples = []
    board = game.get_initial_board()
    cur_player = 1
    step = 0
    mcts = MCTS(game, wrapper, args, pipe=None)

    while True:
        step += 1
        canonical = game.get_canonical_form(board, cur_player)
        temp = int(step < args['tempThreshold'])
        pi = mcts.getActionProb(canonical, temp=temp)

        examples.append([canonical, cur_player, pi, None])

        action = np.random.choice(9, p=pi)
        next_s, _ = game.get_next_state(canonical, action, 1)
        board = game.get_canonical_form(next_s, cur_player)
        cur_player = -cur_player

        r = game.get_game_ended(board, 1)
        if r != 0:
            # 修正：根据胜负结果填充 value
            return [(x[0], x[2], r * x[1]) for x in examples]

def train_epoch(net, optimizer, examples, device, batch_size=128):
    net.train()
    shuffle(examples)
    total_loss = 0
    batches = 0

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        boards, pis, vs = zip(*batch)

        boards_t = torch.FloatTensor(np.array(boards)).to(device).view(-1, 1, 3, 3)
        pis_t    = torch.FloatTensor(np.array(pis)).to(device)
        vs_t     = torch.FloatTensor(np.array(vs)).to(device)

        optimizer.zero_grad()
        log_pi, v = net(boards_t)

        loss_pi = -torch.mean(torch.sum(pis_t * log_pi, dim=1))
        loss_v  = torch.mean((vs_t - v.squeeze()) ** 2)
        loss    = loss_pi + loss_v
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)

class DotDict(dict):
    """把字典包装成可以通过点号访问属性的对象"""
    def __getattr__(self, name):
        return self[name]

def main():
    # 参数
    args = DotDict({
        'numIters':      50,
        'numEps':        100,
        'tempThreshold': 6,
        'num_mcts_sims': 100,
        'cpuct':         1.0,
        'max_depth':     20,
        'epochs':        10,
        'lr':            0.001,
        'checkpoint':    'experiment_results/tictactoe_standalone',
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    game    = TicTacToeGame()
    net     = TicTacToeNet().to(device)
    wrapper = TicTacToeWrapper(net, device)
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])

    history = deque(maxlen=20)
    os.makedirs(args['checkpoint'], exist_ok=True)

    for it in range(1, args['numIters'] + 1):
        print(f"\n------ Iteration {it}/{args['numIters']} ------")

        # 自对弈
        iter_examples = []
        for _ in tqdm(range(args['numEps']), desc="self-play"):
            iter_examples += play_episode(game, wrapper, args)

        history.append(iter_examples)

        all_examples = []
        for e in history:
            all_examples.extend(e)
        shuffle(all_examples)

        # 训练
        print(f"Training on {len(all_examples)} samples...")
        for epoch in range(args['epochs']):
            loss = train_epoch(net, optimizer, all_examples, device)
            print(f"  Epoch {epoch+1}/{args['epochs']} loss={loss:.4f}")

        # 保存
        torch.save({'state_dict': net.state_dict()},
                   os.path.join(args['checkpoint'], 'best.pth.tar'))

    print(f"\n训练完成，模型保存至 {args['checkpoint']}/best.pth.tar")

if __name__ == '__main__':
    main()
