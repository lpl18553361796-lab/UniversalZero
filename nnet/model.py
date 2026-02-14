import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块 (Residual Block)"""
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """
    共享主干网络 (Shared Backbone)
    负责从通用 9x9 棋盘中提取空间特征。

    输入: (batch, 1, 9, 9)
    输出: (batch, num_channels, 9, 9) 高维特征图
    """
    MAX_SIZE = 9

    def __init__(self, in_channels=1, num_channels=64, num_res_blocks=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

    def forward(self, x):
        x = x.view(-1, 1, self.MAX_SIZE, self.MAX_SIZE)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        return x


class PolicyHead(nn.Module):
    """
    策略头 (Policy Head)
    每个游戏拥有独立的策略头，输出维度 = action_size。
    """
    MAX_SIZE = 9

    def __init__(self, num_channels, action_size):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * self.MAX_SIZE * self.MAX_SIZE, action_size)

    def forward(self, x):
        p = F.relu(self.bn(self.conv(x)))
        p = p.view(-1, 4 * self.MAX_SIZE * self.MAX_SIZE)
        p = self.fc(p)
        return F.log_softmax(p, dim=1)


class ValueHead(nn.Module):
    """
    共享价值头 (Shared Value Head)
    所有游戏共享同一个价值头，输出胜率评估 [-1, 1]。
    局面评估在语义上是通用的：无论什么游戏，-1 到 1 的范围含义一致。
    """
    MAX_SIZE = 9

    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.fc1 = nn.Linear(2 * self.MAX_SIZE * self.MAX_SIZE, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        v = F.relu(self.bn(self.conv(x)))
        v = v.view(-1, 2 * self.MAX_SIZE * self.MAX_SIZE)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        return v


class UniversalNet(nn.Module):
    """
    多任务学习 (MTL) 通用网络

    架构：共享主干 (Backbone) + 动态策略头 (Policy Heads) + 共享价值头 (Value Head)

    在初始化时遍历 game_registry，为每个已注册游戏动态生成匹配的策略输出头。
    前向传播通过 game_id 参数路由到正确的策略分支。

    结构图:
        输入 (1, 9, 9)
            |
        [ResNetBackbone] --- 共享
            |
        +---+---+
        |       |
    [PolicyHead] [ValueHead] --- 共享
    (per game)
        |       |
      log_π    tanh(v)
    """
    def __init__(self, game_registry, num_res_blocks=4, num_channels=64):
        super().__init__()
        self.MAX_SIZE = 9
        self.num_channels = num_channels

        # 1. 共享主干
        self.backbone = ResNetBackbone(
            in_channels=1,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks
        )

        # 2. 动态策略头 - 遍历注册表，为每个游戏创建独立的策略头
        #    支持两种注册形式：类 (type) 或已实例化的对象
        self.policy_heads = nn.ModuleDict()
        for game_id, entry in game_registry.items():
            game_instance = entry() if isinstance(entry, type) else entry
            action_size = game_instance.get_action_size()
            self.policy_heads[game_id] = PolicyHead(num_channels, action_size)

        # 3. 共享价值头
        self.value_head = ValueHead(num_channels)

    def forward(self, x, game_id):
        """
        前向传播，根据 game_id 路由到对应的策略头。

        Args:
            x: 输入张量 (batch, 1, 9, 9)
            game_id: 游戏标识符，如 'breakthrough' 或 'hex'

        Returns:
            (log_pi, v): 策略对数概率 和 价值评估
        """
        # 共享特征提取
        features = self.backbone(x)

        # 路由到对应的策略头
        if game_id not in self.policy_heads:
            raise ValueError(
                f"Unknown game_id '{game_id}'. "
                f"Available: {list(self.policy_heads.keys())}"
            )
        p = self.policy_heads[game_id](features)

        # 共享价值评估
        v = self.value_head(features)

        return p, v
