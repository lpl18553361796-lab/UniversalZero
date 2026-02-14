import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
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

class ResNet(nn.Module):
    """
    通用型 ResNet (9x9 Standardized)
    """
    def __init__(self, game, num_res_blocks=4, num_channels=64):
        super().__init__()
        
        # --- 关键修改：定义全宇宙通用的最大尺寸 ---
        self.MAX_SIZE = 9 
        
        self.action_size = game.get_action_size()
        
        # 1. 输入层 (通用)
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # 2. 残差塔 (Backbone - 这是我们要迁移的部分)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 3. 策略头 (Policy Head - 专用)
        self.conv_p = nn.Conv2d(num_channels, 4, kernel_size=1)
        self.bn_p = nn.BatchNorm2d(4)
        # 注意：这里固定用 MAX_SIZE * MAX_SIZE 计算输入维度 (4*9*9)
        self.fc_p = nn.Linear(4 * self.MAX_SIZE * self.MAX_SIZE, self.action_size)
        
        # 4. 价值头 (Value Head - 专用)
        self.conv_v = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(2)
        # 注意：这里固定用 MAX_SIZE * MAX_SIZE 计算输入维度 (2*9*9)
        self.fc_v1 = nn.Linear(2 * self.MAX_SIZE * self.MAX_SIZE, 64)
        self.fc_v2 = nn.Linear(64, 1)

    def forward(self, s):
        # s shape: (batch, 1, 9, 9) <-- 必须永远接收 9x9
        # 确保 view 维度正确
        s = s.view(-1, 1, self.MAX_SIZE, self.MAX_SIZE)
        
        x = F.relu(self.bn1(self.conv1(s)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy
        p = F.relu(self.bn_p(self.conv_p(x)))
        p = p.view(-1, 4 * self.MAX_SIZE * self.MAX_SIZE) # 展平
        p = self.fc_p(p)
        p = F.log_softmax(p, dim=1)
        
        # Value
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(-1, 2 * self.MAX_SIZE * self.MAX_SIZE) # 展平
        v = F.relu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))
        
        return p, v
