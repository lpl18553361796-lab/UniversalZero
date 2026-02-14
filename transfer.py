import torch
import os

# 确保所有游戏已注册
from breakthrough import BreakthroughGame  # noqa: F401
from hex_game import HexGame              # noqa: F401
from game import GAME_REGISTRY
from nnet.model import UniversalNet


def transfer_weights(source_path, target_save_path, target_game_id='hex'):
    """
    MTL 架构下的大脑移植：
    加载已训练的 UniversalNet 检查点（含 Breakthrough 经验），
    重置目标游戏的策略头为随机权重，保留共享主干和价值头。

    效果：目标游戏的策略头从零开始，但 Backbone 保留了源游戏的几何直觉。
    """
    print("=== 开始大脑移植手术 (Backbone Transfer via MTL) ===")

    # 1. 创建 UniversalNet (包含所有已注册游戏的策略头)
    print("1. 初始化 UniversalNet (MTL 架构)...")
    net = UniversalNet(GAME_REGISTRY)

    # 2. 加载源检查点 (例如 Breakthrough 训练过的模型)
    if not os.path.exists(source_path):
        print(f">>> 错误：找不到源模型 {source_path}")
        print(">>> 请先运行训练 (如 main.py) 生成模型检查点。")
        return

    print(f"2. 加载训练过的模型: {source_path}")
    checkpoint = torch.load(source_path, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    print("   >>> 已加载完整模型 (Backbone + 所有策略头 + 价值头)")

    # 3. 重置目标游戏的策略头 (保留 Backbone 和 Value Head)
    if target_game_id not in net.policy_heads:
        print(f">>> 错误：游戏 '{target_game_id}' 未在注册表中找到。")
        print(f">>> 可用游戏: {list(net.policy_heads.keys())}")
        return

    print(f"3. 重置 '{target_game_id}' 的策略头为随机权重...")
    for name, param in net.policy_heads[target_game_id].named_parameters():
        if param.dim() >= 2:
            torch.nn.init.kaiming_normal_(param)
        else:
            torch.nn.init.zeros_(param)
    print("   >>> 策略头已重置 (Backbone 和 Value Head 保留源游戏经验)")

    # 4. 保存移植后的模型
    target_dir = os.path.dirname(target_save_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    torch.save({'state_dict': net.state_dict()}, target_save_path)
    print(f"4. 手术完成！保存到: {target_save_path}")
    print(f"=== '{target_game_id}' 现在拥有了源游戏的几何直觉！ ===")


if __name__ == "__main__":
    SOURCE_MODEL = './temp_strong/best.pth.tar'
    TARGET_MODEL = './temp_transfer/transferred.pth.tar'

    transfer_weights(SOURCE_MODEL, TARGET_MODEL, target_game_id='hex')
