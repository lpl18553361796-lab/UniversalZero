import torch
import os
from breakthrough import BreakthroughGame
from hex_game import HexGame
from nnet.model import ResNet

def transfer_weights(source_path, target_save_path):
    print("=== 开始大脑移植手术 (Backbone Transfer) ===")
    
    # 1. 准备手术对象
    print("1. 加载源模型 (Breakthrough)...")
    # 注意：这里加载模型时会用到新的 model.py (9x9)，所以源模型必须是重新训练过的
    game_source = BreakthroughGame()
    net_source = ResNet(game_source)
    
    if not os.path.exists(source_path):
        print(f">>> 错误：找不到源模型 {source_path}")
        print(">>> 因神经网络结构已升级(加了Padding)，旧模型已失效。")
        print(">>> 请先运行 train_strong.py 重新训练 Breakthrough！")
        return
        
    checkpoint = torch.load(source_path, map_location='cpu')
    net_source.load_state_dict(checkpoint['state_dict'])
    
    print("2. 初始化目标模型 (Hex)...")
    game_target = HexGame(n=7)
    net_target = ResNet(game_target) # 此时是随机初始化的
    
    # 2. 提取并移植权重
    print("3. 开始移植 '脑干' (Backbone Layers)...")
    source_dict = net_source.state_dict()
    target_dict = net_target.state_dict()
    
    transferred_layers = []
    
    for k, v in source_dict.items():
        # 严谨筛选：只移植输入层(conv1)和残差层(res_blocks)
        # 跳过所有决策头(conv_p, fc_p, conv_v, fc_v...)
        if "res_blocks" in k or k.startswith("conv1") or k.startswith("bn1"): 
            target_dict[k] = v
            transferred_layers.append(k)
            
    net_target.load_state_dict(target_dict)
    print(f"   >>> 成功移植了 {len(transferred_layers)} 层权重。")
    print("   >>> (全连接层保持随机初始化，以适应新游戏规则)")
    
    # 3. 保存新模型
    print(f"4. 手术完成！保存混合模型到: {target_save_path}")
    if not os.path.exists(os.path.dirname(target_save_path)):
        os.makedirs(os.path.dirname(target_save_path))
        
    torch.save({'state_dict': net_target.state_dict()}, target_save_path)
    print("=== 移植完毕，Hex 现在拥有了 Breakthrough 的几何直觉！ ===")

if __name__ == "__main__":
    # 源模型路径 (需要重新训练)
    SOURCE_MODEL = './temp_strong/best.pth.tar' 
    # 移植后的 Hex 初始模型
    TARGET_MODEL = './temp_transfer/transferred.pth.tar'
    
    transfer_weights(SOURCE_MODEL, TARGET_MODEL)
