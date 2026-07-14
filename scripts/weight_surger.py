import torch
import os
import sys

# --- 确保项目各层级路径正确 ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    sys.path.append(os.path.join(_project_root, "games"))
    sys.path.append(os.path.join(_project_root, "core"))
    sys.path.append(os.path.join(_project_root, "ui"))

from nnet.model import UniversalNet
from game import GAME_REGISTRY

def surgery_expert_brain(source_path, target_game_id='hex'):
    """
    高级权重手术：从 512 通道的 8x8 专家模型提取核心 DNA 注入到 64 通道的 9x9 架构
    """
    print(f">>> 正在启动跨架构权重移植手术...")
    print(f">>> 源文件: {source_path}")
    
    # 1. 实例化目标模型 (64 通道)
    target_model = UniversalNet(GAME_REGISTRY, num_channels=64)
    target_state = target_model.state_dict()
    
    # 2. 加载源模型 (512 通道)
    try:
        source_checkpoint = torch.load(source_path, map_location='cpu')
        source_state = source_checkpoint.get('state_dict', source_checkpoint)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 3. 建立映射逻辑
    # 源模型: conv1, bn1, conv2, bn2, ...
    # 目标模型: backbone.conv1, backbone.bn1, backbone.res_blocks.0.conv1, ...
    
    mapping = {
        'conv1': 'backbone.conv1',
        'bn1': 'backbone.bn1',
        'conv2': 'backbone.res_blocks.0.conv1',
        'bn2': 'backbone.res_blocks.0.bn1',
        'conv3': 'backbone.res_blocks.1.conv1',
        'bn3': 'backbone.res_blocks.1.bn1',
    }
    
    matched_layers = []

    for s_key_base, t_key_base in mapping.items():
        # 匹配 weight, bias, running_mean, running_var 等
        for suffix in ['.weight', '.bias', '.running_mean', '.running_var']:
            s_key = s_key_base + suffix
            t_key = t_key_base + suffix
            
            if s_key in source_state and t_key in target_state:
                s_param = source_state[s_key]
                t_param = target_state[t_key]
                
                # 如果形状完全一致，直接复制
                if s_param.shape == t_param.shape:
                    target_state[t_key].copy_(s_param)
                    matched_layers.append(t_key)
                # 如果是通道数不一致 (例如 512 vs 64)，执行切片移植
                elif len(s_param.shape) == len(t_param.shape):
                    try:
                        if len(s_param.shape) == 4: # Conv 层: [out, in, k, k]
                            out_c, in_c = t_param.shape[0], t_param.shape[1]
                            target_state[t_key].copy_(s_param[:out_c, :in_c, :, :])
                        elif len(s_param.shape) == 1: # BN 或 Bias: [c]
                            c = t_param.shape[0]
                            target_state[t_key].copy_(s_param[:c])
                        matched_layers.append(t_key + " (Slicing)")
                    except Exception as e:
                        print(f"跳过 {s_key} -> {t_key}: {e}")

    # 4. 保存结果
    target_model.load_state_dict(target_state)
    if not os.path.exists("experiment_results"):
        os.makedirs("experiment_results")
        
    save_path = f"experiment_results/expert_injected_{target_game_id}.pth.tar"
    torch.save({'state_dict': target_model.state_dict()}, save_path)
    
    print(f"\n--- 手术成功 ---")
    print(f"成功移植参数层数: {len(matched_layers)}")
    print(f"已生成零样本种子: {save_path}")
    print(f"提示：该种子现在包含了 Othello 的前 64 个特征通道直觉。")

if __name__ == "__main__":
    expert_file = "pretrained_models/othello_expert_8x8.pth.tar"
    if os.path.exists(expert_file):
        surgery_expert_brain(expert_file)
    else:
        print("未找到专家模型文件。")
