import torch
import os

class TransferManager:
    """
    手术式权重管理器：负责主干共享与头部重置
    """
    @staticmethod
    def transfer_weights(model, checkpoint_path, target_game_id, verbose=True):
        """
        model: 你的 UniversalNet 实例
        checkpoint_path: 源模型路径 (例如已训练好的 breakthrough 权重)
        target_game_id: 你现在要开始训练的新游戏 ID (例如 'hex')
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到源模型文件: {checkpoint_path}")

        # 1. 加载源权重字典
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # 2. 准备新的权重字典，过滤掉所有旧的 Policy Heads
        new_state_dict = {}
        backbone_count = 0
        value_count = 0
        
        for k, v in state_dict.items():
            # 保持主干 (Backbone) 和 价值头 (ValueHead)
            if k.startswith('backbone') or k.startswith('value_head'):
                new_state_dict[k] = v
                if k.startswith('backbone'): backbone_count += 1
                if k.startswith('value_head'): value_count += 1
            # 过滤掉其他的 policy_heads，这些会在 model.load_state_dict(..., strict=False) 时保持随机初始化
        
        # 3. 注入权重
        # strict=False 非常关键：它允许目标模型的 'policy_heads.hex' 找不到对应权重时不报错，保持随机状态
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if verbose:
            print(f"--- 经验转移报告 ---")
            print(f"成功迁移 Backbone 参数项: {backbone_count}")
            print(f"成功迁移 ValueHead 参数项: {value_count}")
            print(f"已重置/未匹配的 Policy Head 分支: {[k for k in missing_keys if 'policy_heads.' + target_game_id in k]}")
            print(f"-------------------")
            
        return model

# 拓展接口预留：
# TODO: 实现 freeze_backbone(model) 接口，用于在树莓派上进行极低功耗的“仅头部微调”
