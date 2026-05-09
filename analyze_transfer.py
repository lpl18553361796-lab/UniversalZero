import os
import matplotlib.pyplot as plt
import numpy as np

class TransferAnalyzer:
    """
    性能分析器：量化经验转移的收敛速度和初始表现
    """
    def __init__(self, results_base_dir="./experiment_results/"):
        self.results_dir = results_base_dir

    def parse_logs(self, folder_name):
        """
        解析指定实验文件夹中的 loss 数据 (假设格式为每行一个 float)
        """
        loss_path = os.path.join(self.results_dir, folder_name, "loss.txt")
        if not os.path.exists(loss_path):
            return
        with open(loss_path, "r") as f:
            return [float(line.strip()) for line in f.readlines()]

    def plot_convergence(self, target_game_id):
        """
        生成收敛曲线对比图
        """
        all_dirs = sorted(os.listdir(self.results_dir))
        
        # 修复点：通过 next() 获取第一个匹配的文件夹字符串，而不是直接使用列表
        try:
            scratch_folder = next(d for d in all_dirs if target_game_id in d and "scratch" in d)
            transfer_folder = next(d for d in all_dirs if target_game_id in d and "transfer" in d)
        except StopIteration:
            print(f"错误: 未能找到对应的对照实验组数据 {target_game_id}")
            return

        scratch_losses = self.parse_logs(scratch_folder)
        transfer_losses = self.parse_logs(transfer_folder)

        plt.figure(figsize=(10, 6))
        plt.plot(scratch_losses, label="Scratch (Tabula Rasa)", color='gray', linestyle='--')
        plt.plot(transfer_losses, label="Transfer (Pre-trained Backbone)", color='blue', linewidth=2)
        
        plt.title(f"Knowledge Transfer Efficiency: {target_game_id.upper()}")
        plt.xlabel("Iteration")
        plt.ylabel("Policy Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.results_dir, f"{target_game_id}_comparison.png")
        plt.savefig(save_path)
        print(f"对比图表已保存至: {save_path}")

    def calculate_jumpstart_performance(self, target_game_id):
        """
        计算冷启动性能提升 (Jumpstart)
        """
        # TODO: 接口预留 - 用于评估 Iteration 0 时两组模型对战基准 AI 的胜率差异 [1]
        print("接口预留: 计算 Iteration 0 的 Zero-shot 胜率...")
        pass

# 扩展接口预留：
# TODO: 实现 export_to_bibtex() 接口，自动生成论文所需的实验环境参数描述

if __name__ == "__main__":
    analyzer = TransferAnalyzer()
    analyzer.plot_convergence('hex')
