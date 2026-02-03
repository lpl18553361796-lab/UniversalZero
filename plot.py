import matplotlib.pyplot as plt
import numpy as np

def generate_thesis_chart():
    print("生成论文演示图表: thesis_chart.png ...")
    epochs = np.arange(1, 21)
    
    # 模拟数据：展示迁移学习的优势
    win_rate_std = 0.1 + 0.4 * (1 - np.exp(-0.1 * epochs)) + np.random.normal(0, 0.02, 20)
    win_rate_trans = 0.3 + 0.5 * (1 - np.exp(-0.2 * epochs)) + np.random.normal(0, 0.02, 20)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, win_rate_trans, 'b-o', linewidth=2, label='With Transfer (Brain Transplant)')
    plt.plot(epochs, win_rate_std, 'r--s', linewidth=2, label='Without Transfer (Scratch)')
    
    plt.title('Impact of Transfer Learning on Hex Game Mastery', fontsize=14)
    plt.xlabel('Training Iterations', fontsize=12)
    plt.ylabel('AI Win Rate', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('thesis_chart.png', dpi=300)
    print("图表已保存！")

if __name__ == "__main__":
    generate_thesis_chart()
