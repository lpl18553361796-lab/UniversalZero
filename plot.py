"""
Metrics Dashboard: 训练收敛指标可视化

生成 3 合 1 仪表盘:
  1. Loss 曲线对比    - policy_loss 和 value_loss 在共享主干下的收敛趋势
  2. Elo 进化曲线     - 训练过程中 AI 对 Random 的 Elo 变化
  3. 迁移加速比       - "从零开始" vs "Backbone 迁移" 的 Hex 胜率上升速度

用法:
    python plot.py                         # 从 experiment_results/ 读取数据
    python plot.py path/to/results/        # 指定数据目录
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


RESULTS_DIR = './experiment_results/'


def load_metrics(results_dir):
    """加载所有实验指标"""
    data = {}
    files = {
        'breakthrough': 'metrics_breakthrough.json',
        'hex_scratch': 'metrics_hex_scratch.json',
        'hex_transfer': 'metrics_hex_transfer.json',
    }
    for key, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
            print(f"  Loaded: {filename}")
        else:
            print(f"  Missing: {filename} (skipped)")
    return data


def plot_loss_curves(ax, data):
    """
    图表 1: Loss 曲线对比
    显示 Breakthrough 和 Hex 的 policy_loss / value_loss 收敛趋势
    """
    ax.set_title('Training Loss Curves (Shared Backbone)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    has_data = False

    # Breakthrough losses
    if 'breakthrough' in data:
        m = data['breakthrough']
        iters = range(1, len(m.get('total_loss', [])) + 1)
        if m.get('policy_loss'):
            ax.plot(iters, m['policy_loss'], 'b-o', markersize=4, linewidth=1.5,
                    label='BT: Policy Loss', alpha=0.9)
            has_data = True
        if m.get('value_loss'):
            ax.plot(iters, m['value_loss'], 'b--s', markersize=4, linewidth=1.5,
                    label='BT: Value Loss', alpha=0.6)

    # Hex scratch losses
    if 'hex_scratch' in data:
        m = data['hex_scratch']
        iters = range(1, len(m.get('total_loss', [])) + 1)
        if m.get('policy_loss'):
            ax.plot(iters, m['policy_loss'], 'r-o', markersize=4, linewidth=1.5,
                    label='Hex (Scratch): Policy Loss', alpha=0.9)
            has_data = True
        if m.get('value_loss'):
            ax.plot(iters, m['value_loss'], 'r--s', markersize=4, linewidth=1.5,
                    label='Hex (Scratch): Value Loss', alpha=0.6)

    # Hex transfer losses
    if 'hex_transfer' in data:
        m = data['hex_transfer']
        iters = range(1, len(m.get('total_loss', [])) + 1)
        if m.get('policy_loss'):
            ax.plot(iters, m['policy_loss'], 'g-o', markersize=4, linewidth=1.5,
                    label='Hex (Transfer): Policy Loss', alpha=0.9)
            has_data = True
        if m.get('value_loss'):
            ax.plot(iters, m['value_loss'], 'g--s', markersize=4, linewidth=1.5,
                    label='Hex (Transfer): Value Loss', alpha=0.6)

    if has_data:
        ax.legend(fontsize=8, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No loss data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)


def plot_elo_evolution(ax, data):
    """
    图表 2: Elo 进化曲线
    显示训练过程中 AI 对 Random 的 Elo 变化
    """
    ax.set_title('Elo Rating vs Random Player', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Elo Rating')

    has_data = False

    # Breakthrough final Elo (单点)
    if 'breakthrough' in data and 'final_elo' in data['breakthrough']:
        bt_elo = data['breakthrough']['final_elo']
        bt_iters = len(data['breakthrough'].get('total_loss', [1]))
        ax.scatter([bt_iters], [bt_elo], color='blue', s=100, zorder=5,
                   marker='*', label=f'BT Final (Elo={bt_elo:.0f})')
        has_data = True

    # Hex scratch Elo evolution
    if 'hex_scratch' in data and data['hex_scratch'].get('elo_history'):
        elos = data['hex_scratch']['elo_history']
        iters = range(1, len(elos) + 1)
        ax.plot(iters, elos, 'r-o', markersize=5, linewidth=2,
                label='Hex (Scratch)')
        has_data = True

    # Hex transfer Elo evolution
    if 'hex_transfer' in data and data['hex_transfer'].get('elo_history'):
        elos = data['hex_transfer']['elo_history']
        iters = range(1, len(elos) + 1)
        ax.plot(iters, elos, 'g-^', markersize=5, linewidth=2,
                label='Hex (Transfer)')
        has_data = True

    # Baseline: Random player Elo
    ax.axhline(y=800, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Random (Elo=800)')

    if has_data:
        ax.legend(fontsize=9, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No Elo data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    ax.grid(True, linestyle='--', alpha=0.4)


def plot_transfer_comparison(ax, data):
    """
    图表 3: 迁移加速比
    "从零开始" vs "Backbone 迁移" 的 Hex 胜率上升对比
    """
    ax.set_title('Transfer Acceleration: Win Rate vs Random', fontsize=13, fontweight='bold')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Win Rate vs Random')

    has_data = False

    # Hex scratch win rate
    if 'hex_scratch' in data and data['hex_scratch'].get('win_rate_history'):
        wr = data['hex_scratch']['win_rate_history']
        iters = range(1, len(wr) + 1)
        ax.plot(iters, wr, 'r--s', markersize=6, linewidth=2,
                label='From Scratch', alpha=0.9)
        # 填充区域
        ax.fill_between(iters, 0, wr, alpha=0.1, color='red')
        has_data = True

    # Hex transfer win rate
    if 'hex_transfer' in data and data['hex_transfer'].get('win_rate_history'):
        wr = data['hex_transfer']['win_rate_history']
        iters = range(1, len(wr) + 1)
        ax.plot(iters, wr, 'g-o', markersize=6, linewidth=2,
                label='With Backbone Transfer', alpha=0.9)
        ax.fill_between(iters, 0, wr, alpha=0.1, color='green')
        has_data = True

    # 50% baseline
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='50% baseline')

    if has_data:
        ax.legend(fontsize=9, loc='lower right')

        # 标注加速比
        if ('hex_scratch' in data and data['hex_scratch'].get('win_rate_history') and
                'hex_transfer' in data and data['hex_transfer'].get('win_rate_history')):
            wr_s = data['hex_scratch']['win_rate_history']
            wr_t = data['hex_transfer']['win_rate_history']
            if wr_s and wr_t:
                final_scratch = wr_s[-1]
                final_transfer = wr_t[-1]
                if final_scratch > 0:
                    ratio = final_transfer / final_scratch
                    ax.annotate(
                        f'Transfer advantage: {ratio:.1f}x',
                        xy=(len(wr_t), final_transfer),
                        xytext=(-120, 20), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='green',
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    )
    else:
        ax.text(0.5, 0.5, 'No win rate data available\nRun: python experiments.py --quick',
                transform=ax.transAxes, ha='center', va='center', fontsize=11, color='gray')

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)


def generate_dashboard(results_dir=RESULTS_DIR, save_path=None):
    """生成 3 合 1 仪表盘"""
    print("\n=== Generating Metrics Dashboard ===")

    data = load_metrics(results_dir)

    if not data:
        print("No data found. Run experiments first:")
        print("  python experiments.py --quick")
        return

    # 创建 3 合 1 布局
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_loss_curves(ax1, data)
    plot_elo_evolution(ax2, data)
    plot_transfer_comparison(ax3, data)

    fig.suptitle('UniversalZero MTL Training Dashboard', fontsize=16, fontweight='bold', y=1.02)

    if save_path is None:
        save_path = os.path.join(results_dir, 'dashboard.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nDashboard saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    results_dir = RESULTS_DIR
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        results_dir = sys.argv[1]
    generate_dashboard(results_dir)
