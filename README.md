# UniversalZero: 多任务棋类迁移学习平台

UniversalZero 是一个基于 AlphaZero 算法的通用强化学习框架，旨在探索不同复杂度的棋类游戏（Hex, Othello, TicTacToe）之间的特征迁移与知识共享。

## 🌟 核心特性
- **共享主干网络 (Shared Backbone)**: 使用统一的 ResNet 提取跨游戏的通用空间特征。
- **多任务路由 (MTL Routing)**: 针对不同游戏动态挂载独立的策略头。
- **知识迁移实验**: 支持从简单任务（如 TicTacToe/Othello）向复杂任务（如 Hex）迁移权重。
- **交互式 UI**: 基于 Streamlit 的可视化对局与模型性能评估工具。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行交互式 UI
```bash
streamlit run ui/app.py
```

### 3. 开启训练
- **从零训练**: `python scripts/main.py --game hex --iters 60`
- **迁移学习**: `python scripts/transfer.py --source final_models/othello_expert.pth.tar --target_game hex --iters 60 --freeze`

## 📊 实验结论 (2024.05)
- **迁移加速**: 在 Hex (7x7/11x11) 任务中，加载 Othello 专家权重的模型在前期探索阶段比从零训练快约 30%。
- **尺度敏感性**: 发现 TicTacToe (3x3) 由于在 9x9 输入空间中过于稀疏，其特征对于大型棋盘的迁移贡献度有限。

## 📂 项目结构
- `core/`: MCTS 引擎与训练调度器
- `games/`: 各棋类游戏逻辑实现
- `nnet/`: 通用神经网络架构与迁移管理器
- `ui/`: Streamlit 交互界面
- `scripts/`: 训练与实验脚本
