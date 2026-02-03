# UniversalZero: General-Purpose AlphaZero Engine

UniversalZero is a modular, high-performance implementation of the AlphaZero algorithm, designed to learn multiple board games from scratch without human knowledge. It features a standardized neural network architecture and a dynamic game registry system.

## 🚀 Key Features

### 🧠 Universal Brain (Standardized ResNet)
- **9x9 Input Normalization**: The neural network (ResNet) accepts a standardized 9x9 input, allowing it to adapt to any board game fitting within these dimensions (e.g., 8x8 Breakthrough, 7x7 Hex) via zero-padding.
- **Backbone Transfer**: Supports "Brain Surgery" (`transfer.py`), allowing a new game to inherit the geometric intuition (backbone weights) from a previously trained model.

### 🛡️ Robust MCTS Engine
- **Safety**: Implemented recursion depth limits (`max_depth=100`) to prevent stack overflows during deep searches.
- **Temperature Control**: Adjustable exploration/exploitation balance via temperature parameters.

### ⚡ Optimization (Hex Union-Find)
- **O(N²) -> O(1) Speedup**: Replaced DFS-based connectivity checks in Hex with an incremental **Union-Find (Disjoint Set Union)** data structure.
- **Virtual Nodes**: Uses virtual Top/Bottom/Left/Right nodes for efficient edge connection detection.

### 🧩 Dynamic Game Registry
- **Hot-Swapping**: Games are registered in a central `GAME_REGISTRY`.
- **String IDs**: Switch between 'breakthrough' and 'hex' dynamically at runtime without changing the core engine.

## 🎮 Supported Games

1.  **Breakthrough (8x8)**: A racing game of strategy and blocking.
    *   *Complexity*: Moderate
    *   *Avg Game Length*: ~40-60 moves
2.  **Hex (7x7)**: A connection game played on a hexagonal grid.
    *   *Complexity*: High (Connection strategy)
    *   *Optimization*: Union-Find enabled

## 🛠️ Quick Start

### Prerequisites
```bash
pip install numpy torch
# Optional for GUI
pip install pygame matplotlib
```

### Training
Train the AI from scratch on Breakthrough:
```bash
python main.py
```

Train on Hex:
```bash
python main_hex.py
```

### Human vs AI
Play against your trained model in the terminal:
```bash
python play.py
```
*(Follow the on-screen prompts. For Hex, inputs are `row col`)*

### Visualization (GUI)
Launch the graphical interface (requires pygame):
```bash
python gui.py breakthrough
# or
python gui.py hex
```

## 📂 Project Structure
- `mcts.py`: Monte Carlo Tree Search core.
- `nnet/`: Neural Network implementation (PyTorch).
- `coach.py`: Self-play and training loop orchestrator.
- `game.py`: Abstract base class and Registry.
- `breakthrough.py` / `hex_game.py`: Game rule implementations.
- `transfer.py`: Weight transfer utility.

---
*Built with ❤️ by Antigravity*
