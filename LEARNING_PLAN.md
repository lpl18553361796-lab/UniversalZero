# UniversalZero Learning Plan: Layer by Layer

To master the AlphaZero system, we will deconstruct it into 5 logical layers, from the bottom up. We will study each layer's responsibility, key files, and core concepts.

## Layer 1: The Foundation (Game Logic)
*Everything starts with the rules of the game.*
*   **Key Files**: `game.py` (Base), `breakthrough.py`, `hex_game.py`.
*   **Core Concepts**:
    *   **State Representation**: How an $8 \times 8$ board is stored (numpy array).
    *   **Action Space**: Mapping moves to integers (0-191).
    *   **Canonical Form**: Why AI always sees the board from the "White" perspective.
    *   **Symmetry**: Handling rotations/flips for data augmentation.

## Layer 2: The Intuition (Neural Network)
*The AI needs a way to evaluate positions instantly without thinking.*
*   **Key Files**: `nnet/model.py` (ResNet), `nnet/nnet.py` (Wrapper).
*   **Core Concepts**:
    *   **ResNet Architecture**: Convolutional blocks + Residual connections.
    *   **Policy Head**: Outputting a probability distribution over moves ($\boldsymbol{p}$).
    *   **Value Head**: Outputting a scalar win probability ($v \in [-1, 1]$).
    *   **Standardization**: Padding inputs to fixed size ($9 \times 9$) for universality.

## Layer 3: The Brain (MCTS Search)
*Intuition is fast but prone to errors; search corrects it.*
*   **Key Files**: `mcts.py`.
*   **Core Concepts**:
    *   **PUCT Algorithm**: Balancing Exploration ($U$) vs Exploitation ($Q$).
    *   **Simulation**: Thinking ahead into the future.
    *   **Backpropagation**: Updating node statistics ($N, Q$).
    *   **Temperature**: Controlling randomness in move selection.

## Layer 4: The Coach (Reinforcement Learning)
*How the system improves itself over time.*
*   **Key Files**: `coach.py`.
*   **Core Concepts**:
    *   **Self-Play**: Generating data by playing against itself.
    *   **Data Collection**: Storing `(Board, Policy, Value)` tuples.
    *   **Experience Replay**: Training on a shuffled buffer of past games.
    *   **Iterative Process**: Self-Play $\rightarrow$ Train $\rightarrow$ Update Best Model.

## Layer 5: The Interface (Human Interaction)
*Bringing it all together for the user.*
*   **Key Files**: `main.py`, `play.py`, `gui.py`.
*   **Core Concepts**:
    *   **Orchestration**: Intitializing components.
    *   **Visualization**: Rendering state and AI metrics (Heatmaps).
    *   **User Input**: Handling mouse/keyboard events.

---
**Recommended Learning Path**:
Start with **Layer 1** to understand the "Physics" of the world, then move to **Layer 3** to understand how decisions are made, then **Layer 2** for the evaluation mechanism, and finally **Layer 4** to see how it all learns.

## 🗂️ Project Survival Map (File Index)

Use this map to find the file you need based on what you want to do.

| File | Layer | Description |
| :--- | :---: | :--- |
| `game.py` | 1 | Abstract Base Class defining the interface for all games. |
| `breakthrough.py` | 1 | Rules for the main game (8x8 Breakthrough). |
| `hex_game.py` | 1 | Rules for the secondary game (7x7 Hex). |
| `mcts.py` | 3 | **The Engine**. Monte Carlo Tree Search algorithm. |
| `nnet/model.py` | 2 | **The Brain**. ResNet architecture definition. |
| `nnet/nnet.py` | 2 | **The Body**. PyTorch wrapper for training & inference. |
| `coach.py` | 4 | **The Teacher**. Orchestrates self-play and training loops. |
| `main.py` | 5 | Entry point for training Breakthrough. |
| `play.py` | 5 | Entry point for Human vs AI (CLI). |
| `gui.py` | 5 | Entry point for Human vs AI (Graphical). |
| `transfer.py` | Adv | Script for transplanting weights between games. |
| `plot.py` | Adv | Script for generating thesis performance charts. |

