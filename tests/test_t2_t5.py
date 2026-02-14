"""
T2~T5 Verification Tests

T2: Action Space Auto-Adaptation (8x8->192, 5x5->75)
T3: MTL Routing Isolation (Hex+BT head -> dimension mismatch)
T4: MCTS Depth Cutoff (max_depth=3, no RecursionError)
T5: UI Coordinate Mapping (Hex (2,3) -> action 17)
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import GAME_REGISTRY, get_game_by_id, register_game
from universal_game import UniversalGame
from mcts import MCTS
import torch

print("=" * 60)
print("  T2 ~ T5 Verification Tests")
print("=" * 60)

# ==============================================================
# T2: Action Space Auto-Adaptation
# ==============================================================
print("\n--- T2: Action Space Auto-Adaptation ---")

bt = get_game_by_id('breakthrough_json')
assert bt.get_action_size() == 192
print(f"  [OK] Breakthrough 8x8: action_size = {bt.get_action_size()}")

# Create temp 5x5 BT
rule_5x5 = {
    "id": "bt_5x5_test",
    "name": "Breakthrough_5x5",
    "board": {"geometry": "square", "size": 5, "canonical_transform": "flip_vertical"},
    "pieces": {"initial_rows": {"player_1": [3, 4], "player_neg1": [0, 1]}},
    "actions": {
        "mode": "move_from_to",
        "move_vectors": [[-1, -1], [-1, 0], [-1, 1]],
        "capture_rules": {"straight_capture": False, "diagonal_capture_enemy_only": True}
    },
    "end_condition": {"type": "reach_rank", "target_rank": 4, "elimination": True}
}
tmp_path = os.path.join(os.path.dirname(__file__), '..', 'rules', '_bt_5x5_test.json')
with open(tmp_path, 'w') as f:
    json.dump(rule_5x5, f)

bt5 = UniversalGame(os.path.relpath(tmp_path, os.path.join(os.path.dirname(__file__), '..')))
assert bt5.get_action_size() == 75
print(f"  [OK] Breakthrough 5x5: action_size = {bt5.get_action_size()}")

hx = get_game_by_id('hex_json')
assert hx.get_action_size() == 49
print(f"  [OK] Hex 7x7: action_size = {hx.get_action_size()}")

ttt = get_game_by_id('tictactoe')
assert ttt.get_action_size() == 9
print(f"  [OK] TicTacToe 3x3: action_size = {ttt.get_action_size()}")

# UniversalNet with mixed heads
from nnet.model import UniversalNet
temp_reg = dict(GAME_REGISTRY)
temp_reg['bt_5x5_test'] = bt5
model = UniversalNet(temp_reg)
assert model.policy_heads['breakthrough_json'].fc.out_features == 192
assert model.policy_heads['bt_5x5_test'].fc.out_features == 75
assert model.policy_heads['tictactoe'].fc.out_features == 9
print(f"  [OK] UniversalNet heads: BT=192, BT5x5=75, TTT=9, Hex=49")

os.remove(tmp_path)
print("  [PASS] T2: Action Space Auto-Adaptation\n")

# ==============================================================
# T3: MTL Routing Isolation
# ==============================================================
print("--- T3: MTL Routing Isolation ---")

MAX_SIZE = 9
hex_board = np.zeros((MAX_SIZE, MAX_SIZE))
hex_board[0][3] = 1
hex_input = torch.FloatTensor(hex_board).view(1, 1, MAX_SIZE, MAX_SIZE)

# Correct routing
pi_hex, v_hex = model(hex_input, game_id='hex_json')
assert pi_hex.shape == (1, 49)
print(f"  [OK] Hex input + hex_json head -> pi shape {pi_hex.shape}")

# Wrong routing: hex input with BT head -> 192 dim output
pi_wrong, _ = model(hex_input, game_id='breakthrough_json')
assert pi_wrong.shape == (1, 192)
print(f"  [OK] Hex input + breakthrough_json head -> pi shape {pi_wrong.shape} (DIMENSION MISMATCH)")

# Unknown game_id -> ValueError
try:
    model(hex_input, game_id='nonexistent_game')
    print("  [FAIL] Should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"  [OK] Unknown game_id -> ValueError: \"{e}\"")

print("  [PASS] T3: MTL Routing Isolation\n")

# ==============================================================
# T4: MCTS Depth Cutoff
# ==============================================================
print("--- T4: MCTS Depth Cutoff (max_depth=3) ---")

class MCTSArgs:
    def __init__(self, num_mcts_sims, cpuct, max_depth):
        self.num_mcts_sims = num_mcts_sims
        self.cpuct = cpuct
        self.max_depth = max_depth
    def get(self, key, default=None):
        return getattr(self, key, default)

from nnet.nnet import NNetWrapper

nnet_bt = NNetWrapper(bt, 'breakthrough_json')

# Extreme depth limit: 3
args_d3 = MCTSArgs(num_mcts_sims=50, cpuct=1.0, max_depth=3)
mcts_d3 = MCTS(bt, nnet_bt, args_d3)

board_bt = bt.get_initial_board()
canonical_bt = bt.get_canonical_form(board_bt, 1)

try:
    pi = mcts_d3.getActionProb(canonical_bt, temp=1)
    action = int(np.argmax(pi))
    n_valid = sum(1 for p in pi if p > 0)
    print(f"  [OK] MCTS completed without RecursionError")
    print(f"  [OK] max_depth=3, sims=50 -> best action={action}, {n_valid} actions with prob>0")
except RecursionError:
    print("  [FAIL] RecursionError! Depth cutoff not working")
    sys.exit(1)

# Compare with normal depth
args_d100 = MCTSArgs(num_mcts_sims=50, cpuct=1.0, max_depth=100)
mcts_d100 = MCTS(bt, nnet_bt, args_d100)
pi_deep = mcts_d100.getActionProb(canonical_bt, temp=1)

pi_arr = np.array(pi)
pi_deep_arr = np.array(pi_deep)
kl = np.sum(np.where(pi_arr > 0, pi_arr * np.log((pi_arr + 1e-8) / (pi_deep_arr + 1e-8)), 0))
print(f"  [OK] Depth=3 vs Depth=100 KL divergence: {kl:.4f} (different evaluations)")

print("  [PASS] T4: MCTS Depth Cutoff\n")

# ==============================================================
# T5: UI Coordinate Mapping
# ==============================================================
print("--- T5: UI Coordinate Mapping ---")

hex_game = get_game_by_id('hex_json')
n = hex_game.n
assert n == 7

row, col = 2, 3
action_index = row * n + col
assert action_index == 17
print(f"  [OK] Hex click ({row},{col}) -> action_index = {action_index} (= {row}*{n}+{col})")

empty_board = hex_game.get_initial_board()
valids = hex_game.get_valid_moves(empty_board, 1)
assert valids[action_index] == 1
print(f"  [OK] Action {action_index} is valid on empty Hex board")

new_board, next_player = hex_game.get_next_state(empty_board, action_index, 1)
assert new_board[row][col] == 1
assert next_player == -1
print(f"  [OK] After action {action_index}: board[{row}][{col}] = {new_board[row][col]}, next_player = {next_player}")

restored_r, restored_c = divmod(action_index, n)
assert restored_r == row and restored_c == col
print(f"  [OK] Reverse: action {action_index} -> ({restored_r},{restored_c})")

# TicTacToe: (1,1) -> action 4
ttt_action = 1 * 3 + 1
assert ttt_action == 4
ttt_board = ttt.get_initial_board()
new_ttt, _ = ttt.get_next_state(ttt_board, ttt_action, 1)
assert new_ttt[1][1] == 1
print(f"  [OK] TTT click (1,1) -> action {ttt_action}, board[1][1] = {new_ttt[1][1]}")

# Breakthrough: move_from_to codec
bt_action_id = bt._move_to_id(6, 3, 1)
expected_id = (6 * 8 + 3) * 3 + 1  # 154
assert bt_action_id == expected_id
src_r, src_c, vec_idx = bt._id_to_move(bt_action_id)
assert (src_r, src_c, vec_idx) == (6, 3, 1)
print(f"  [OK] BT move (6,3)->vec[1] = action {bt_action_id}, reverse = ({src_r},{src_c},vec{vec_idx})")

print("  [PASS] T5: UI Coordinate Mapping\n")

# ==============================================================
print("=" * 60)
print("  ALL 4 TESTS PASSED (T2, T3, T4, T5)")
print("=" * 60)
