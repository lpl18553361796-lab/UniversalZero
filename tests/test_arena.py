"""
Multi-Task Arena Verification Tests

Test 1: Routing Correctness
    - Run BT + Hex games, no dimension mismatch errors

Test 2: Evolution Decision Simulation
    - BT-strong / Hex-weak model -> REJECT update
    - Balanced model -> ACCEPT update

Test 3: Result Persistence
    - JSON report saved to assets/arena_results/
"""
import sys, os, json, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import GAME_REGISTRY, get_game_by_id
from nnet.nnet import NNetWrapper
from arena import MultiTaskArena, Arena, MCTSPlayer, RandomPlayer, _TaskProxy

print("=" * 60)
print("  Multi-Task Arena Verification Tests")
print("=" * 60)


# ==============================================================
# Test 1: Routing Correctness
# ==============================================================
print("\n--- Test 1: Routing Correctness ---")

bt = get_game_by_id('breakthrough_json')
hx = get_game_by_id('hex_json')
ttt = get_game_by_id('tictactoe')

games = {
    'breakthrough_json': bt,
    'hex_json': hx,
    'tictactoe': ttt,
}

# Create two NNetWrappers (new and old)
nnet_new = NNetWrapper(bt, 'breakthrough_json')
nnet_old = NNetWrapper(bt, 'breakthrough_json')

# Test _TaskProxy routing
print("  Testing _TaskProxy routing...")
for gid, game in games.items():
    proxy = _TaskProxy(nnet_new, gid, game)
    board = game.get_initial_board()
    canonical = game.get_canonical_form(board, 1)

    pi, v = proxy.predict(canonical)
    expected_action_size = game.get_action_size()
    assert len(pi) == expected_action_size, (
        f"{gid}: expected pi length {expected_action_size}, got {len(pi)}")
    assert pi.shape == (expected_action_size,)
    print(f"  [OK] {gid}: pi shape = {pi.shape}, v = {v[0]:.4f}")

# Verify original game_id unchanged after proxy calls
assert nnet_new.game_id == 'breakthrough_json', (
    f"game_id was mutated! Got {nnet_new.game_id}")
print("  [OK] NNetWrapper game_id preserved after proxy calls")

# Run actual arena matches (small: 2 games each)
print("  Running cross-game arena matches (2 per game)...")
arena = MultiTaskArena(games, num_games_per_task=2, num_sims=5)

try:
    report = arena.evaluate(nnet_new, nnet_old, verbose=True)
    print("  [OK] All 3 games completed without dimension errors")
except RuntimeError as e:
    print(f"  [FAIL] Dimension error: {e}")
    sys.exit(1)

# Verify report structure
assert 'results' in report
assert 'should_update' in report
assert 'reason' in report
for gid in games:
    assert gid in report['results']
    r = report['results'][gid]
    assert 'wins' in r and 'losses' in r and 'draws' in r
    assert 'win_rate' in r and 'elo' in r
    total = r['wins'] + r['losses'] + r['draws']
    assert total == 2, f"{gid}: expected 2 total games, got {total}"

print("  [PASS] Test 1: Routing Correctness\n")


# ==============================================================
# Test 2: Evolution Decision Simulation
# ==============================================================
print("--- Test 2: Evolution Decision Simulation ---")

# Case A: BT strong (80%) but Hex weak (20%) -> REJECT
print("  Case A: BT strong / Hex weak -> should REJECT")
results_a = {
    'breakthrough_json': {
        'wins': 8, 'losses': 2, 'draws': 0,
        'win_rate': 0.8, 'elo': 1200,
    },
    'hex_json': {
        'wins': 2, 'losses': 8, 'draws': 0,
        'win_rate': 0.2, 'elo': 800,
    },
}
should_update_a, reason_a = arena.check_replacement_threshold(results_a)
assert should_update_a is False, f"Expected REJECT, got ACCEPT: {reason_a}"
assert 'REJECT' in reason_a
assert 'hex_json' in reason_a  # The failing game should be mentioned
print(f"    Decision: {reason_a}")
print(f"    [OK] Correctly REJECTED (catastrophic forgetting in Hex)")

# Case B: Both moderate, one strong -> ACCEPT
print("  Case B: Both >= 45%, one >= 55% -> should ACCEPT")
results_b = {
    'breakthrough_json': {
        'wins': 6, 'losses': 4, 'draws': 0,
        'win_rate': 0.6, 'elo': 1100,
    },
    'hex_json': {
        'wins': 5, 'losses': 5, 'draws': 0,
        'win_rate': 0.5, 'elo': 1000,
    },
}
should_update_b, reason_b = arena.check_replacement_threshold(results_b)
assert should_update_b is True, f"Expected ACCEPT, got REJECT: {reason_b}"
assert 'ACCEPT' in reason_b
print(f"    Decision: {reason_b}")
print(f"    [OK] Correctly ACCEPTED (BT strong, Hex stable)")

# Case C: Both okay but none strong -> REJECT
print("  Case C: All >= 45% but none >= 55% -> should REJECT")
results_c = {
    'breakthrough_json': {
        'wins': 5, 'losses': 5, 'draws': 0,
        'win_rate': 0.5, 'elo': 1000,
    },
    'hex_json': {
        'wins': 5, 'losses': 5, 'draws': 0,
        'win_rate': 0.5, 'elo': 1000,
    },
}
should_update_c, reason_c = arena.check_replacement_threshold(results_c)
assert should_update_c is False, f"Expected REJECT, got ACCEPT: {reason_c}"
assert 'REJECT' in reason_c
print(f"    Decision: {reason_c}")
print(f"    [OK] Correctly REJECTED (no sufficient improvement)")

# Case D: Edge case - one game at exactly 45%, one at 55%
print("  Case D: Boundary (45% + 55%) -> should ACCEPT")
results_d = {
    'breakthrough_json': {
        'wins': 45, 'losses': 55, 'draws': 0,
        'win_rate': 0.45, 'elo': 965,
    },
    'hex_json': {
        'wins': 55, 'losses': 45, 'draws': 0,
        'win_rate': 0.55, 'elo': 1035,
    },
}
should_update_d, reason_d = arena.check_replacement_threshold(results_d)
assert should_update_d is True, f"Expected ACCEPT, got REJECT: {reason_d}"
print(f"    Decision: {reason_d}")
print(f"    [OK] Correctly ACCEPTED (boundary case)")

print("  [PASS] Test 2: Evolution Decision Simulation\n")


# ==============================================================
# Test 3: Result Persistence
# ==============================================================
print("--- Test 3: Result Persistence ---")

test_report_dir = os.path.join(os.path.dirname(__file__),
                                '..', 'assets', 'arena_results')

# Clean up any previous test files
test_filename = 'test_report.json'
test_filepath = os.path.join(test_report_dir, test_filename)
if os.path.exists(test_filepath):
    os.remove(test_filepath)

# Save report
saved_path = arena.save_report(
    report, folder=test_report_dir, filename=test_filename)

assert os.path.exists(saved_path), f"Report file not found: {saved_path}"
print(f"  [OK] Report saved: {saved_path}")

# Load and verify content
with open(saved_path, 'r', encoding='utf-8') as f:
    loaded = json.load(f)

assert 'results' in loaded
assert 'should_update' in loaded
assert 'reason' in loaded
for gid in games:
    assert gid in loaded['results']

print(f"  [OK] Report content verified (keys: {list(loaded.keys())})")
print(f"  [OK] Games in report: {list(loaded['results'].keys())}")

# Also test auto-filename (timestamp)
auto_path = arena.save_report(report, folder=test_report_dir)
assert os.path.exists(auto_path)
print(f"  [OK] Auto-named report: {os.path.basename(auto_path)}")

# Clean up
os.remove(test_filepath)
os.remove(auto_path)
print("  [OK] Test files cleaned up")

print("  [PASS] Test 3: Result Persistence\n")


# ==============================================================
print("=" * 60)
print("  ALL 3 TESTS PASSED")
print("=" * 60)
