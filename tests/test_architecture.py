"""
UniversalNet MTL 架构单元测试

测试内容：
  1. 多头输出对齐 - 各策略头维度 + Softmax 归一化
  2. 价值头共享 - 同一输入通过不同 game_id 应得到相同 value
  3. 权重共享证明 - 扰动 backbone 权重后两条分支同步变化
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from breakthrough import BreakthroughGame  # noqa: F401
from hex_game import HexGame               # noqa: F401
from game import GAME_REGISTRY
from nnet.model import UniversalNet


def test_multihead_output_alignment():
    """验证各策略头输出维度和 Softmax 归一化"""
    print('[TEST 1] Multi-Head Output Alignment')
    print('-' * 40)

    net = UniversalNet(GAME_REGISTRY)
    net.eval()
    x = torch.randn(1, 1, 9, 9)

    with torch.no_grad():
        # Breakthrough head
        p_bt, v_bt = net(x, game_id='breakthrough')
        p_bt_probs = torch.exp(p_bt)
        bt_dim = p_bt.shape[1]
        bt_sum = p_bt_probs.sum().item()

        print(f'  [Breakthrough]')
        print(f'    Policy dim  : {bt_dim} (expect 192)')
        print(f'    Softmax sum : {bt_sum:.6f} (expect 1.0)')
        assert bt_dim == 192, f'FAIL: expected 192, got {bt_dim}'
        assert abs(bt_sum - 1.0) < 1e-5, f'FAIL: softmax sum = {bt_sum}'
        print(f'    >>> PASS')

        # Hex head
        p_hex, v_hex = net(x, game_id='hex')
        p_hex_probs = torch.exp(p_hex)
        hex_dim = p_hex.shape[1]
        hex_sum = p_hex_probs.sum().item()

        print(f'  [Hex]')
        print(f'    Policy dim  : {hex_dim} (expect 49)')
        print(f'    Softmax sum : {hex_sum:.6f} (expect 1.0)')
        assert hex_dim == 49, f'FAIL: expected 49, got {hex_dim}'
        assert abs(hex_sum - 1.0) < 1e-5, f'FAIL: softmax sum = {hex_sum}'
        print(f'    >>> PASS')

    print('[TEST 1] PASSED\n')


def test_shared_value_head():
    """验证共享价值头：同一输入经不同 game_id 应产生相同 value"""
    print('[TEST 2] Shared Value Head Consistency')
    print('-' * 40)

    net = UniversalNet(GAME_REGISTRY)
    net.eval()
    x = torch.randn(1, 1, 9, 9)

    with torch.no_grad():
        _, v_bt = net(x, game_id='breakthrough')
        _, v_hex = net(x, game_id='hex')

        v_bt_val = v_bt.item()
        v_hex_val = v_hex.item()

        print(f'  v (breakthrough) : {v_bt_val:.6f}')
        print(f'  v (hex)          : {v_hex_val:.6f}')
        assert abs(v_bt_val - v_hex_val) < 1e-6, \
            f'FAIL: values differ ({v_bt_val} vs {v_hex_val})'
        print(f'  >>> PASS (identical)')

    print('[TEST 2] PASSED\n')


def test_backbone_weight_sharing():
    """扰动 backbone 权重，证明两条分支共享同一主干"""
    print('[TEST 3] Backbone Weight Sharing Proof')
    print('-' * 40)

    net = UniversalNet(GAME_REGISTRY)
    net.eval()
    x = torch.randn(1, 1, 9, 9)

    with torch.no_grad():
        # 扰动前
        _, v_bt_before = net(x, game_id='breakthrough')
        _, v_hex_before = net(x, game_id='hex')
        v_bt_before = v_bt_before.item()
        v_hex_before = v_hex_before.item()

        print(f'  [Before]  v_bt = {v_bt_before:.6f},  v_hex = {v_hex_before:.6f}')

        # 扰动 backbone.conv1.weight
        net.backbone.conv1.weight.data += 10.0

        # 扰动后
        _, v_bt_after = net(x, game_id='breakthrough')
        _, v_hex_after = net(x, game_id='hex')
        v_bt_after = v_bt_after.item()
        v_hex_after = v_hex_after.item()

        print(f'  [After]   v_bt = {v_bt_after:.6f},  v_hex = {v_hex_after:.6f}')

        bt_changed = abs(v_bt_after - v_bt_before) > 1e-4
        hex_changed = abs(v_hex_after - v_hex_before) > 1e-4
        both_same = abs(v_bt_after - v_hex_after) < 1e-6

        print(f'  bt changed  : {bt_changed} (delta={abs(v_bt_after - v_bt_before):.6f})')
        print(f'  hex changed : {hex_changed} (delta={abs(v_hex_after - v_hex_before):.6f})')
        print(f'  still equal : {both_same}')

        assert bt_changed, 'FAIL: Breakthrough value unchanged after backbone perturbation'
        assert hex_changed, 'FAIL: Hex value unchanged after backbone perturbation'
        assert both_same, f'FAIL: values diverged ({v_bt_after} vs {v_hex_after})'
        print(f'  >>> PASS (backbone is truly shared)')

    print('[TEST 3] PASSED\n')


def main():
    print('=' * 60)
    print('  UniversalNet MTL Architecture Unit Tests')
    print('=' * 60)
    print()

    test_multihead_output_alignment()
    test_shared_value_head()
    test_backbone_weight_sharing()

    print('=' * 60)
    print('  ALL ARCHITECTURE TESTS PASSED')
    print('=' * 60)
    print()
    print('  Proven:')
    print('  [1] breakthrough head -> 192-dim, softmax sum = 1.0')
    print('  [2] hex head          ->  49-dim, softmax sum = 1.0')
    print('  [3] value head is shared (same output for same input)')
    print('  [4] backbone is shared (perturbation affects both branches)')


if __name__ == "__main__":
    main()
