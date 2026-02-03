import numpy as np
from hex_game import HexGame

def main():
    print("=== Hex 逻辑验证 ===")
    g = HexGame(n=5)
    b = g.get_initial_board()
    
    print("1. 模拟白方连通上下...")
    # 制造一条白色的竖线
    for r in range(5): b[r][1] = 1
    g.display(b)
    
    if g.get_game_ended(b, 1) == 1:
        print(">>> [PASS] 白方获胜判定正确！")
    else:
        print(">>> [FAIL] 判定失败")

    print("\n2. 模拟黑方连通左右...")
    b = g.get_initial_board()
    # 制造一条黑色的横线
    for c in range(5): b[2][c] = -1
    g.display(b)
    
    # 注意：黑方连左右，get_game_ended 应该能识别
    if g.get_game_ended(b, -1) == 1:
        print(">>> [PASS] 黑方获胜判定正确！")
    else:
        print(">>> [FAIL] 判定失败")

if __name__ == "__main__":
    main()
