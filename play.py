import numpy as np
import sys
from breakthrough import BreakthroughGame
from nnet.nnet import NNetWrapper
from mcts import MCTS
from utils import dotdict
import os

def get_human_action(game, board):
    """
    获取人类输入
    格式：r1 c1 r2 c2 (例如: 6 0 5 0 表示从 (6,0) 走到 (5,0))
    """
    valid_moves = game.get_valid_moves(board, 1) # 获取白方合法动作
    
    while True:
        try:
            user_input = input("你的回合 (输入 '原行 原列 目标行 目标列', 如 6 0 5 0): ")
            if user_input.lower() in ['q', 'exit']: sys.exit()
            
            coords = list(map(int, user_input.split()))
            if len(coords) != 4:
                print("格式错误，请输入 4 个数字。")
                continue
                
            r1, c1, r2, c2 = coords
            
            # 1. 基础规则检查
            if r2 != r1 - 1:
                print(">>> 错误：白方只能往上走 (行号减 1)")
                continue
            
            dir_offset = c2 - c1
            if abs(dir_offset) > 1:
                print(">>> 错误：只能直走或斜走一格")
                continue
                
            # 2. 计算动作 ID
            # 公式: (行 * 8 + 列) * 3 + (方向偏移 + 1)
            # 方向偏移: -1(左), 0(前), 1(右) -> 映射为 0, 1, 2
            action_id = (r1 * 8 + c1) * 3 + (dir_offset + 1)
            
            # 3. 检查合法性 (是否被挡住、是否有子等)
            if 0 <= action_id < game.get_action_size() and valid_moves[action_id] == 1:
                return action_id
            else:
                print(">>> 错误：该动作无效 (可能目标格子有子，或者原格子没子)")
                
        except ValueError:
            print(">>> 输入无效，请输入数字。")

def main():
    print("=== 人机对战 (Human vs AlphaZero) ===")
    print("规则：你执白(W)在下方，往上攻。AI执黑(B)在上方，往下攻。")
    print("输入 q 退出。")
    
    # 1. 初始化
    game = BreakthroughGame()
    nnet = NNetWrapper(game, game_id='breakthrough')
    
    # 2. 加载模型
    checkpoint_path = './temp/'
    checkpoint_file = 'best.pth.tar'
    full_path = os.path.join(checkpoint_path, checkpoint_file)

    if os.path.exists(full_path):
        print(f"正在加载模型: {full_path} ...")
        nnet.load_checkpoint(checkpoint_path, checkpoint_file)
        print(">>> 模型加载成功！准备战斗！")
    else:
        print(f">>> 警告: 找不到模型文件 {full_path}")
        print(">>> AI 将使用未训练的随机大脑进行对战。")

    # 3. 配置 AI 思考参数
    args = dotdict({'num_mcts_sims': 50, 'cpuct': 1.0})
    mcts = MCTS(game, nnet, args)
    
    board = game.get_initial_board()
    player = 1 # 1:Human(白), -1:AI(黑)
    step = 0
    
    game.display(board)
    
    while True:
        step += 1
        print(f"\n----------- 第 {step} 步 -----------")
        
        # 检查全局胜负
        # get_game_ended(board, 1) 返回相对于白方的结果
        result = game.get_game_ended(board, 1)
        if result != 0:
            if result == 1: print("\n>>> 恭喜！你赢了！ (White Wins) <<<")
            else:           print("\n>>> 遗憾，AI 赢了！ (Black Wins) <<<")
            break

        if player == 1:
            # === 人类回合 (白) ===
            action = get_human_action(game, board)
            # 执行动作 (全局视角)
            board, _ = game.get_next_state(board, action, 1)
            
        else:
            # === AI 回合 (黑) ===
            print("AlphaZero 正在思考...", end="", flush=True)
            
            # 1. 视角转换：AI 认为自己是白方往上攻
            canonical_board = game.get_canonical_form(board, -1)
            
            # 2. MCTS 思考 (temp=0 竞技模式)
            probs = mcts.getActionProb(canonical_board, temp=0)
            action = np.argmax(probs)
            
            # 3. 执行动作 (在 AI 视角下)
            next_canon, _ = game.get_next_state(canonical_board, action, 1)
            
            # 4. 视角还原：翻转回全局
            board = game.get_canonical_form(next_canon, -1)
            print(" 完成。")
            
        game.display(board)
        player = -player

if __name__ == "__main__":
    main()
