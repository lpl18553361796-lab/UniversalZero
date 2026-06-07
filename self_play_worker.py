import argparse
import os
import pickle
import random
import sys
import numpy as np
import torch

def setup_paths(project_root):
    print(f"[DEBUG] project_root: {project_root}", flush=True)
    for path in [project_root,
                 os.path.join(project_root, "games"),
                 os.path.join(project_root, "core"),
                 os.path.join(project_root, "ui")]:
        if path not in sys.path:
            sys.path.insert(0, path)
    print(f"[DEBUG] sys.path[0:5] = {sys.path[0:5]}", flush=True)

def play_one_episode(game, nnet, game_id, mcts_args, temp_threshold, MCTS):
    """串行跑一局完整自对弈"""
    trainExamples = []
    board = game.get_initial_board()
    curPlayer = 1
    episodeStep = 0

    # 这里的 pipe 必须为 None，走本地同步推理
    mcts = MCTS(game, nnet, mcts_args, pipe=None)

    while True:
        episodeStep += 1
        canonicalBoard = game.get_canonical_form(board, curPlayer)
        temp = int(episodeStep < temp_threshold)
        pi = mcts.getActionProb(canonicalBoard, temp=temp)

        trainExamples.append([canonicalBoard, curPlayer, pi, None, game_id])

        action = np.random.choice(len(pi), p=pi)
        next_state_canonical, _ = game.get_next_state(canonicalBoard, action, 1)
        board = game.get_canonical_form(next_state_canonical, curPlayer)
        curPlayer = -curPlayer

        r = game.get_game_ended(board, 1)
        if r != 0:
            # 返回带胜负结果的样本
            return [(x[0], x[2], r * x[1], x[4]) for x in trainExamples]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root',   required=True)
    parser.add_argument('--model_path',      required=True)
    parser.add_argument('--game',            required=True)
    parser.add_argument('--num_episodes',    type=int, required=True)
    parser.add_argument('--num_mcts_sims',   type=int, default=200)
    parser.add_argument('--temp_threshold',  type=int, default=15)
    parser.add_argument('--cpuct',           type=float, default=1.0)
    parser.add_argument('--max_depth',       type=int, default=100)
    parser.add_argument('--output_path',     required=True)
    parser.add_argument('--seed',            type=int, default=0)
    args = parser.parse_args()

    # 路径设置必须在 import 项目模块前完成
    setup_paths(args.project_root)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 延迟导入项目模块
    from game import GAME_REGISTRY
    from nnet.nnet import NNetWrapper
    from mcts import MCTS
    from utils import dotdict

    # 构造参数对象
    mcts_args = dotdict({
        'num_mcts_sims': args.num_mcts_sims,
        'cpuct':         args.cpuct,
        'max_depth':     args.max_depth,
        'lr':            0.001,
    })

    # 准备游戏和网络
    game_entry = GAME_REGISTRY[args.game]
    game = game_entry() if isinstance(game_entry, type) else game_entry

    nnet = NNetWrapper(game, args.game, args=mcts_args)
    nnet.load_checkpoint(
        folder=os.path.dirname(args.model_path),
        filename=os.path.basename(args.model_path),
    )

    # 执行自对弈
    all_examples = []
    for ep in range(args.num_episodes):
        examples = play_one_episode(game, nnet, args.game, mcts_args, args.temp_threshold, MCTS)
        all_examples.extend(examples)
        print(f"[W{args.seed}] ep {ep+1}/{args.num_episodes} -> {len(examples)} steps", flush=True)

    # 将结果持久化到磁盘
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(all_examples, f)

    print(f"[W{args.seed}] DONE: {len(all_examples)} samples -> {args.output_path}", flush=True)

if __name__ == '__main__':
    main()
