import os
import sys
import pickle
import time
import subprocess
import numpy as np
from collections import deque
from random import shuffle
from tqdm import tqdm

class Coach:
    """
    分发版 Coach - 通过 subprocess 启动独立的 Worker 进程
    这是 Windows 上处理多进程 AI 训练最稳健的方案。
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.game_id = getattr(nnet, 'game_id', 'unknown')
        self.trainExamplesHistory = []

    def learn(self):
        num_workers = getattr(self.args, 'num_workers', 4)
        num_eps = self.args.numEps
        metrics = []
        
        # 解决问题3：平均分配任务，余数分给前面的 worker
        eps_per_worker = num_eps // num_workers
        remainder = num_eps % num_workers
        
        temp_dir = os.path.join(self.args.checkpoint, 'temp_data')
        os.makedirs(temp_dir, exist_ok=True)

        for i in range(1, self.args.numIters + 1):
            print(f'\n------ Iteration {i}/{self.args.numIters} ------')
            
            model_path = os.path.join(temp_dir, 'current_model.pth.tar')
            self.nnet.save_checkpoint(folder=temp_dir, filename='current_model.pth.tar')

            processes = []
            output_files = []
            project_root = os.getcwd()

            print(f"Launching {num_workers} workers (Total {num_eps} eps)...")
            for j in range(num_workers):
                output_path = os.path.join(temp_dir, f'worker_{j}.pkl')
                output_files.append(output_path)
                
                # 分配局数
                current_worker_eps = eps_per_worker + (1 if j < remainder else 0)
                
                cmd = [
                    sys.executable, 'self_play_worker.py',
                    '--project_root', project_root,
                    '--model_path', model_path,
                    '--game', self.game_id,
                    '--num_episodes', str(current_worker_eps),
                    '--num_mcts_sims', str(self.args.num_mcts_sims),
                    '--temp_threshold', str(self.args.tempThreshold),
                    '--cpuct', str(self.args.cpuct),
                    '--output_path', output_path,
                    '--seed', str(int(time.time()) + j)
                ]
                
                # 方案A：通过环境变量隔离 GPU，不再使用命令行参数
                env = os.environ.copy()
                if j > 0:
                    env['CUDA_VISIBLE_DEVICES'] = ''

                p = subprocess.Popen(cmd, env=env)
                processes.append((p, j)) # 记录进程和索引

            # 解决问题1 & 2：更高频率轮询 + 错误捕获
            finished_indices = set()
            print(f"Waiting for workers to finish...")
            with tqdm(total=num_workers, desc="Workers Progress") as pbar:
                while len(finished_indices) < num_workers:
                    time.sleep(0.5) # 0.5秒轮询，响应更实时
                    for p, idx in processes:
                        if idx not in finished_indices and p.poll() is not None:
                            if p.returncode != 0:
                                print(f"\n[!] 警告: Worker {idx} 异常退出 (ReturnCode: {p.returncode})")
                            finished_indices.add(idx)
                            pbar.update(1)

            # 汇总数据
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            for f_path in output_files:
                if os.path.exists(f_path):
                    try:
                        with open(f_path, 'rb') as f:
                            iterationTrainExamples += pickle.load(f)
                        os.remove(f_path)
                    except Exception as e:
                        print(f"读取文件 {f_path} 失败: {e}")

            # 5. 训练
            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > 20:
                self.trainExamplesHistory.pop(0)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            print(f"Training on {len(trainExamples)} samples...")
            loss_history = self.nnet.train(trainExamples)
            if loss_history:
                row = {'iteration': i, 'samples': len(trainExamples)}
                for key, values in loss_history.items():
                    row[key] = float(np.mean(values)) if values else None
                metrics.append(row)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

        return {'iterations': metrics}
