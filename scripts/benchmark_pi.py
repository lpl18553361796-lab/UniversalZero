import os
import sys
import time
import platform
import json
import argparse
import numpy as np

# ────────────────────────────────────────────────────────
# Helper to read Linux RSS memory directly without psutil
# ────────────────────────────────────────────────────────
def get_memory_usage_mb():
    # Primary: check Linux /proc/self/status which is 0-dependency and reliable on Raspberry Pi
    try:
        if os.path.exists('/proc/self/status'):
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        # VmRSS:     12345 kB
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0  # KB -> MB
    except Exception:
        pass
    
    # Secondary: fallback to psutil for debugging on Windows/macOS hosts
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)  # B -> MB
    except ImportError:
        return 0.0

# ────────────────────────────────────────────────────────
# Stage 1: Cold start baseline memory
# ────────────────────────────────────────────────────────
baseline_mem = get_memory_usage_mb()

# Stage 2: Track Torch loading memory footprint
print("Loading torch modules...")
t0_torch = time.time()
import torch
torch_load_time = time.time() - t0_torch
torch_load_mem = get_memory_usage_mb()
torch_delta_mem = torch_load_mem - baseline_mem

# --- Resolve project imports ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    sys.path.append(os.path.join(_project_root, "games"))
    sys.path.append(os.path.join(_project_root, "core"))

from game import get_game_by_id
from nnet.nnet import NNetWrapper
from core.arena import MCTSPlayer

def run_pi_benchmark(model_path, game_id='hex', num_sims=50, num_eval_steps=10):
    print(f"\n==========================================")
    print(f"  Raspberry Pi Resource Benchmark Utility ")
    print(f"==========================================")
    
    # Validate model
    if not os.path.exists(model_path):
        print(f"[!] Warning: Model path '{model_path}' not found!")
        print("    Please copy your trained .pth.tar model file to this path.")
        print("    For demo purposes, we will proceed with an UNTRAINED (random-initialized) network model.")
        use_random_init = True
    else:
        use_random_init = False
        
    # Setup game
    game = get_game_by_id(game_id)
    
    # Stage 3: Load network & model file
    t0_model = time.time()
    nnet = NNetWrapper(game, game_id)
    
    if not use_random_init:
        print(f"Loading checkpoint weights from: {model_path}")
        nnet.load_checkpoint(folder=os.path.dirname(model_path), filename=os.path.basename(model_path))
    else:
        print("Initializing network model with random weights...")
        
    model_load_time = time.time() - t0_model
    model_load_mem = get_memory_usage_mb()
    model_delta_mem = model_load_mem - torch_load_mem
    
    # Stage 4: Set up MCTS search and evaluate latency
    player = MCTSPlayer(game, nnet, num_sims=num_sims)
    
    # Generate typical states to run decisions on
    board = game.get_initial_board()
    cur_player = 1
    
    latencies = []
    mem_during_search = []
    
    print(f"Executing {num_eval_steps} MCTS benchmark decisions ({num_sims} simulations/move)...")
    
    # Run a dummy step to warm up CUDA/inference engines
    _ = player.play(board)
    
    for i in range(num_eval_steps):
        # Obtain canonical board state
        canonical = game.get_canonical_form(board, cur_player)
        
        # Benchmark time
        t_start = time.perf_counter()
        action = player.play(canonical)
        t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000.0
        latencies.append(latency_ms)
        
        # Track memory peaks
        mem_during_search.append(get_memory_usage_mb())
        
        # Transition state
        next_canonical, _ = game.get_next_state(canonical, action, 1)
        board = game.get_canonical_form(next_canonical, cur_player)
        cur_player = -cur_player
        
        # If game ends, reset to keep benchmarking
        if game.get_game_ended(board, cur_player) != 0:
            board = game.get_initial_board()
            cur_player = 1
            
        print(f"  Step {i+1}/{num_eval_steps}: {latency_ms:.1f} ms | Mem: {mem_during_search[-1]:.1f} MB")
        
    # Analyze results
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    peak_mem = np.max(mem_during_search)
    search_delta_mem = peak_mem - model_load_mem
    
    # Simulations Per Second (SPS)
    simulations_per_second = (num_sims) / (avg_latency / 1000.0)
    avg_sim_ms = avg_latency / num_sims
    
    # Compile performance results report
    report = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'os': platform.system(),
            'release': platform.release(),
            'machine_architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'device_type': 'CPU' if not torch.cuda.is_available() else 'GPU',
        },
        'memory_profile_mb': {
            'baseline_cold_start': round(baseline_mem, 2),
            'torch_imported': round(torch_load_mem, 2),
            'torch_library_overhead': round(torch_delta_mem, 2),
            'model_loaded': round(model_load_mem, 2),
            'model_parameters_overhead': round(model_delta_mem, 2),
            'active_mcts_search_peak': round(peak_mem, 2),
            'mcts_runtime_tree_overhead': round(search_delta_mem, 2),
            'total_memory_footprint': round(peak_mem, 2)
        },
        'inference_profile': {
            'torch_load_time_seconds': round(torch_load_time, 2),
            'model_load_time_seconds': round(model_load_time, 2),
            'average_decision_latency_ms': round(avg_latency, 2),
            'minimum_decision_latency_ms': round(min_latency, 2),
            'maximum_decision_latency_ms': round(max_latency, 2),
            'mcts_simulations_budget': num_sims,
            'simulations_per_second': round(simulations_per_second, 2),
            'single_simulation_latency_ms': round(avg_sim_ms, 3)
        }
    }
    
    # Save Report
    save_path = 'experiment_results/pi_benchmark_results.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Output formatted report suitable for copy-pasting to Thesis Section 5.1
    print(f"\n=======================================================")
    print(f"        BENCHMARK REPORT (SECTION 5.1 RESOURCE PROFILE) ")
    print(f"=======================================================")
    print(f"Hardware Profile:")
    print(f"  Architecture      : {report['metadata']['machine_architecture']}")
    print(f"  Execution Device  : {report['metadata']['device_type']}")
    print(f"\nMemory Allocation Profile:")
    print(f"  Cold Start Memory : {report['memory_profile_mb']['baseline_cold_start']:.2f} MB")
    print(f"  PyTorch Overhead  : +{report['memory_profile_mb']['torch_library_overhead']:.2f} MB (Total: {report['memory_profile_mb']['torch_imported']:.2f} MB)")
    print(f"  Model Loading     : +{report['memory_profile_mb']['model_parameters_overhead']:.2f} MB (Total: {report['memory_profile_mb']['model_loaded']:.2f} MB)")
    print(f"  Active MCTS Search: +{report['memory_profile_mb']['mcts_runtime_tree_overhead']:.2f} MB (Total Peak: {report['memory_profile_mb']['active_mcts_search_peak']:.2f} MB)")
    print(f"  Total Peak RSS    : {report['memory_profile_mb']['total_memory_footprint']:.2f} MB")
    print(f"\nInference Performance Profile:")
    print(f"  Avg Move Latency  : {report['inference_profile']['average_decision_latency_ms']:.1f} ms (Range: {report['inference_profile']['minimum_decision_latency_ms']:.1f} - {report['inference_profile']['maximum_decision_latency_ms']:.1f} ms)")
    print(f"  Search Budget     : {report['inference_profile']['mcts_simulations_budget']} simulations per decision step")
    print(f"  Inference Rate    : {report['inference_profile']['simulations_per_second']:.2f} MCTS Sims / Sec")
    print(f"  Avg Sim Latency   : {report['inference_profile']['single_simulation_latency_ms']:.3f} ms per simulation")
    print(f"=======================================================")
    print(f"Results successfully saved to: {save_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniversalZero Raspberry Pi Benchmark tool')
    parser.add_argument('--model', type=str, default='final_models/transfer_hex_final.pth.tar',
                        help='Path to the model to load for testing')
    parser.add_argument('--game', type=str, default='hex', help='Game ID, default: hex')
    parser.add_argument('--sims', type=int, default=50, help='MCTS simulations per decision step (default: 50)')
    parser.add_argument('--steps', type=int, default=10, help='Number of evaluation steps to run (default: 10)')
    
    args = parser.parse_args()
    
    run_pi_benchmark(
        model_path=args.model,
        game_id=args.game,
        num_sims=args.sims,
        num_eval_steps=args.steps
    )
