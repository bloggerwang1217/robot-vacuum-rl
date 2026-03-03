"""
Training Profiler - 測量訓練各環節的耗時瓶頸
使用 train_dqn.py 的實際類別來測量
"""

import torch
import numpy as np
import random
import time
import argparse
from collections import defaultdict

from train_dqn import IndependentDQNAgent, MultiAgentTrainer
from gym import RobotVacuumGymEnv


class TimingStats:
    """收集和統計耗時"""
    def __init__(self):
        self.times = defaultdict(list)

    def record(self, name: str, duration: float):
        self.times[name].append(duration)

    def summary(self) -> str:
        lines = ["\n" + "="*60]
        lines.append("PROFILING SUMMARY")
        lines.append("="*60)

        # 計算總時間
        total_time = sum(sum(v) for v in self.times.values())

        # 按總耗時排序
        sorted_items = sorted(
            self.times.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )

        lines.append(f"{'Component':<30} {'Total(s)':<10} {'Mean(ms)':<10} {'%':<8} {'Count':<8}")
        lines.append("-"*60)

        for name, durations in sorted_items:
            total = sum(durations)
            mean_ms = (total / len(durations)) * 1000
            pct = (total / total_time) * 100 if total_time > 0 else 0
            count = len(durations)
            lines.append(f"{name:<30} {total:<10.3f} {mean_ms:<10.3f} {pct:<8.1f} {count:<8}")

        lines.append("-"*60)
        lines.append(f"{'TOTAL':<30} {total_time:<10.3f}")
        lines.append("="*60)

        return "\n".join(lines)


def run_profiling(num_episodes: int = 5, max_steps: int = 100):
    """執行 profiling - 使用實際的 train_dqn 類別"""

    print("="*60)
    print("TRAINING PROFILER (Optimized Code)")
    print("="*60)

    # 建立 mock args
    class Args:
        env_n = 5
        initial_energy = 100
        robot_0_energy = None
        robot_1_energy = None
        robot_2_energy = None
        robot_3_energy = None
        e_move = 1
        e_charge = 5
        e_collision = 3
        e_boundary = 50
        max_episode_steps = max_steps
        charger_positions = "2,2"  # 單一充電座
        lr = 0.0001
        gamma = 0.99
        n_step = 1
        use_epsilon_decay = False
        epsilon = 0.2
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        memory_size = 10000
        batch_size = 128
        replay_start_size = 200  # 較低以便測試 train
        target_update_frequency = 1000
        seed = 42
        save_dir = "./models/profiler_test"
        use_torch_compile = False  # 測試時關閉，避免 JIT overhead

    args = Args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Stats collector
    stats = TimingStats()

    # Environment
    print("Creating environment...")
    env = RobotVacuumGymEnv(
        n=args.env_n,
        initial_energy=args.initial_energy,
        e_move=args.e_move,
        e_charge=args.e_charge,
        e_collision=args.e_collision,
        n_steps=args.max_episode_steps,
        charger_positions=[(2, 2)]
    )

    # Agents
    print("Creating agents...")
    agent_ids = ['robot_0', 'robot_1', 'robot_2', 'robot_3']
    obs_dim = env.observation_space.shape[0]
    action_dim = 5

    agents = {
        agent_id: IndependentDQNAgent(agent_id, obs_dim, action_dim, device, args)
        for agent_id in agent_ids
    }

    # Pre-allocated buffers (同 MultiAgentTrainer)
    _obs_batch_np = np.zeros((4, obs_dim), dtype=np.float32)
    _obs_batch_tensor = torch.zeros(4, obs_dim, dtype=torch.float32, device=device)

    print(f"\nRunning {num_episodes} episodes, max {max_steps} steps each...")
    print("-"*60)

    global_step = 0

    for episode in range(num_episodes):
        # Reset
        t0 = time.perf_counter()
        observations, infos = env.reset()
        stats.record("env/reset", time.perf_counter() - t0)

        step_count = 0
        done = False

        while not done and step_count < max_steps:
            # ========== SELECT ACTIONS (Batch Optimized) ==========
            t_select_start = time.perf_counter()

            actions = [None, None, None, None]
            need_inference_indices = []

            # 決定 random actions
            t_epsilon_start = time.perf_counter()
            for i, agent_id in enumerate(agent_ids):
                epsilon = agents[agent_id].epsilon
                if random.random() < epsilon:
                    actions[i] = random.randint(0, action_dim - 1)
                else:
                    _obs_batch_np[len(need_inference_indices)] = observations[agent_id]
                    need_inference_indices.append(i)
            stats.record("select_action/epsilon_check", time.perf_counter() - t_epsilon_start)

            # 批量推理
            if need_inference_indices:
                n_infer = len(need_inference_indices)

                # CPU -> GPU transfer
                t_transfer_start = time.perf_counter()
                _obs_batch_tensor[:n_infer].copy_(
                    torch.from_numpy(_obs_batch_np[:n_infer]), non_blocking=True
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                stats.record("select_action/cpu_to_gpu", time.perf_counter() - t_transfer_start)

                # Forward passes
                t_forward_start = time.perf_counter()
                with torch.no_grad():
                    for idx, agent_idx in enumerate(need_inference_indices):
                        agent_id = agent_ids[agent_idx]
                        obs_single = _obs_batch_tensor[idx:idx+1]
                        q_values = agents[agent_id].q_net(obs_single)
                        actions[agent_idx] = q_values.argmax().item()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                stats.record("select_action/forward", time.perf_counter() - t_forward_start)

            stats.record("select_action/total", time.perf_counter() - t_select_start)

            # ========== ENVIRONMENT STEP ==========
            t0 = time.perf_counter()
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            stats.record("env/step", time.perf_counter() - t0)

            # ========== REMEMBER ==========
            t0 = time.perf_counter()
            for i, agent_id in enumerate(agent_ids):
                agents[agent_id].remember(
                    observations[agent_id],
                    actions[i],
                    rewards[agent_id],
                    next_obs[agent_id],
                    terms[agent_id]
                )
            stats.record("remember", time.perf_counter() - t0)

            # ========== TRAIN ==========
            t_train_start = time.perf_counter()
            for agent_id in agent_ids:
                train_result = agents[agent_id].train_step(args.replay_start_size)
            train_time = time.perf_counter() - t_train_start
            if train_time > 0.0001:  # 只記錄有訓練的 step
                stats.record("train/total", train_time)

            observations = next_obs
            step_count += 1
            global_step += 1

            # Check termination
            alive = sum(1 for a in agent_ids if not terms[a])
            if alive <= 1 or any(truncs.values()):
                done = True

        print(f"Episode {episode+1}/{num_episodes} done, steps: {step_count}, global_step: {global_step}")

    # Print summary
    print(stats.summary())

    # Additional analysis
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)

    # 計算各類別佔比
    categories = {
        'env': ['env/reset', 'env/step'],
        'select_action': [k for k in stats.times.keys() if k.startswith('select_action')],
        'remember': ['remember'],
        'train': [k for k in stats.times.keys() if k.startswith('train')]
    }

    total_time = sum(sum(v) for v in stats.times.values())

    print(f"\n{'Category':<20} {'Time(s)':<12} {'%':<10}")
    print("-"*40)

    for cat_name, keys in categories.items():
        cat_time = sum(sum(stats.times[k]) for k in keys if k in stats.times)
        pct = (cat_time / total_time) * 100 if total_time > 0 else 0
        print(f"{cat_name:<20} {cat_time:<12.3f} {pct:<10.1f}")

    print("-"*40)
    print(f"{'TOTAL':<20} {total_time:<12.3f}")

    # GPU utilization hint
    if device.type == 'cuda':
        print("\n[Hint] 若 train/* 佔比低，表示 GPU 大部分時間在等待 CPU")
        print("[Hint] 若 env/step 佔比高，Vectorized Env 會有顯著效益")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    run_profiling(num_episodes=args.episodes, max_steps=args.max_steps)
