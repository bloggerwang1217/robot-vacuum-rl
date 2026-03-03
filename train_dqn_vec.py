"""
Vectorized DQN Training Script for Multi-Robot Energy Survival Environment

優化版本：
1. 支援 N 個環境並行（Vectorized）
2. 共享 Replay Buffer（所有 agents 共用）
3. 可調整訓練頻率（train-frequency）
4. 減少 wandb logging 頻率
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import argparse
import wandb
from collections import deque
from typing import Dict, List, Tuple

# Import the DQN network and components from dqn.py
from dqn import DQN, init_weights

# Import environments
from gym import RobotVacuumGymEnv
from vec_env import VectorizedRobotVacuumEnv, SubprocVecEnv
from batch_env import BatchRobotVacuumEnv


class SharedBufferDQNAgent:
    """
    DQN Agent that uses a shared replay buffer
    
    與 IndependentDQNAgent 的差異：
    - 不自帶 replay buffer，使用外部共享 buffer
    - 移除 remember() 方法，由 trainer 統一管理
    """
    def __init__(self,
                 agent_id: str,
                 observation_dim: int,
                 action_dim: int,
                 device: torch.device,
                 args: argparse.Namespace):
        self.agent_id = agent_id
        self.device = device
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # DQN hyperparameters
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.n_step = getattr(args, 'n_step', 1)

        # Epsilon configuration
        self.use_epsilon_decay = args.use_epsilon_decay
        if self.use_epsilon_decay:
            self.epsilon = args.epsilon_start
            self.epsilon_start = args.epsilon_start
            self.epsilon_end = args.epsilon_end
            self.epsilon_decay = args.epsilon_decay
        else:
            self.epsilon = args.epsilon

        # Eval epsilon
        self.eval_epsilon = getattr(args, 'eval_epsilon', 0.0)

        # Network architecture
        self.q_net = DQN(action_dim, observation_dim).to(device)
        self.target_net = DQN(action_dim, observation_dim).to(device)
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Compile models (optional)
        use_compile = getattr(args, 'use_torch_compile', False)
        if use_compile and device.type == 'cuda' and hasattr(torch, 'compile'):
            self.q_net = torch.compile(self.q_net, mode='default')
            self.target_net = torch.compile(self.target_net, mode='default')

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Mixed Precision Training
        self.use_amp = device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Counters
        self.train_count = 0

    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """
        Execute one training step using provided batch

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones, actual_n_steps) as tensors

        Returns:
            Training statistics
        """
        states, actions, rewards, next_states, dones, actual_n_steps = batch
        self.train_count += 1

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # 計算 per-sample 的 gamma^k（k = 實際累積步數）
                    gamma_n = self.gamma ** actual_n_steps
                    next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                    target_q_values = rewards + gamma_n * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # 計算 per-sample 的 gamma^k（k = 實際累積步數）
                gamma_n = self.gamma ** actual_n_steps
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                target_q_values = rewards + gamma_n * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_std': q_values.std().item()
        }

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_epsilon_by_progress(self, progress: float):
        """Set epsilon based on training progress (0.0 to 1.0)"""
        if self.use_epsilon_decay:
            # Linear decay from epsilon_start to epsilon_end
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def save(self, filepath: str):
        # 若 torch.compile 已包裝成 OptimizedModule，取出原始權重存檔
        net = self.q_net._orig_mod if hasattr(self.q_net, '_orig_mod') else self.q_net
        torch.save(net.state_dict(), filepath)

    def load(self, filepath: str):
        state = torch.load(filepath, map_location=self.device, weights_only=True)
        # 若 torch.compile 已包裝成 OptimizedModule，載入原始 module
        net = self.q_net._orig_mod if hasattr(self.q_net, '_orig_mod') else self.q_net
        net.load_state_dict(state)
        self.target_net.load_state_dict(self.q_net.state_dict())


class NumpyReplayBuffer:
    """
    Numpy ring buffer — 比 deque 快約 3x。
    避免 sample_batch 每次 list(deque) 的 O(capacity) 轉換。
    """
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self._states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)
        self._n_steps     = np.ones(capacity,  dtype=np.float32)
        self._ptr  = 0
        self._size = 0

    def append(self, transition):
        state, action, reward, next_state, done, n_step = transition
        i = self._ptr
        self._states[i]      = state
        self._next_states[i] = next_state
        self._actions[i]     = action
        self._rewards[i]     = reward
        self._dones[i]       = float(done)
        self._n_steps[i]     = n_step
        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def append_batch(self, states, actions, rewards, next_states, dones, n_steps):
        """一次寫入 n 筆 transition，避免 Python for 迴圈。"""
        n = len(states)
        if n == 0:
            return
        idx = (self._ptr + np.arange(n)) % self.capacity
        self._states[idx]      = states
        self._next_states[idx] = next_states
        self._actions[idx]     = actions
        self._rewards[idx]     = rewards
        self._dones[idx]       = dones
        self._n_steps[idx]     = n_steps
        self._ptr  = int((self._ptr + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        return (self._states[idx], self._actions[idx], self._rewards[idx],
                self._next_states[idx], self._dones[idx], self._n_steps[idx])

    def __len__(self):
        return self._size


class VectorizedMultiAgentTrainer:
    """
    Vectorized Multi-Agent DQN Trainer
    
    支援：
    - N 個環境並行
    - 共享 Replay Buffer
    - 可調整訓練頻率
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.num_envs = args.num_envs
        self.train_frequency = args.train_frequency

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        print(f"Number of parallel environments: {self.num_envs}")
        print(f"Train frequency: every {self.train_frequency} steps")

        # Random seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Number of robots
        self.num_robots = args.num_robots

        # Prepare environment config (only for the robots that exist)
        all_robot_energies = [
            args.robot_0_energy,
            args.robot_1_energy,
            args.robot_2_energy,
            args.robot_3_energy,
        ]
        robot_energies = all_robot_energies[:self.num_robots]

        all_robot_speeds = [
            args.robot_0_speed,
            args.robot_1_speed,
            args.robot_2_speed,
            args.robot_3_speed,
        ]
        robot_speeds = all_robot_speeds[:self.num_robots]
        self.robot_speeds = robot_speeds  # turns per step for each robot

        # Scripted robots: always play STAY, not trained
        scripted_str = getattr(args, 'scripted_robots', '')
        self.scripted_robots = set(int(x) for x in scripted_str.split(',') if x.strip()) if scripted_str else set()

        # Random robots: random actions, not trained
        random_str = getattr(args, 'random_robots', '')
        self.random_robots = set(int(x) for x in random_str.split(',') if x.strip()) if random_str else set()

        # Flee robots: flee heuristic, not trained
        flee_str = getattr(args, 'flee_robots', '')
        self.flee_robots = set(int(x) for x in flee_str.split(',') if x.strip()) if flee_str else set()

        # Safe-random robots: wall-avoiding random walk, not trained
        safe_random_str = getattr(args, 'safe_random_robots', '')
        self.safe_random_robots = set(int(x) for x in safe_random_str.split(',') if x.strip()) if safe_random_str else set()

        for label, s in [('STAY', self.scripted_robots), ('RANDOM', self.random_robots),
                         ('SAFE_RANDOM', self.safe_random_robots), ('FLEE', self.flee_robots)]:
            if s:
                print(f"  robots [{sorted(s)}] → {label} (not trained)")

        charger_positions = None
        if args.charger_positions is not None:
            try:
                charger_positions = []
                for pos_str in args.charger_positions.split(';'):
                    y, x = map(int, pos_str.split(','))
                    charger_positions.append((y, x))
                print(f"Using custom charger positions: {charger_positions}")
            except Exception as e:
                print(f"Error parsing charger positions: {e}")
                charger_positions = None

        env_kwargs = {
            'n': args.env_n,
            'num_robots': self.num_robots,
            'initial_energy': args.initial_energy,
            'robot_energies': robot_energies,
            'robot_speeds': robot_speeds,
            'e_move': args.e_move,
            'e_charge': args.e_charge,
            'e_collision': args.e_collision,
            'e_boundary': args.e_boundary,
            'n_steps': args.max_episode_steps,
            'charger_positions': charger_positions,
            'dust_max': args.dust_max,
            'dust_rate': args.dust_rate,
            'dust_epsilon': args.dust_epsilon,
            'charger_dust_max_ratio': args.charger_dust_max_ratio,
            'charger_dust_rate_ratio': args.charger_dust_rate_ratio,
            'dust_reward_scale': args.dust_reward_scale,
            'dust_enabled': not args.no_dust,
            'exclusive_charging': args.exclusive_charging,
            'random_start_robots': set(
                int(x) for x in getattr(args, 'random_start_robots', '').split(',') if x.strip()
            ),
        }

        # Create vectorized environment
        num_workers = getattr(args, 'num_workers', 1)
        use_batch   = getattr(args, 'batch_env', False)
        if self.num_envs == 1:
            # 單環境模式（向後兼容）
            self.use_vec_env = False
            self.env = RobotVacuumGymEnv(**env_kwargs)
            observation_dim = self.env.observation_space.shape[0]
        elif use_batch:
            # 全 numpy 批次環境（最快，無 Python 物件 overhead）
            self.use_vec_env = True
            self.env = BatchRobotVacuumEnv(self.num_envs, env_kwargs)
            observation_dim = self.env.observation_space.shape[0]
            print(f"Using BatchRobotVacuumEnv (fully vectorized numpy)")
        elif num_workers > 1:
            # Multiprocessing 模式：num_workers 個 subprocess 並行跑 env
            self.use_vec_env = True
            self.env = SubprocVecEnv(self.num_envs, env_kwargs, num_workers)
            observation_dim = self.env.observation_space.shape[0]
        else:
            # Vectorized 模式（單 process，原始）
            self.use_vec_env = True
            self.env = VectorizedRobotVacuumEnv(self.num_envs, env_kwargs)
            observation_dim = self.env.observation_space.shape[0]

        # Initialize agents (only for the robots that exist)
        self.agent_ids = [f'robot_{i}' for i in range(self.num_robots)]
        self.n_agents = self.num_robots
        action_dim = 5

        self.agents = {}
        for agent_id in self.agent_ids:
            self.agents[agent_id] = SharedBufferDQNAgent(
                agent_id=agent_id,
                observation_dim=observation_dim,
                action_dim=action_dim,
                device=self.device,
                args=args
            )

        # Load pre-trained models if specified (for curriculum phase continuation)
        load_dir = getattr(args, 'load_model_dir', None)
        if load_dir:
            print(f"Loading models from: {load_dir}")
            for i, agent_id in enumerate(self.agent_ids):
                model_path = os.path.join(load_dir, f"{agent_id}.pt")
                if os.path.exists(model_path):
                    self.agents[agent_id].load(model_path)
                    print(f"  Loaded {agent_id} from {model_path}")
                else:
                    print(f"  WARNING: {model_path} not found, using fresh weights for {agent_id}")

        # Independent Replay Buffer (每個 agent 獨立) — numpy ring buffer
        self.memories = {
            agent_id: NumpyReplayBuffer(args.memory_size, observation_dim)
            for agent_id in self.agent_ids
        }

        # N-step buffer (per environment, per agent)
        self.n_step = getattr(args, 'n_step', 1)
        if self.n_step > 1:
            if self.use_vec_env:
                # (num_envs, n_agents) 的 n-step buffers
                self.n_step_buffers = [[deque(maxlen=self.n_step) for _ in range(self.n_agents)] 
                                       for _ in range(self.num_envs)]
            else:
                self.n_step_buffers = [[deque(maxlen=self.n_step) for _ in range(self.n_agents)]]
        else:
            self.n_step_buffers = None

        # 向量化 n-step buffer（vec_env 模式專用，取代 Python deque）
        if self.n_step > 1 and self.use_vec_env:
            _n, _k, _d = self.num_envs, self.n_step, observation_dim
            self.ns_states      = np.zeros((_n, self.n_agents, _k, _d), dtype=np.float32)
            self.ns_next_states = np.zeros((_n, self.n_agents, _k, _d), dtype=np.float32)
            self.ns_actions     = np.zeros((_n, self.n_agents, _k),     dtype=np.int32)
            self.ns_rewards     = np.zeros((_n, self.n_agents, _k),     dtype=np.float32)
            self.ns_dones       = np.zeros((_n, self.n_agents, _k),     dtype=bool)
            self.ns_ptr         = np.zeros((_n, self.n_agents),         dtype=np.int32)
            self.ns_count       = np.zeros((_n, self.n_agents),         dtype=np.int32)

        # 預算 gamma^k 權重（n-step return 向量化用）
        if self.n_step > 1:
            _gamma = self.agents[self.agent_ids[0]].gamma
            self._gamma_weights = np.array([_gamma**k for k in range(self.n_step)], dtype=np.float32)
            self._gamma_scalar  = _gamma
        else:
            self._gamma_weights = None
            self._gamma_scalar  = None

        # Training counters
        self.global_step = 0
        # Parse episode offset from load_model_dir (e.g. "episode_2000" → offset=2000)
        # so that checkpoint names continue from the last run (episode_2500, 3000, ...)
        self.episode_offset = 0
        if load_dir:
            dirname = os.path.basename(load_dir.rstrip('/'))
            parts = dirname.replace('_interrupted', '').split('_')
            if parts[0] == 'episode' and len(parts) >= 2:
                try:
                    self.episode_offset = int(parts[1])
                    print(f"  Episode offset: {self.episode_offset} (checkpoints will continue from here)")
                except ValueError:
                    pass
        self.total_episodes = self.episode_offset

        # Cumulative death counter
        self.cumulative_deaths = {agent_id: 0 for agent_id in self.agent_ids}

        # Model saving
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Best model tracking
        self.best_metric = float('-inf')

        # Pre-allocated tensors for batch inference
        if self.use_vec_env:
            self._obs_tensor = torch.zeros(self.num_envs, self.n_agents, observation_dim, 
                                          dtype=torch.float32, device=self.device)
        else:
            self._obs_tensor = torch.zeros(self.n_agents, observation_dim,
                                          dtype=torch.float32, device=self.device)

        # Episode statistics tracking (for logging)
        _n = self.num_envs if self.use_vec_env else 1
        self._episode_rewards = np.zeros((_n, self.n_agents))
        self._episode_steps = np.zeros(_n, dtype=np.int32)
        self._episode_immediate_kills = np.zeros(_n, dtype=np.int32)
        self._episode_active_collisions = np.zeros((_n, self.n_agents), dtype=np.int32)
        self._robot0_death_step = np.full(_n, -1, dtype=np.int32)  # -1 = alive

    def remember(self, state, action, reward, next_state, done, env_idx=0, agent_idx=0):
        """
        Store experience to agent's independent replay buffer with N-step support

        Replay buffer 格式: (state, action, reward, next_state, done, actual_n_step)
        - actual_n_step: 實際累積的步數（用於計算正確的 γ^k）

        Episode 結束時的特殊處理：
        - 將 buffer 中所有剩餘的 transition 都存入
        - 確保直接導致死亡的動作能被學習到
        """
        agent_id = self.agent_ids[agent_idx]
        agent_memory = self.memories[agent_id]
        gamma = self.agents[agent_id].gamma

        if self.n_step == 1:
            # 1-step: actual_n_step = 1
            agent_memory.append((state, action, reward, next_state, done, 1))
        else:
            buffer = self.n_step_buffers[env_idx][agent_idx]
            buffer.append((state, action, reward, next_state, done))

            if done:
                # Episode 結束：backward recurrence O(k) 取代原本 O(k²) 雙層迴圈
                buffer_list = list(buffer)
                k = len(buffer_list)
                _, _, _, end_next_state, end_done = buffer_list[-1]

                # 向量化計算每個起始位置的 n-step return
                rewards_arr = np.array([t[2] for t in buffer_list], dtype=np.float32)
                returns = np.empty(k, dtype=np.float32)
                returns[k - 1] = rewards_arr[k - 1]
                for i in range(k - 2, -1, -1):
                    returns[i] = rewards_arr[i] + self._gamma_scalar * returns[i + 1]

                for i in range(k):
                    s, a = buffer_list[i][0], buffer_list[i][1]
                    agent_memory.append((s, a, float(returns[i]), end_next_state, end_done, k - i))

                buffer.clear()

            elif len(buffer) == self.n_step:
                # 正常情況：np.dot 取代 Python 迴圈
                rewards_arr = np.array([t[2] for t in buffer], dtype=np.float32)
                n_step_return = float(np.dot(self._gamma_weights, rewards_arr))

                start_state, start_action = buffer[0][0], buffer[0][1]
                _, _, _, end_next_state, end_done = buffer[-1]

                agent_memory.append((start_state, start_action, n_step_return, end_next_state, end_done, self.n_step))

                buffer.popleft()

    def remember_batch(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                       next_obs: np.ndarray, terminated: np.ndarray, robot_id: int):
        """
        向量化版 remember()：一次處理所有 num_envs 個 environment。

        以 numpy circular buffer（ns_*）取代 Python deque，避免 256 次 Python 函式呼叫。
        僅在 use_vec_env=True 且 n_step>1 時完整向量化；其他情況 fallback 到 remember()。

        Args:
            obs:        (E, obs_dim)  當前觀測
            actions:    (E,)          動作
            rewards:    (E,)          即時 reward
            next_obs:   (E, obs_dim)  下一步觀測
            terminated: (E,)          bool，是否 episode 結束
            robot_id:   int           哪個 agent（= agent_idx）
        """
        agent_idx = robot_id
        agent_id  = self.agent_ids[agent_idx]
        agent_memory = self.memories[agent_id]
        E = self.num_envs

        # ── 1-step：直接 append_batch，最快 ──────────────────────────────────────
        if self.n_step == 1:
            agent_memory.append_batch(
                obs, actions, rewards, next_obs,
                terminated.astype(np.float32),
                np.ones(E, dtype=np.float32)
            )
            return

        # ── n-step，non-vec fallback（一般不會走到）──────────────────────────────
        if not self.use_vec_env:
            self.remember(obs[0], actions[0], rewards[0], next_obs[0], terminated[0], 0, robot_id)
            return

        k = self.n_step
        env_range = np.arange(E)

        # ── Step 1：寫入所有 env 的新 transition ────────────────────────────────
        old_ptrs = self.ns_ptr[:, agent_idx].copy()          # (E,)
        self.ns_states     [env_range, agent_idx, old_ptrs] = obs
        self.ns_actions    [env_range, agent_idx, old_ptrs] = actions
        self.ns_rewards    [env_range, agent_idx, old_ptrs] = rewards
        self.ns_next_states[env_range, agent_idx, old_ptrs] = next_obs
        self.ns_dones      [env_range, agent_idx, old_ptrs] = terminated

        self.ns_ptr  [:, agent_idx] = (old_ptrs + 1) % k
        self.ns_count[:, agent_idx] = np.minimum(self.ns_count[:, agent_idx] + 1, k)

        # ── Step 2：Done envs → flush（反向遞推，O(k)）──────────────────────────
        done_envs = np.where(terminated)[0]
        if len(done_envs) > 0:
            for env_idx in done_envs:
                c       = int(self.ns_count[env_idx, agent_idx])
                ptr_now = int(self.ns_ptr  [env_idx, agent_idx])
                oldest  = (ptr_now - c) % k
                ord_idx = [(oldest + i) % k for i in range(c)]

                rewards_arr = self.ns_rewards[env_idx, agent_idx][ord_idx]
                returns     = np.empty(c, dtype=np.float32)
                returns[c - 1] = rewards_arr[c - 1]
                for i in range(c - 2, -1, -1):
                    returns[i] = rewards_arr[i] + self._gamma_scalar * returns[i + 1]

                end_ns   = self.ns_next_states[env_idx, agent_idx, ord_idx[-1]]
                end_done = True
                for i in range(c):
                    s = self.ns_states [env_idx, agent_idx, ord_idx[i]]
                    a = int(self.ns_actions[env_idx, agent_idx, ord_idx[i]])
                    agent_memory.append((s, a, float(returns[i]), end_ns, end_done, c - i))

                # Reset this env's buffer
                self.ns_ptr  [env_idx, agent_idx] = 0
                self.ns_count[env_idx, agent_idx] = 0

        # ── Step 3：Full buffer，non-done envs → 向量化 n-step return ────────────
        full_mask = (self.ns_count[:, agent_idx] >= k) & (~terminated)
        full_envs = np.where(full_mask)[0]
        if len(full_envs) == 0:
            return

        # ns_ptr = oldest position（寫完後 ptr 已指向下一格，即最舊的格子）
        oldest_ptrs  = self.ns_ptr[full_envs, agent_idx]              # (F,)
        newest_ptrs  = (oldest_ptrs - 1 + k) % k                      # (F,)

        # Gather rewards in chronological order: (F, k)
        i_range     = np.arange(k)
        gather_idx  = (oldest_ptrs[:, None] + i_range[None, :]) % k   # (F, k)
        rew_ordered = self.ns_rewards[full_envs[:, None], agent_idx, gather_idx]

        n_step_returns = rew_ordered @ self._gamma_weights             # (F,)
        start_states   = self.ns_states     [full_envs, agent_idx, oldest_ptrs]  # (F, d)
        start_actions  = self.ns_actions    [full_envs, agent_idx, oldest_ptrs]  # (F,)
        end_next_states= self.ns_next_states[full_envs, agent_idx, newest_ptrs]  # (F, d)
        end_dones      = self.ns_dones      [full_envs, agent_idx, newest_ptrs].astype(np.float32)

        agent_memory.append_batch(
            start_states, start_actions, n_step_returns,
            end_next_states, end_dones,
            np.full(len(full_envs), k, dtype=np.float32)
        )

    def sample_batch(self, agent_id: str) -> Tuple:
        """Sample a batch from agent's NumpyReplayBuffer and convert to tensors"""
        agent_memory = self.memories[agent_id]
        if len(agent_memory) == 0:
            return None
        batch_size = min(self.args.batch_size, len(agent_memory))

        # NumpyReplayBuffer.sample() 直接回傳 numpy arrays，無需 list() 轉換
        states_np, actions_np, rewards_np, next_states_np, dones_np, actual_n_steps_np = \
            agent_memory.sample(batch_size)

        states          = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        next_states     = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        actions         = torch.from_numpy(actions_np).to(self.device, non_blocking=True)
        rewards         = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        dones           = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
        actual_n_steps_t = torch.from_numpy(actual_n_steps_np).to(self.device, non_blocking=True)

        return states, actions, rewards, next_states, dones, actual_n_steps_t

    def select_actions_vectorized(self, observations: np.ndarray) -> np.ndarray:
        """
        Vectorized action selection for all environments and agents
        
        Args:
            observations: shape (num_envs, n_agents, obs_dim) or (n_agents, obs_dim)
            
        Returns:
            actions: shape (num_envs, n_agents) or (n_agents,)
        """
        if self.use_vec_env:
            # (num_envs, n_agents, obs_dim)
            num_envs, n_agents, obs_dim = observations.shape
            actions = np.zeros((num_envs, n_agents), dtype=np.int32)
            
            for agent_idx, agent_id in enumerate(self.agent_ids):
                agent = self.agents[agent_id]
                epsilon = agent.epsilon
                
                # 決定哪些環境用 random，哪些用網路
                random_mask = np.random.random(num_envs) < epsilon
                
                # Random actions
                actions[random_mask, agent_idx] = np.random.randint(0, agent.action_dim, size=random_mask.sum())
                
                # Network actions for remaining environments
                if (~random_mask).any():
                    obs_batch = observations[~random_mask, agent_idx]  # (N, obs_dim)
                    obs_tensor = torch.from_numpy(obs_batch).to(self.device)
                    
                    with torch.no_grad():
                        q_values = agent.q_net(obs_tensor)
                        net_actions = q_values.argmax(dim=1).cpu().numpy()
                    
                    actions[~random_mask, agent_idx] = net_actions
            
            return actions
        else:
            # Single environment mode: (n_agents, obs_dim)
            n_agents, obs_dim = observations.shape
            actions = np.zeros(n_agents, dtype=np.int32)
            
            for agent_idx, agent_id in enumerate(self.agent_ids):
                agent = self.agents[agent_id]
                
                if random.random() < agent.epsilon:
                    actions[agent_idx] = random.randint(0, agent.action_dim - 1)
                else:
                    obs_tensor = torch.from_numpy(observations[agent_idx:agent_idx+1]).to(self.device)
                    with torch.no_grad():
                        q_values = agent.q_net(obs_tensor)
                        actions[agent_idx] = q_values.argmax().item()
            
            return actions

    def train(self):
        """Main training loop - Vectorized version"""
        if self.use_vec_env:
            self._train_vectorized()
        else:
            self._train_single()

    def select_actions_for_robot(self, robot_id: int, observations: np.ndarray) -> np.ndarray:
        """
        為指定 robot 在所有環境中選擇動作

        Args:
            robot_id: 機器人 ID (0 到 n_agents-1)
            observations: shape (num_envs, obs_dim) 的觀測 array

        Returns:
            actions: shape (num_envs,) 的動作 array
        """
        agent_id = self.agent_ids[robot_id]
        agent = self.agents[agent_id]
        epsilon = agent.epsilon

        num_envs = observations.shape[0]
        actions = np.zeros(num_envs, dtype=np.int32)

        # 決定哪些環境用 random，哪些用網路
        random_mask = np.random.random(num_envs) < epsilon

        # Random actions
        actions[random_mask] = np.random.randint(0, agent.action_dim, size=random_mask.sum())

        # Network actions for remaining environments
        if (~random_mask).any():
            obs_batch = observations[~random_mask]  # (N, obs_dim)
            obs_tensor = torch.from_numpy(obs_batch).to(self.device)

            with torch.no_grad():
                q_values = agent.q_net(obs_tensor)
                net_actions = q_values.argmax(dim=1).cpu().numpy()

            actions[~random_mask] = net_actions

        return actions

    def _safe_random_actions(self, robot_id: int, obs: np.ndarray) -> np.ndarray:
        """
        Random walk that never hits walls.
        Decodes grid position from obs[0,1] (normalized x,y), then samples
        uniformly from the subset of actions that stay in bounds.
        Action space: 0=UP(y-1) 1=DOWN(y+1) 2=LEFT(x-1) 3=RIGHT(x+1) 4=STAY
        """
        n = self.args.env_n
        x = np.round(obs[:, 0] * (n - 1)).astype(int)
        y = np.round(obs[:, 1] * (n - 1)).astype(int)
        actions = np.empty(self.num_envs, dtype=np.int32)
        for i in range(self.num_envs):
            valid = [4]                         # STAY always ok
            if y[i] > 0:     valid.append(0)    # UP
            if y[i] < n - 1: valid.append(1)    # DOWN
            if x[i] > 0:     valid.append(2)    # LEFT
            if x[i] < n - 1: valid.append(3)    # RIGHT
            actions[i] = np.random.choice(valid)
        return actions

    def _flee_actions(self, robot_id: int, obs: np.ndarray) -> np.ndarray:
        """
        Flee heuristic: move away from the first opponent, wall-aware.

        Obs layout:
          [x, y, energy, wall_up, wall_down, wall_left, wall_right, opp_dx, opp_dy, ...]
          dx > 0 → opponent RIGHT → primary flee LEFT  (action 2)
          dx < 0 → opponent LEFT  → primary flee RIGHT (action 3)
          dy > 0 → opponent BELOW → primary flee UP    (action 0)
          dy < 0 → opponent ABOVE → primary flee DOWN  (action 1)

        If primary flee direction is blocked by wall, fall back to
        the other axis. If both blocked, STAY.
        """
        dx = obs[:, 7]
        dy = obs[:, 8]
        wall_up    = obs[:, 3]  # 1.0 if y==0
        wall_down  = obs[:, 4]  # 1.0 if y==n-1
        wall_left  = obs[:, 5]  # 1.0 if x==0
        wall_right = obs[:, 6]  # 1.0 if x==n-1

        # Wall-check per action: 1 = blocked
        blocked = {
            0: wall_up,    # UP
            1: wall_down,  # DOWN
            2: wall_left,  # LEFT
            3: wall_right, # RIGHT
        }

        # Preferred flee action per axis
        h_pref = np.where(dx > 0, 2, np.where(dx < 0, 3, 4))  # LEFT / RIGHT / STAY
        v_pref = np.where(dy > 0, 0, np.where(dy < 0, 1, 4))  # UP   / DOWN  / STAY

        # Primary axis: whichever opponent distance is larger
        use_h_primary = np.abs(dx) >= np.abs(dy)

        actions = np.full(self.num_envs, 4, dtype=np.int32)
        for i in range(self.num_envs):
            primary = int(h_pref[i]) if use_h_primary[i] else int(v_pref[i])
            fallback = int(v_pref[i]) if use_h_primary[i] else int(h_pref[i])

            if primary != 4 and blocked[primary][i] < 0.5:
                actions[i] = primary
            elif fallback != 4 and blocked[fallback][i] < 0.5:
                actions[i] = fallback
            else:
                actions[i] = 4  # STAY（被牆逼死角）

        return actions

    def _train_vectorized(self):
        """Training loop for vectorized environments (sequential actions)"""
        print(f"Starting vectorized training with {self.num_envs} environments (sequential mode)...")

        # Reset all environments
        observations, _ = self.env.reset()  # (num_envs, n_agents, obs_dim)

        # Reset episode statistics
        self._episode_rewards.fill(0)
        self._episode_steps.fill(0)

        # Track terminations per environment
        env_terminations = np.zeros((self.num_envs, self.n_agents), dtype=bool)

        import signal as _signal
        _interrupted = [False]
        _orig_handler = _signal.getsignal(_signal.SIGINT)
        def _sigint(sig, frame):
            _interrupted[0] = True
        _signal.signal(_signal.SIGINT, _sigint)

        while self.total_episodes < self.episode_offset + self.args.num_episodes and not _interrupted[0]:
            # Sequential actions: each robot acts one at a time (across all envs)
            # robot_speeds[i] determines how many turns per step each robot gets
            for robot_id in range(self.n_agents):
                n_turns = self.robot_speeds[robot_id]
                is_scripted     = robot_id in self.scripted_robots
                is_random       = robot_id in self.random_robots
                is_safe_random  = robot_id in self.safe_random_robots
                is_flee         = robot_id in self.flee_robots
                is_learning = not (is_scripted or is_random or is_safe_random or is_flee)

                for _ in range(n_turns):
                    # --- Action selection (4 modes) ---
                    if is_scripted:
                        actions = np.full(self.num_envs, 4, dtype=np.int32)  # STAY
                    elif is_random:
                        actions = np.random.randint(0, 5, size=self.num_envs).astype(np.int32)
                    elif is_safe_random:
                        obs = self.env.get_observation(robot_id)
                        actions = self._safe_random_actions(robot_id, obs)
                    elif is_flee:
                        obs = self.env.get_observation(robot_id)
                        actions = self._flee_actions(robot_id, obs)
                    else:
                        obs = self.env.get_observation(robot_id)
                        actions = self.select_actions_for_robot(robot_id, obs)

                    # --- Execute ---
                    next_obs, rewards, terminated, truncated, infos = self.env.step_single(robot_id, actions)

                    # --- Bookkeeping (all modes) ---
                    if self.use_vec_env and isinstance(self.env, BatchRobotVacuumEnv):
                        # 全向量化：直接從 BatchRobotVacuumEnv 的 numpy array 讀碰撞數
                        self._episode_rewards[:, robot_id] += rewards
                        self._episode_active_collisions[:, robot_id] += \
                            self.env.active_collisions_with[:, robot_id, :].sum(axis=1)
                        env_terminations[:, robot_id] |= terminated
                        if robot_id == 0:
                            new_deaths = terminated & (self._robot0_death_step == -1)
                            self._robot0_death_step[new_deaths] = \
                                self._episode_steps[new_deaths] + 1
                    else:
                        for env_idx in range(self.num_envs):
                            self._episode_rewards[env_idx, robot_id] += rewards[env_idx]
                            self._episode_active_collisions[env_idx, robot_id] += sum(
                                infos[env_idx].get(f'active_collisions_with_{j}', 0)
                                for j in range(self.n_agents)
                            )
                            if terminated[env_idx]:
                                env_terminations[env_idx, robot_id] = True
                                if robot_id == 0 and self._robot0_death_step[env_idx] == -1:
                                    self._robot0_death_step[env_idx] = self._episode_steps[env_idx] + 1

                    # --- Store transitions (learning mode only) ---
                    if is_learning:
                        if self.use_vec_env:
                            self.remember_batch(obs, actions, rewards, next_obs, terminated, robot_id)
                        else:
                            self.remember(obs[0], actions[0], rewards[0], next_obs[0], terminated[0], 0, robot_id)

            # 5. Advance step count after all robots have acted
            done_mask, done_envs = self.env.advance_step()

            for env_idx in range(self.num_envs):
                self._episode_steps[env_idx] += 1

            # Train (controlled by train_frequency)
            if self.global_step % self.train_frequency == 0:
                # 每個 agent 從自己的 buffer 抽樣訓練
                for agent_id in self.agent_ids:
                    agent_memory = self.memories[agent_id]
                    if len(agent_memory) >= max(self.args.replay_start_size, self.args.batch_size):
                        batch = self.sample_batch(agent_id)

                        if batch is not None:
                            train_stats = self.agents[agent_id].train_step(batch)

                            # Log training stats (reduced frequency)
                            if train_stats and self.global_step % 10000 == 0:
                                wandb.log({
                                    f"{agent_id}/loss": train_stats['loss'],
                                    f"{agent_id}/q_mean": train_stats['q_mean'],
                                    "global_step": self.global_step
                                })

            # Update target networks
            if self.global_step % self.args.target_update_frequency == 0:
                for agent in self.agents.values():
                    agent.update_target_network()

            # Handle completed episodes
            for env_idx in done_envs:
                self.total_episodes += 1

                # Log episode summary
                mean_reward = self._episode_rewards[env_idx].mean()
                steps = self._episode_steps[env_idx]

                if self.total_episodes % 100 == 0:
                    current_epsilon = self.agents['robot_0'].epsilon
                    rewards_str = " | ".join(
                        f"r{i}:{self._episode_rewards[env_idx, i]:.1f}"
                        for i in range(self.n_agents)
                    )
                    collisions_str = ";".join(
                        str(self._episode_active_collisions[env_idx, i])
                        for i in range(self.n_agents)
                    )
                    death_step = self._robot0_death_step[env_idx]
                    death_str = f"r0_died@{death_step}" if death_step != -1 else "r0_alive"
                    print(f"[Episode {self.total_episodes}] Steps:{steps} | "
                          f"{rewards_str} | Collisions({collisions_str}) | {death_str} | ε:{current_epsilon:.3f}")

                    # 計算所有 buffer 的總大小
                    total_buffer_size = sum(len(m) for m in self.memories.values())
                    log_dict = {
                        "episode": self.total_episodes,
                        "episode_length": steps,
                        "mean_episode_reward": mean_reward,
                        "epsilon": current_epsilon,
                        "global_step": self.global_step,
                        "buffer_size": total_buffer_size
                    }
                    for i in range(self.n_agents):
                        log_dict[f"robot_{i}/episode_reward"] = self._episode_rewards[env_idx, i]
                        log_dict[f"robot_{i}/active_collisions"] = int(self._episode_active_collisions[env_idx, i])
                    if death_step != -1:
                        log_dict["robot_0/death_step"] = int(death_step)
                    wandb.log(log_dict)

                # Reset episode statistics for this environment
                self._episode_rewards[env_idx].fill(0)
                self._episode_steps[env_idx] = 0
                self._episode_active_collisions[env_idx].fill(0)
                env_terminations[env_idx, :] = False
                self._robot0_death_step[env_idx] = -1

                # Save models periodically (skip the very first step = offset itself)
                new_ep = self.total_episodes - self.episode_offset
                if new_ep > 0 and new_ep % self.args.save_frequency == 0:
                    self.save_models(f"episode_{self.total_episodes}")

            # Update epsilon based on training progress (relative to this run's episodes)
            if len(done_envs) > 0:
                progress = (self.total_episodes - self.episode_offset) / self.args.num_episodes
                for agent in self.agents.values():
                    agent.set_epsilon_by_progress(progress)

            self.global_step += 1

        _signal.signal(_signal.SIGINT, _orig_handler)  # restore handler

        if _interrupted[0]:
            ckpt = f"episode_{self.total_episodes}_interrupted"
            print("\n\n[Interrupted] Ctrl+C detected — saving current model...")
            self.save_models(ckpt)
            print(f"[Interrupted] Saved to: {self.save_dir}/{ckpt}")
            print(f"[Interrupted] To continue from this checkpoint, add:")
            print(f"  --load-model-dir {self.save_dir}/{ckpt}")
        else:
            self.save_models(f"episode_{self.total_episodes}")
            print(f"Training completed! Total episodes: {self.total_episodes}")

    def _train_single(self):
        """Training loop for single environment (sequential actions)"""
        print("Starting single-environment training (sequential mode)...")

        for episode in range(self.args.num_episodes):
            self.total_episodes = episode

            obs_dict, infos = self.env.reset()

            episode_rewards = np.zeros(self.n_agents)
            step_count = 0
            done = False

            # Track terminations
            terminations = {agent_id: False for agent_id in self.agent_ids}

            while not done:
                # Sequential actions: each robot acts one at a time
                for robot_id in range(self.n_agents):
                    agent_id = self.agent_ids[robot_id]
                    agent = self.agents[agent_id]

                    # 1. Get current observation
                    obs = self.env.get_observation(robot_id)

                    # 2. Select action
                    if random.random() < agent.epsilon:
                        action = random.randint(0, agent.action_dim - 1)
                    else:
                        obs_tensor = torch.from_numpy(obs[np.newaxis, :]).to(self.device)
                        with torch.no_grad():
                            q_values = agent.q_net(obs_tensor)
                            action = q_values.argmax().item()

                    # 3. Execute action
                    next_obs, reward, terminated, truncated, info = self.env.step_single(robot_id, action)

                    # 4. Store transition
                    self.remember(obs, action, reward, next_obs, terminated, env_idx=0, agent_idx=robot_id)
                    episode_rewards[robot_id] += reward

                    # Update termination status
                    if terminated:
                        terminations[agent_id] = True

                # 5. Advance step count
                max_steps_reached, truncations = self.env.advance_step()
                step_count += 1

                # Train - 每個 agent 從自己的 buffer 抽樣訓練
                if self.global_step % self.train_frequency == 0:
                    for agent_id in self.agent_ids:
                        agent_memory = self.memories[agent_id]
                        if len(agent_memory) >= self.args.replay_start_size:
                            batch = self.sample_batch(agent_id)
                            if batch is not None:
                                self.agents[agent_id].train_step(batch)

                # Update target networks
                if self.global_step % self.args.target_update_frequency == 0:
                    for agent in self.agents.values():
                        agent.update_target_network()

                self.global_step += 1

                # Check termination
                alive_count = sum(1 for agent_id in self.agent_ids if not terminations[agent_id])

                if alive_count == 0 or max_steps_reached:
                    done = True

            # Episode summary
            mean_reward = episode_rewards.mean()

            if (episode + 1) % 100 == 0:
                current_epsilon = self.agents['robot_0'].epsilon
                print(f"[Episode {episode + 1}] Steps: {step_count} | "
                      f"Mean Reward: {mean_reward:.2f} | Epsilon: {current_epsilon:.3f}")

                wandb.log({
                    "episode": episode + 1,
                    "episode_length": step_count,
                    "mean_episode_reward": mean_reward,
                    "epsilon": current_epsilon,
                    "global_step": self.global_step
                })

            # Update epsilon based on training progress
            progress = (episode + 1) / self.args.num_episodes
            for agent in self.agents.values():
                agent.set_epsilon_by_progress(progress)

            # Save models
            if (episode + 1) % self.args.save_frequency == 0:
                self.save_models(f"episode_{episode + 1}")

    def save_models(self, subfolder: str):
        """Save all agents' models"""
        save_path = os.path.join(self.save_dir, subfolder)
        os.makedirs(save_path, exist_ok=True)

        for agent_id, agent in self.agents.items():
            model_path = os.path.join(save_path, f"{agent_id}.pt")
            agent.save(model_path)

        print(f"Models saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Vectorized DQN Training for Multi-Robot Environment")

    # Environment parameters
    parser.add_argument("--env-n", type=int, default=3, help="Environment grid size (n×n)")
    parser.add_argument("--num-robots", type=int, default=4, help="Number of robots (1-4)")
    parser.add_argument("--initial-energy", type=int, default=100, help="Initial energy for all robots")
    parser.add_argument("--robot-0-energy", type=int, default=1000, help="Initial energy for robot 0")
    parser.add_argument("--robot-1-energy", type=int, default=100, help="Initial energy for robot 1")
    parser.add_argument("--robot-2-energy", type=int, default=100, help="Initial energy for robot 2")
    parser.add_argument("--robot-3-energy", type=int, default=100, help="Initial energy for robot 3")
    parser.add_argument("--robot-0-speed", type=int, default=1, help="Move speed (teleport N cells) for robot 0")
    parser.add_argument("--robot-1-speed", type=int, default=1, help="Move speed (teleport N cells) for robot 1")
    parser.add_argument("--robot-2-speed", type=int, default=1, help="Move speed (teleport N cells) for robot 2")
    parser.add_argument("--robot-3-speed", type=int, default=1, help="Move speed (teleport N cells) for robot 3")
    parser.add_argument("--e-move", type=int, default=1, help="Energy cost per move")
    parser.add_argument("--e-charge", type=float, default=1.5, help="Energy gain per charge")
    parser.add_argument("--e-collision", type=int, default=3, help="Energy loss per collision")
    parser.add_argument("--e-boundary", type=int, default=50, help="Energy loss when hitting wall")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--charger-positions", type=str, default=None,
                       help='Charger positions as "y1,x1;y2,x2;..."')
    parser.add_argument("--dust-max", type=float, default=10.0, help="Max dust per normal cell")
    parser.add_argument("--dust-rate", type=float, default=0.5, help="Dust sigmoid growth rate")
    parser.add_argument("--dust-epsilon", type=float, default=0.5, help="Dust growth seed value")
    parser.add_argument("--charger-dust-max-ratio", type=float, default=0.3, help="Charger cell max dust ratio vs normal")
    parser.add_argument("--charger-dust-rate-ratio", type=float, default=0.5, help="Charger cell growth rate ratio vs normal")
    parser.add_argument("--dust-reward-scale", type=float, default=0.05, help="Dust reward multiplier")
    parser.add_argument("--no-dust", action="store_true", default=False, help="Disable dust system (removes n² obs dimensions)")
    parser.add_argument("--exclusive-charging", action="store_true", default=False,
                        help="充電座獨佔模式：有其他 robot 在 3x3 範圍內時充電無效")
    parser.add_argument("--scripted-robots", type=str, default="",
                        help='Comma-separated robot IDs to fix as STAY (no training), e.g. "1" or "1,2"')
    parser.add_argument("--random-robots", type=str, default="",
                        help='Comma-separated robot IDs that use random actions (no training), e.g. "1"')
    parser.add_argument("--safe-random-robots", type=str, default="",
                        help='Comma-separated robot IDs that use wall-avoiding random walk (no training), e.g. "1"')
    parser.add_argument("--flee-robots", type=str, default="",
                        help='Comma-separated robot IDs that use flee heuristic (no training), e.g. "1"')
    parser.add_argument("--random-start-robots", type=str, default="",
                        help='Comma-separated robot IDs to randomize start position each episode, e.g. "0"')
    parser.add_argument("--load-model-dir", type=str, default=None,
                        help='Load pre-trained models from this dir to continue training (curriculum phase)')

    # DQN hyperparameters
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n-step", type=int, default=1, help="N-step return")

    # Epsilon configuration
    parser.add_argument("--use-epsilon-decay", action=argparse.BooleanOptionalAction, default=True, help="Use epsilon decay (use --no-use-epsilon-decay to disable)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Fixed epsilon")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon for decay")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")

    parser.add_argument("--memory-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--replay-start-size", type=int, default=1000, help="Minimum buffer size before training")
    parser.add_argument("--target-update-frequency", type=int, default=1000, help="Target network update frequency")

    # Vectorized training parameters
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Subprocess workers for env parallelism (1=disabled, N>1=SubprocVecEnv). "
                             "num_envs must be divisible by num_workers. "
                             "建議: --num-envs 32 --num-workers 8 (8 cores, 4 envs each)")
    parser.add_argument("--train-frequency", type=int, default=4, help="Train every N steps")
    parser.add_argument("--batch-env", action="store_true", default=False,
                        help="使用全 numpy 批次環境（BatchRobotVacuumEnv），大幅加速 env_step")

    # Training settings
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--save-frequency", type=int, default=1000, help="Model save frequency (episodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-torch-compile", action="store_true", help="Enable torch.compile")

    # Wandb and logging
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--wandb-project", type=str, default="multi-robot-idqn", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default="idqn-vec", help="Wandb run name")
    parser.add_argument("--wandb-mode", type=str, default="offline", help="Wandb mode")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save models")

    # Auto-evaluation after training
    parser.add_argument("--eval-after-training", action=argparse.BooleanOptionalAction, default=True, help="Run evaluation after training (use --no-eval-after-training to disable)")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Max steps for evaluation")

    args = parser.parse_args()

    # Auto-generate run name if not specified (or using default)
    if args.wandb_run_name == "idqn-vec":
        auto_name = f"nstep{args.n_step}_episode{args.num_episodes}"
        args.wandb_run_name = auto_name
        print(f"Auto-generated run name: {auto_name}")

    # Auto-set save-dir based on run name if using default
    if args.save_dir == "./models":
        args.save_dir = f"./models/{args.wandb_run_name}"
        print(f"Auto-generated save dir: {args.save_dir}")

    # Initialize wandb
    wandb_config = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": vars(args),
        "save_code": True,
        "mode": args.wandb_mode
    }
    if args.wandb_entity:
        wandb_config["entity"] = args.wandb_entity

    wandb.init(**wandb_config)

    # Train
    trainer = VectorizedMultiAgentTrainer(args)
    trainer.train()

    wandb.finish()

    # Auto-evaluation after training
    if args.eval_after_training:
        import subprocess

        # Find the latest model directory
        final_episode = args.num_episodes
        model_dir = os.path.join(trainer.save_dir, f"episode_{final_episode}")

        if not os.path.exists(model_dir):
            # Find the highest episode folder
            episode_dirs = [d for d in os.listdir(trainer.save_dir) if d.startswith("episode_")]
            if episode_dirs:
                episode_nums = [int(d.split("_")[1]) for d in episode_dirs]
                final_episode = max(episode_nums)
                model_dir = os.path.join(trainer.save_dir, f"episode_{final_episode}")

        print(f"\n{'='*50}")
        print(f"Running evaluation on {model_dir}")
        print(f"{'='*50}\n")

        # Build evaluation command
        eval_cmd = [
            "python", "evaluate_models.py",
            "--model-dir", model_dir,
            "--num-robots", str(args.num_robots),
            "--env-n", str(args.env_n),
            "--e-move", str(args.e_move),
            "--e-charge", str(args.e_charge),
            "--e-collision", str(args.e_collision),
            "--e-boundary", str(args.e_boundary),
            "--max-steps", str(args.eval_steps),
            "--eval-epsilon", "0",
            "--gamma", str(args.gamma),
        ]

        # Add individual robot energies (only for robots that exist)
        if args.num_robots >= 1:
            eval_cmd.extend(["--robot-0-energy", str(args.robot_0_energy)])
            eval_cmd.extend(["--robot-0-speed", str(args.robot_0_speed)])
        if args.num_robots >= 2:
            eval_cmd.extend(["--robot-1-energy", str(args.robot_1_energy)])
            eval_cmd.extend(["--robot-1-speed", str(args.robot_1_speed)])
        if args.num_robots >= 3:
            eval_cmd.extend(["--robot-2-energy", str(args.robot_2_energy)])
            eval_cmd.extend(["--robot-2-speed", str(args.robot_2_speed)])
        if args.num_robots >= 4:
            eval_cmd.extend(["--robot-3-energy", str(args.robot_3_energy)])
            eval_cmd.extend(["--robot-3-speed", str(args.robot_3_speed)])

        # Add charger positions if specified
        if args.charger_positions:
            eval_cmd.extend(["--charger-positions", args.charger_positions])

        # Add boolean flags
        if args.no_dust:
            eval_cmd.append("--no-dust")
        if args.exclusive_charging:
            eval_cmd.append("--exclusive-charging")
        if args.scripted_robots:
            eval_cmd.extend(["--scripted-robots", args.scripted_robots])
        if args.random_robots:
            eval_cmd.extend(["--random-robots", args.random_robots])
        if args.safe_random_robots:
            eval_cmd.extend(["--safe-random-robots", args.safe_random_robots])
        if args.flee_robots:
            eval_cmd.extend(["--flee-robots", args.flee_robots])

        # Add wandb settings
        if args.wandb_entity:
            eval_cmd.extend(["--wandb-entity", args.wandb_entity])
        eval_cmd.extend(["--wandb-project", "robot-vacuum-eval"])
        eval_cmd.extend(["--wandb-run-name", f"{args.wandb_run_name}-eval"])
        eval_cmd.extend(["--wandb-mode", args.wandb_mode])

        # Run evaluation
        subprocess.run(eval_cmd)


if __name__ == "__main__":
    main()
