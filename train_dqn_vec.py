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
from dqn import DQN, init_weights, build_network, NoisyLinear, C51DQN

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

        # Epsilon configuration (per-robot overrides global args)
        self.use_epsilon_decay = args.use_epsilon_decay
        try:
            robot_idx = int(agent_id.split('_')[1])
        except (IndexError, ValueError):
            robot_idx = None
        if robot_idx is not None:
            _es = getattr(args, f'robot_{robot_idx}_epsilon_start', None)
            _ee = getattr(args, f'robot_{robot_idx}_epsilon_end', None)
            _sc = getattr(args, f'robot_{robot_idx}_epsilon_schedule', None)
            eps_start = _es if _es is not None else args.epsilon_start
            eps_end   = _ee if _ee is not None else args.epsilon_end
            schedule  = _sc if _sc is not None else getattr(args, 'epsilon_schedule', 'linear')
        else:
            eps_start = args.epsilon_start
            eps_end   = args.epsilon_end
            schedule  = getattr(args, 'epsilon_schedule', 'linear')
        self.epsilon_schedule = schedule
        if self.use_epsilon_decay:
            self.epsilon       = eps_start
            self.epsilon_start = eps_start
            self.epsilon_end   = eps_end
            self.epsilon_decay = args.epsilon_decay
        else:
            self.epsilon = args.epsilon

        # Shape params for exp_tail / sigmoid schedules (per-robot overrides global)
        if robot_idx is not None:
            self.exp_tail_k = getattr(args, f'robot_{robot_idx}_exp_tail_k', None) or getattr(args, 'exp_tail_k', 15.0)
            self.sigmoid_s  = getattr(args, f'robot_{robot_idx}_sigmoid_s',  None) or getattr(args, 'sigmoid_s',  14.0)
            self.sigmoid_c  = getattr(args, f'robot_{robot_idx}_sigmoid_c',  None) or getattr(args, 'sigmoid_c',  0.45)
        else:
            self.exp_tail_k = getattr(args, 'exp_tail_k', 15.0)
            self.sigmoid_s  = getattr(args, 'sigmoid_s',  14.0)
            self.sigmoid_c  = getattr(args, 'sigmoid_c',  0.45)

        # Eval epsilon
        self.eval_epsilon = getattr(args, 'eval_epsilon', 0.0)

        # Rainbow flags
        self.use_noisy = getattr(args, 'noisy', False)
        self.use_c51 = getattr(args, 'c51', False)
        self.use_dueling = getattr(args, 'dueling', False)
        self.num_atoms = getattr(args, 'num_atoms', 51)
        self.v_min = getattr(args, 'v_min', -100.0)
        self.v_max = getattr(args, 'v_max', 100.0)

        # Network architecture
        self.q_net = build_network(
            action_dim, observation_dim,
            dueling=self.use_dueling, noisy=self.use_noisy,
            c51=self.use_c51, num_atoms=self.num_atoms,
            v_min=self.v_min, v_max=self.v_max,
        ).to(device)
        self.target_net = build_network(
            action_dim, observation_dim,
            dueling=self.use_dueling, noisy=self.use_noisy,
            c51=self.use_c51, num_atoms=self.num_atoms,
            v_min=self.v_min, v_max=self.v_max,
        ).to(device)
        if not self.use_noisy:
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

    def train_step(self, batch: Tuple, weights: torch.Tensor = None) -> Dict[str, float]:
        """
        Execute one training step using provided batch.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones, actual_n_steps) as tensors
            weights: Optional IS-correction weights for PER (shape: batch_size,)

        Returns:
            Training statistics. When weights are provided, also includes 'td_errors' (numpy array).
        """
        states, actions, rewards, next_states, dones, actual_n_steps = batch
        self.train_count += 1

        if self.use_c51:
            loss, td_errors, q_values = self._train_step_c51(
                states, actions, rewards, next_states, dones, actual_n_steps, weights)
        else:
            loss, td_errors, q_values = self._train_step_dqn(
                states, actions, rewards, next_states, dones, actual_n_steps, weights)

        # Reset noise after each training step (NoisyNet)
        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        stats = {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_std': q_values.std().item(),
        }
        if weights is not None:
            stats['td_errors'] = td_errors.detach().float().cpu().numpy()
        return stats

    def _train_step_dqn(self, states, actions, rewards, next_states, dones, actual_n_steps, weights):
        """Standard (Double) DQN loss."""
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    gamma_n = self.gamma ** actual_n_steps
                    next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                    target_q_values = rewards + gamma_n * next_q_values * (1 - dones)
                td_errors = q_values - target_q_values
                if weights is not None:
                    loss = (weights * td_errors.pow(2)).mean()
                else:
                    loss = td_errors.pow(2).mean()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                gamma_n = self.gamma ** actual_n_steps
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                target_q_values = rewards + gamma_n * next_q_values * (1 - dones)
            td_errors = q_values - target_q_values
            if weights is not None:
                loss = (weights * td_errors.pow(2)).mean()
            else:
                loss = td_errors.pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, td_errors, q_values

    def _train_step_c51(self, states, actions, rewards, next_states, dones, actual_n_steps, weights):
        """Categorical (C51) distributional loss with Double-DQN action selection."""
        import torch.nn.functional as _F
        net = self.q_net
        tgt = self.target_net
        B = states.size(0)
        N = self.num_atoms
        v_min, v_max = self.v_min, self.v_max
        delta_z = (v_max - v_min) / (N - 1)
        atoms = net.atoms  # (N,)

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                # Current distribution
                log_probs = net.dist(states)  # (B, A, N)
                log_probs_a = log_probs[torch.arange(B, device=states.device), actions]  # (B, N)

                with torch.no_grad():
                    # Double DQN: use online net to select action
                    next_actions = net(next_states).argmax(1)  # (B,)
                    # Target distribution for selected action
                    target_log_probs = tgt.dist(next_states)
                    target_probs = target_log_probs.exp()
                    target_probs_a = target_probs[torch.arange(B, device=states.device), next_actions]  # (B, N)

                    # Project target distribution
                    gamma_n = self.gamma ** actual_n_steps  # (B,)
                    Tz = rewards.unsqueeze(1) + gamma_n.unsqueeze(1) * (1 - dones.unsqueeze(1)) * atoms.unsqueeze(0)
                    Tz = Tz.clamp(v_min, v_max)
                    b = (Tz - v_min) / delta_z  # (B, N)
                    l = b.floor().long()
                    u = b.ceil().long()
                    # Fix edge case: l == u
                    l = l.clamp(0, N - 1)
                    u = u.clamp(0, N - 1)

                    proj = torch.zeros_like(target_probs_a)
                    proj.scatter_add_(1, l, target_probs_a * (u.float() - b))
                    proj.scatter_add_(1, u, target_probs_a * (b - l.float()))

                # Cross-entropy loss
                elem_loss = -(_F.log_softmax(log_probs[torch.arange(B, device=states.device), actions], dim=1) * proj).sum(dim=1)

                if weights is not None:
                    loss = (weights * elem_loss).mean()
                else:
                    loss = elem_loss.mean()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Current distribution
            log_probs = net.dist(states)
            log_probs_a = log_probs[torch.arange(B, device=states.device), actions]

            with torch.no_grad():
                next_actions = net(next_states).argmax(1)
                target_log_probs = tgt.dist(next_states)
                target_probs = target_log_probs.exp()
                target_probs_a = target_probs[torch.arange(B, device=states.device), next_actions]

                gamma_n = self.gamma ** actual_n_steps
                Tz = rewards.unsqueeze(1) + gamma_n.unsqueeze(1) * (1 - dones.unsqueeze(1)) * atoms.unsqueeze(0)
                Tz = Tz.clamp(v_min, v_max)
                b = (Tz - v_min) / delta_z
                l = b.floor().long().clamp(0, N - 1)
                u = b.ceil().long().clamp(0, N - 1)

                proj = torch.zeros_like(target_probs_a)
                proj.scatter_add_(1, l, target_probs_a * (u.float() - b))
                proj.scatter_add_(1, u, target_probs_a * (b - l.float()))

            elem_loss = -(log_probs_a * proj).sum(dim=1)

            if weights is not None:
                loss = (weights * elem_loss).mean()
            else:
                loss = elem_loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # TD errors for PER (use element-wise loss as proxy)
        td_errors = elem_loss.detach()
        # Q-values for logging
        with torch.no_grad():
            q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        return loss, td_errors, q_values

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_epsilon_by_progress(self, progress: float):
        """Set epsilon based on training progress (0.0 to 1.0).

        Schedules:
          linear      : ε = start + (end-start) * progress
          exponential : ε = start * (end/start)^progress
          exp_tail    : ε = end + (start-end) * exp(-k*progress)
                        uses epsilon_decay as k (default 15)
          sigmoid     : ε = end + (start-end) / (1 + exp(s*(progress-c)))
                        s = epsilon_decay (default 14), c = 0.45
        """
        if self.use_epsilon_decay:
            x = progress
            if self.epsilon_schedule == 'exponential':
                self.epsilon = self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** x
            elif self.epsilon_schedule == 'exp_tail':
                k = getattr(self, 'exp_tail_k', 15.0)
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-k * x)
            elif self.epsilon_schedule == 'sigmoid':
                s = getattr(self, 'sigmoid_s', 14.0)
                c = getattr(self, 'sigmoid_c', 0.45)
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) / (1.0 + np.exp(s * (x - c)))
            else:
                # Linear
                self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * x

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


class SumTree:
    """
    Binary sum tree for O(log N) priority sampling.
    Leaves store per-sample priorities; each internal node = sum of children.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def update(self, idx: int, priority: float):
        """Update leaf at data index idx and propagate upward."""
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def sample(self, value: float) -> int:
        """Walk tree downward to find leaf for given cumulative value."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = 2 * idx + 2
        return idx - (self.capacity - 1)

    def sample_batch(self, values: np.ndarray) -> np.ndarray:
        """Vectorized batch tree walk. values shape: (n,). Returns data indices (n,)."""
        n = len(values)
        idx = np.zeros(n, dtype=np.int64)  # start at root
        depth = int(np.ceil(np.log2(self.capacity + 1)))
        for _ in range(depth):
            left = 2 * idx + 1
            right = left + 1
            # Clamp to valid tree range
            left = np.minimum(left, len(self.tree) - 1)
            right = np.minimum(right, len(self.tree) - 1)
            left_val = self.tree[left]
            go_right = values > left_val
            values = np.where(go_right, values - left_val, values)
            idx = np.where(go_right, right, left)
        return idx - (self.capacity - 1)

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Schaul et al. 2015).
    Same append interface as NumpyReplayBuffer; sample() additionally returns
    indices and IS-correction weights.
    """
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, recency_lambda: float = 0.0):
        self.capacity = capacity
        self.alpha = alpha
        self.recency_lambda = recency_lambda
        self.tree = SumTree(capacity)

        self._states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)
        self._n_steps     = np.ones(capacity,  dtype=np.float32)
        self._timestamps  = np.zeros(capacity, dtype=np.int64)
        self._write_count = 0
        self._ptr  = 0
        self._size = 0
        self._max_priority = 1.0

    def _set(self, i: int, state, action, reward, next_state, done, n_step):
        self._states[i]      = state
        self._next_states[i] = next_state
        self._actions[i]     = action
        self._rewards[i]     = reward
        self._dones[i]       = float(done)
        self._n_steps[i]     = n_step
        self._timestamps[i]  = self._write_count
        self._write_count   += 1
        self.tree.update(i, self._max_priority ** self.alpha)

    def append(self, transition):
        state, action, reward, next_state, done, n_step = transition
        self._set(self._ptr, state, action, reward, next_state, done, n_step)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def append_batch(self, states, actions, rewards, next_states, dones, n_steps):
        n = len(states)
        if n == 0:
            return
        priority = self._max_priority ** self.alpha
        for j in range(n):
            i = (self._ptr + j) % self.capacity
            self._states[i]      = states[j]
            self._next_states[i] = next_states[j]
            self._actions[i]     = actions[j]
            self._rewards[i]     = rewards[j]
            self._dones[i]       = dones[j]
            self._n_steps[i]     = n_steps[j]
            self._timestamps[i]  = self._write_count + j
            self.tree.update(i, priority)
        self._write_count += n
        self._ptr  = int((self._ptr + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Stratified sampling with optional recency weighting.
        When recency_lambda > 0, draws 2x candidates from the tree then
        resamples using recency weights, so the tree structure is unchanged.
        Returns (states, actions, rewards, next_states, dones, n_steps, indices, weights).
        """
        total = self.tree.total
        use_recency = self.recency_lambda > 0.0

        # Draw candidates (2x if recency resampling, else exactly batch_size)
        n_candidates = batch_size * 2 if use_recency else batch_size
        segment = total / n_candidates

        # Vectorized stratified sampling: generate all values at once, walk tree in batch
        lo = segment * np.arange(n_candidates)
        hi = lo + segment
        values = np.random.uniform(lo, hi)
        cand_indices = self.tree.sample_batch(values)
        cand_indices = np.clip(cand_indices, 0, self._size - 1)
        cand_priorities = self.tree.tree[cand_indices + self.capacity - 1]

        if use_recency:
            # Recency weight: exp(-lambda * age / capacity), age = write_count - timestamp
            ages = self._write_count - self._timestamps[cand_indices]
            recency_w = np.exp(-self.recency_lambda * ages / max(self.capacity, 1))
            combined = cand_priorities * recency_w
            combined = combined / combined.sum()
            chosen = np.random.choice(n_candidates, size=batch_size, replace=False, p=combined)
            indices = cand_indices[chosen]
            priorities = cand_priorities[chosen]
            # Deduplicate with numpy: replace duplicates with unused candidates
            _, first_occ = np.unique(indices, return_index=True)
            if len(first_occ) < batch_size:
                used = set(indices.tolist())
                unused = [c for c in range(n_candidates) if cand_indices[c] not in used]
                dup_pos = [i for i in range(batch_size) if i not in first_occ]
                for pos, rep in zip(dup_pos, unused):
                    indices[pos] = cand_indices[rep]
                    priorities[pos] = cand_priorities[rep]
                    used.add(indices[pos])
        else:
            indices = cand_indices
            priorities = cand_priorities

        # IS weights: w_i = (N * P(i))^{-beta}, normalised by max weight
        probs = priorities / total
        weights = (self._size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            self._n_steps[indices],
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-6):
        priorities = (np.abs(td_errors) + epsilon) ** self.alpha
        for idx, p in zip(indices, priorities):
            self.tree.update(int(idx), float(p))
        self._max_priority = max(self._max_priority, float(priorities.max()))

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
        if args.gpu is not None:
            self.device = torch.device(f"cuda:{args.gpu}")
        elif torch.cuda.is_available():
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

        # Frozen robots: use loaded model (greedy, eval_epsilon) but do NOT update weights
        frozen_str = getattr(args, 'frozen_robots', '')
        self.frozen_robots = set(int(x) for x in frozen_str.split(',') if x.strip()) if frozen_str else set()

        # Auto-freeze robots: automatically frozen once their epsilon reaches epsilon_end
        auto_freeze_str = getattr(args, 'auto_freeze_robots', '')
        self.auto_freeze_robots = set(int(x) for x in auto_freeze_str.split(',') if x.strip()) if auto_freeze_str else set()
        self._auto_freeze_logged = set()  # track which robots have been logged as frozen

        # Seek-charger robots: walk toward charger and STAY, not trained
        seek_str = getattr(args, 'seek_charger_robots', '')
        self.seek_charger_robots = set(int(x) for x in seek_str.split(',') if x.strip()) if seek_str else set()
        # Opportunistic charger: seek charger when safe, flee when opponent near
        opp_str = getattr(args, 'opportunistic_charger_robots', '')
        self.opportunistic_charger_robots = set(int(x) for x in opp_str.split(',') if x.strip()) if opp_str else set()
        self._agent_types_mode = getattr(args, 'agent_types_mode', 'off')

        for label, s in [('STAY', self.scripted_robots), ('RANDOM', self.random_robots),
                         ('SAFE_RANDOM', self.safe_random_robots), ('FLEE', self.flee_robots),
                         ('FROZEN', self.frozen_robots), ('SEEK_CHARGER', self.seek_charger_robots),
                         ('OPP_CHARGER', self.opportunistic_charger_robots)]:
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

        # Parse robot start positions (optional)
        robot_start_positions = {}
        rsp_str = getattr(args, 'robot_start_positions', None)
        if rsp_str:
            for i, pos_str in enumerate(rsp_str.split(';')):
                y, x = map(int, pos_str.split(','))
                robot_start_positions[i] = (y, x)
            print(f"Using custom robot start positions: { {i: pos for i, pos in robot_start_positions.items()} }")

        # Build per-robot attack powers
        all_attack_powers = [
            args.robot_0_attack_power,
            args.robot_1_attack_power,
            args.robot_2_attack_power,
            args.robot_3_attack_power,
        ]
        if any(p is not None for p in all_attack_powers[:self.num_robots]):
            robot_attack_powers = [
                p if p is not None else args.e_collision
                for p in all_attack_powers[:self.num_robots]
            ]
            print(f"Per-robot attack powers: {robot_attack_powers}")
        else:
            robot_attack_powers = None

        # Build per-robot docking steps
        all_docking = [
            getattr(args, 'robot_0_docking_steps', None),
            getattr(args, 'robot_1_docking_steps', None),
            getattr(args, 'robot_2_docking_steps', None),
            getattr(args, 'robot_3_docking_steps', None),
        ]
        if any(d is not None for d in all_docking[:self.num_robots]):
            robot_docking_steps = [
                d if d is not None else args.docking_steps
                for d in all_docking[:self.num_robots]
            ]
            print(f"Per-robot docking steps: {robot_docking_steps}")
        else:
            robot_docking_steps = None

        # Build per-robot stun steps
        all_stun = [
            getattr(args, 'robot_0_stun_steps', None),
            getattr(args, 'robot_1_stun_steps', None),
            getattr(args, 'robot_2_stun_steps', None),
            getattr(args, 'robot_3_stun_steps', None),
        ]
        if any(s is not None for s in all_stun[:self.num_robots]):
            robot_stun_steps = [
                s if s is not None else args.stun_steps
                for s in all_stun[:self.num_robots]
            ]
            print(f"Per-robot stun steps: {robot_stun_steps}")
        else:
            robot_stun_steps = None

        # Parse alliance groups
        _alliance_groups = parse_alliance_groups(
            getattr(args, 'alliance_groups', None) or '',
            self.num_robots,
        )
        self._alliance_groups = _alliance_groups
        _energy_sharing_events = [
            e.strip() for e in getattr(args, 'energy_sharing_events', 'charge,collision').split(',')
            if e.strip()
        ]

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
            'charger_range': args.charger_range,
            'random_start_robots': (
                set(range(args.num_robots))
                if getattr(args, 'random_start_robots', 'all') == 'all'
                else set(int(x) for x in args.random_start_robots.split(',') if x.strip())
            ),
            'robot_start_positions': robot_start_positions,
            'agent_types_mode': args.agent_types_mode,
            'triangle_agent_id': args.triangle_agent_id,
            'heterotype_charge_mode': args.heterotype_charge_mode,
            'heterotype_charge_factor': args.heterotype_charge_factor,
            'energy_cap': args.energy_cap,
            'e_decay': args.e_decay,
            'robot_attack_powers': robot_attack_powers,
            'thief_spawn': args.thief_spawn,
            'reward_mode': args.reward_mode,
            'reward_alpha': args.reward_alpha,
            'docking_steps': args.docking_steps,
            'robot_docking_steps': robot_docking_steps,
            'stun_steps': args.stun_steps,
            'robot_stun_steps': robot_stun_steps,
            # Alliance energy-sharing (optional, defaults to disabled)
            'alliance_groups': _alliance_groups,
            'alliance_zone': getattr(args, 'alliance_zone', False),
            'energy_sharing_mode': getattr(args, 'energy_sharing_mode', 'none'),
            'energy_sharing_events': _energy_sharing_events,
            'energy_sharing_self_weight': getattr(args, 'energy_sharing_self_weight', 2.0 / 3.0),
            'energy_sharing_ally_weight': getattr(args, 'energy_sharing_ally_weight', 1.0 / 3.0),
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

        # Print Rainbow DQN features
        rainbow_features = []
        if getattr(args, 'dueling', False): rainbow_features.append('Dueling')
        if getattr(args, 'noisy', False):   rainbow_features.append('NoisyNet')
        if getattr(args, 'c51', False):     rainbow_features.append(f'C51(atoms={args.num_atoms}, v=[{args.v_min},{args.v_max}])')
        if rainbow_features:
            print(f"Rainbow DQN features: {', '.join(rainbow_features)}")

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

        # Independent Replay Buffer (每個 agent 獨立)
        self.use_per = getattr(args, 'per', False)
        if self.use_per:
            per_alpha = getattr(args, 'per_alpha', 0.6)
            self.memories = {
                agent_id: PrioritizedReplayBuffer(args.memory_size, observation_dim, alpha=per_alpha,
                                                   recency_lambda=getattr(args, 'recency_lambda', 0.0))
                for agent_id in self.agent_ids
            }
            print(f"Using Prioritized Experience Replay (alpha={per_alpha})")
        else:
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
        self._write_training_config()

        # Best model tracking
        self.best_metric = float('-inf')

        # Pre-allocated tensors for batch inference
        if self.use_vec_env:
            self._obs_tensor = torch.zeros(self.num_envs, self.n_agents, observation_dim,
                                          dtype=torch.float32, device=self.device)
        else:
            self._obs_tensor = torch.zeros(self.n_agents, observation_dim,
                                          dtype=torch.float32, device=self.device)

        # Per-robot persistent CUDA buffers for select_actions_for_robot
        # Avoids CUDA allocator overhead (~1-5μs per call) across 2000+ inference calls per 500 steps
        _use_cuda = str(self.device).startswith('cuda')
        self._infer_obs_buf: dict = {
            i: torch.empty(self.num_envs, observation_dim, device=self.device, dtype=torch.float32)
            for i in range(self.n_agents)
        } if _use_cuda else {}
        self._infer_mask_buf: torch.Tensor | None = (
            torch.empty(self.num_envs, 5, device=self.device, dtype=torch.bool)
            if _use_cuda else None
        )

        # Pre-allocated pinned CPU buffers for training batch GPU transfer (DMA path)
        # torch.from_numpy + .to(device) uses pageable memory → slow pinning per call.
        # With pre-allocated pinned buffers, transfer goes through DMA (faster).
        bs = args.batch_size
        if _use_cuda:
            self._pin_states  = torch.zeros(bs, observation_dim, pin_memory=True, dtype=torch.float32)
            self._pin_nstates = torch.zeros(bs, observation_dim, pin_memory=True, dtype=torch.float32)
            self._pin_rewards = torch.zeros(bs, pin_memory=True, dtype=torch.float32)
            self._pin_actions = torch.zeros(bs, pin_memory=True, dtype=torch.int64)
            self._pin_dones   = torch.zeros(bs, pin_memory=True, dtype=torch.float32)
            self._pin_nsteps  = torch.zeros(bs, pin_memory=True, dtype=torch.int32)
            self._pin_weights = torch.zeros(bs, pin_memory=True, dtype=torch.float32)
        else:
            self._pin_states = None

        # Episode statistics tracking (for logging)
        _n = self.num_envs if self.use_vec_env else 1
        self._episode_rewards = np.zeros((_n, self.n_agents))
        self._episode_steps = np.zeros(_n, dtype=np.int32)
        self._episode_immediate_kills = np.zeros(_n, dtype=np.int32)
        self._episode_active_collisions = np.zeros((_n, self.n_agents), dtype=np.int32)
        self._robot_death_step = np.full((_n, self.n_agents), -1, dtype=np.int32)  # -1 = alive
        self._monopoly_step = np.full(_n, -1, dtype=np.int32)   # step when all weak robots dead
        self._episode_collisions_matrix = np.zeros((_n, self.n_agents, self.n_agents), dtype=np.int32)  # [env, attacker, victim]
        # 累計兩者 Manhattan distance（僅 2-robot 有意義）
        self._episode_dist_sum = np.zeros(_n, dtype=np.float32)
        self._episode_dist_steps = np.zeros(_n, dtype=np.int32)
        # Per-robot charger occupancy and distance-to-charger tracking
        self._episode_charger_steps = np.zeros((_n, self.n_agents), dtype=np.int32)
        self._episode_dist_to_charger_sum = np.zeros((_n, self.n_agents), dtype=np.float32)
        self._episode_dist_to_charger_count = np.zeros((_n, self.n_agents), dtype=np.int32)
        # First collision step
        self._episode_first_collision_step = np.full(_n, -1, dtype=np.int32)
        # Pursuit metrics: off-charger hits (主動追殺指標)
        self._episode_off_charger_hits = np.zeros((_n, self.n_agents), dtype=np.int32)  # 不在充電座時的碰撞
        self._episode_on_charger_hits = np.zeros((_n, self.n_agents), dtype=np.int32)   # 在充電座時的碰撞
        self._episode_off_charger_kill = np.zeros(_n, dtype=bool)  # r1 被殺時 r0 不在充電座
        # Pursuit ratio: r0 靠近 r1 的步數比例
        self._prev_dist_r0_r1 = np.full(_n, -1, dtype=np.float32)  # -1 = not initialized
        self._episode_approach_steps = np.zeros(_n, dtype=np.int32)  # r0 靠近 r1 的步數
        self._episode_both_alive_steps = np.zeros(_n, dtype=np.int32)  # 雙方都活著的步數
        # Outcome tracking (rolling window for wandb)
        self._outcome_window_size = 1000
        self._outcome_r0_wins = 0
        self._outcome_r1_wins = 0
        self._outcome_both_alive = 0
        self._outcome_both_dead = 0
        self._outcome_count = 0

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
                       next_obs: np.ndarray, terminated: np.ndarray, robot_id: int,
                       alive_mask: np.ndarray = None):
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

        # ── Filter out post-death transitions ────────────────────────────────
        # alive_mask[e]=True means robot was alive in env e BEFORE step_single.
        # Death step (alive→dead) is kept; post-death junk is skipped.
        if alive_mask is not None and not np.all(alive_mask):
            dead_envs = ~alive_mask
            # For dead envs: zero out reward, mark as not-terminated so
            # n-step buffer doesn't flush, and don't advance their pointer.
            rewards = rewards.copy()
            terminated = terminated.copy()
            rewards[dead_envs] = 0.0
            terminated[dead_envs] = False
            # We'll also skip writing to n-step buffer for dead envs below
            # by using alive_mask in the write step.

        # ── 1-step：直接 append_batch，最快 ──────────────────────────────────────
        if self.n_step == 1:
            if alive_mask is not None:
                live = np.where(alive_mask)[0]
                if len(live) == 0:
                    return
                agent_memory.append_batch(
                    obs[live], actions[live], rewards[live], next_obs[live],
                    terminated[live].astype(np.float32),
                    np.ones(len(live), dtype=np.float32)
                )
            else:
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

        # ── Determine which envs to process ──────────────────────────────────
        if alive_mask is not None:
            env_range = np.where(alive_mask)[0]
            if len(env_range) == 0:
                return
        else:
            env_range = np.arange(E)

        # ── Step 1：寫入 alive env 的新 transition ───────────────────────────────
        old_ptrs = self.ns_ptr[env_range, agent_idx].copy()
        self.ns_states     [env_range, agent_idx, old_ptrs] = obs[env_range]
        self.ns_actions    [env_range, agent_idx, old_ptrs] = actions[env_range]
        self.ns_rewards    [env_range, agent_idx, old_ptrs] = rewards[env_range]
        self.ns_next_states[env_range, agent_idx, old_ptrs] = next_obs[env_range]
        self.ns_dones      [env_range, agent_idx, old_ptrs] = terminated[env_range]

        self.ns_ptr  [env_range, agent_idx] = (old_ptrs + 1) % k
        self.ns_count[env_range, agent_idx] = np.minimum(self.ns_count[env_range, agent_idx] + 1, k)

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

    def _get_per_beta(self) -> float:
        """Linearly anneal beta from per_beta_start → 1.0 over per_beta_steps global steps."""
        beta_start = getattr(self.args, 'per_beta_start', 0.4)
        beta_steps = getattr(self.args, 'per_beta_steps', 100000)
        progress = min(1.0, self.global_step / max(beta_steps, 1))
        return beta_start + (1.0 - beta_start) * progress

    def sample_batch(self, agent_id: str):
        """
        Sample a batch from agent's replay buffer and convert to tensors.
        In PER mode returns (states, actions, rewards, next_states, dones, n_steps, indices, weights).
        In uniform mode returns (states, actions, rewards, next_states, dones, n_steps).
        """
        agent_memory = self.memories[agent_id]
        if len(agent_memory) == 0:
            return None
        batch_size = min(self.args.batch_size, len(agent_memory))

        def _to_tensor(arr, dtype=None):
            t = torch.from_numpy(arr)
            if dtype is not None:
                t = t.to(dtype)
            return t.to(self.device, non_blocking=True)

        def _to_tensor_pinned(arr, pin_buf, dtype=None):
            # Use pre-allocated pinned buffer for DMA transfer (avoids per-call pinning overhead)
            n = arr.shape[0]
            src = torch.from_numpy(arr)
            if dtype is not None:
                src = src.to(dtype)
            pin_buf[:n].copy_(src)
            return pin_buf[:n].to(self.device, non_blocking=True)

        _fast = self._pin_states is not None and batch_size <= self.args.batch_size

        if self.use_per:
            beta = self._get_per_beta()
            states_np, actions_np, rewards_np, next_states_np, dones_np, n_steps_np, indices, weights_np = \
                agent_memory.sample(batch_size, beta)
            if _fast:
                return (
                    _to_tensor_pinned(states_np,   self._pin_states),
                    _to_tensor_pinned(actions_np,  self._pin_actions),
                    _to_tensor_pinned(rewards_np,  self._pin_rewards),
                    _to_tensor_pinned(next_states_np, self._pin_nstates),
                    _to_tensor_pinned(dones_np,    self._pin_dones),
                    _to_tensor_pinned(n_steps_np,  self._pin_nsteps),
                    indices,
                    _to_tensor_pinned(weights_np,  self._pin_weights),
                )
            return (
                _to_tensor(states_np),
                _to_tensor(actions_np),
                _to_tensor(rewards_np),
                _to_tensor(next_states_np),
                _to_tensor(dones_np),
                _to_tensor(n_steps_np),
                indices,
                _to_tensor(weights_np),
            )
        else:
            states_np, actions_np, rewards_np, next_states_np, dones_np, n_steps_np = \
                agent_memory.sample(batch_size)
            if _fast:
                return (
                    _to_tensor_pinned(states_np,   self._pin_states),
                    _to_tensor_pinned(actions_np,  self._pin_actions),
                    _to_tensor_pinned(rewards_np,  self._pin_rewards),
                    _to_tensor_pinned(next_states_np, self._pin_nstates),
                    _to_tensor_pinned(dones_np,    self._pin_dones),
                    _to_tensor_pinned(n_steps_np,  self._pin_nsteps),
                )
            return (
                _to_tensor(states_np),
                _to_tensor(actions_np),
                _to_tensor(rewards_np),
                _to_tensor(next_states_np),
                _to_tensor(dones_np),
                _to_tensor(n_steps_np),
            )

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

                if agent.use_noisy:
                    # NoisyNet: no epsilon-greedy
                    obs_batch = observations[:, agent_idx]
                    obs_tensor = torch.from_numpy(obs_batch).to(self.device)
                    with torch.no_grad():
                        q_values = agent.q_net(obs_tensor)
                        actions[:, agent_idx] = q_values.argmax(dim=1).cpu().numpy()
                    continue

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

                if agent.use_noisy:
                    obs_tensor = torch.from_numpy(observations[agent_idx:agent_idx+1]).to(self.device)
                    with torch.no_grad():
                        q_values = agent.q_net(obs_tensor)
                        actions[agent_idx] = q_values.argmax().item()
                    continue

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

    def select_actions_for_robot(self, robot_id: int, observations: np.ndarray, frozen: bool = False,
                                  action_mask: np.ndarray = None) -> np.ndarray:
        """
        為指定 robot 在所有環境中選擇動作

        Args:
            robot_id: 機器人 ID (0 到 n_agents-1)
            observations: shape (num_envs, obs_dim) 的觀測 array
            frozen: if True, use greedy policy (epsilon=0.01) without decaying epsilon
            action_mask: (num_envs, 5) bool — True=legal. If provided, illegal actions
                         get Q=-inf so they are never chosen by argmax.

        Returns:
            actions: shape (num_envs,) 的動作 array
        """
        agent_id = self.agent_ids[robot_id]
        agent = self.agents[agent_id]

        num_envs = observations.shape[0]

        def _masked_argmax(q: torch.Tensor) -> np.ndarray:
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).to(q.device)
                q = q.masked_fill(~mask_t, float('-inf'))
            return q.argmax(dim=1).cpu().numpy()

        # NoisyNet: no epsilon-greedy, noise in weights provides exploration
        if agent.use_noisy:
            if robot_id in self._infer_obs_buf:
                self._infer_obs_buf[robot_id].copy_(torch.from_numpy(observations))
                obs_tensor = self._infer_obs_buf[robot_id]
            else:
                obs_tensor = torch.from_numpy(observations).to(self.device)
            with torch.no_grad():
                q_values = agent.q_net(obs_tensor)
                if action_mask is not None and self._infer_mask_buf is not None:
                    self._infer_mask_buf.copy_(torch.from_numpy(action_mask))
                    q_values = q_values.masked_fill(~self._infer_mask_buf, float('-inf'))
                    actions = q_values.argmax(dim=1).cpu().numpy()
                else:
                    actions = _masked_argmax(q_values)
            return actions

        epsilon = 0.01 if frozen else agent.epsilon
        actions = np.zeros(num_envs, dtype=np.int32)

        # 決定哪些環境用 random，哪些用網路
        random_mask = np.random.random(num_envs) < epsilon

        # Random actions — sample only from legal actions
        if random_mask.any():
            for i in np.where(random_mask)[0]:
                if action_mask is not None:
                    legal = np.where(action_mask[i])[0]
                    actions[i] = np.random.choice(legal) if len(legal) > 0 else 4
                else:
                    actions[i] = np.random.randint(0, agent.action_dim)

        # Network actions for remaining environments
        if (~random_mask).any():
            obs_batch = observations[~random_mask]
            obs_tensor = torch.from_numpy(obs_batch).to(self.device)
            nonrandom_mask = action_mask[~random_mask] if action_mask is not None else None

            with torch.no_grad():
                q_values = agent.q_net(obs_tensor)
                if nonrandom_mask is not None:
                    q_values = q_values.masked_fill(
                        ~torch.from_numpy(nonrandom_mask).to(self.device), float('-inf')
                    )
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
          [x, y, energy, self_type, wall_up, wall_down, wall_left, wall_right, opp_dx, opp_dy, ...]
          dx > 0 → opponent RIGHT → primary flee LEFT  (action 2)
          dx < 0 → opponent LEFT  → primary flee RIGHT (action 3)
          dy > 0 → opponent BELOW → primary flee UP    (action 0)
          dy < 0 → opponent ABOVE → primary flee DOWN  (action 1)

        If primary flee direction is blocked by wall, fall back to
        the other axis. If both blocked, STAY.
        """
        dx = obs[:, 8]
        dy = obs[:, 9]
        wall_up    = obs[:, 4]  # 1.0 if y==0
        wall_down  = obs[:, 5]  # 1.0 if y==n-1
        wall_left  = obs[:, 6]  # 1.0 if x==0
        wall_right = obs[:, 7]  # 1.0 if x==n-1

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

    def _seek_charger_actions(self, robot_id: int, obs: np.ndarray) -> np.ndarray:
        """
        Seek-charger heuristic: move toward the first charger, STAY when on it.
        Uses charger_dx (obs col after opponent fields) and charger_dy.

        Obs layout (agent_types_mode=off, 2 robots, 1 charger):
          [0] x, [1] y, [2] energy,
          [3] wall_up, [4] wall_down, [5] wall_left, [6] wall_right,
          [7] opp_dx, [8] opp_dy, [9] opp_energy,
          [10] charger_dx, [11] charger_dy
        """
        # Compute charger field offset: 3 + 4 + (R-1)*3 = 7 + (R-1)*3
        # For 2 robots: 7 + 3 = 10
        n_others = self.num_robots - 1
        fields_per_other = 4 if getattr(self, '_agent_types_mode', 'off') == 'observe' else 3
        charger_col = 3 + (1 if getattr(self, '_agent_types_mode', 'off') == 'observe' else 0) + 4 + n_others * fields_per_other

        cdx = obs[:, charger_col]      # positive = charger is RIGHT
        cdy = obs[:, charger_col + 1]  # positive = charger is DOWN

        # At charger → STAY
        at_charger = (np.abs(cdx) < 0.01) & (np.abs(cdy) < 0.01)

        # Move toward charger: pick axis with larger distance
        # dx>0 → RIGHT(3), dx<0 → LEFT(2); dy>0 → DOWN(1), dy<0 → UP(0)
        h_act = np.where(cdx > 0.01, 3, np.where(cdx < -0.01, 2, 4))
        v_act = np.where(cdy > 0.01, 1, np.where(cdy < -0.01, 0, 4))

        use_h = np.abs(cdx) >= np.abs(cdy)
        actions = np.where(use_h, h_act, v_act)
        actions[at_charger] = 4  # STAY at charger

        return actions.astype(np.int32)

    def _opportunistic_charger_actions(self, robot_id: int, obs: np.ndarray) -> np.ndarray:
        """
        Opportunistic charger: seek charger when r0 is far, flee when r0 is near.
        Creates 「時而充電、時而逃跑」harassment pattern.
        Uses env state directly (batch_env only).
        """
        my_pos = self.env.pos[:, robot_id]    # (N, 2) [y, x]
        cpos = self.env.charger_positions[0]  # (y, x)

        # Charger direction
        cdx = cpos[1] - my_pos[:, 1]  # positive = RIGHT
        cdy = cpos[0] - my_pos[:, 0]  # positive = DOWN
        at_charger = (cdx == 0) & (cdy == 0)

        # Find nearest alive opponent
        opp_id = 1 - robot_id if self.num_robots == 2 else 0
        opp_pos = self.env.pos[:, opp_id]
        opp_alive = self.env.alive[:, opp_id]
        odx = opp_pos[:, 1] - my_pos[:, 1]  # opponent direction
        ody = opp_pos[:, 0] - my_pos[:, 0]
        dist_to_opp = np.abs(odx) + np.abs(ody)

        # Flee threshold: flee when opponent within 2 tiles
        FLEE_DIST = 2
        should_flee = opp_alive & (dist_to_opp <= FLEE_DIST)

        # Seek-charger actions
        h_seek = np.where(cdx > 0, 3, np.where(cdx < 0, 2, 4))
        v_seek = np.where(cdy > 0, 1, np.where(cdy < 0, 0, 4))
        seek_act = np.where(np.abs(cdx) >= np.abs(cdy), h_seek, v_seek)
        seek_act[at_charger] = 4  # STAY on charger

        # Flee actions (move AWAY from opponent)
        h_flee = np.where(odx > 0, 2, np.where(odx < 0, 3, 4))
        v_flee = np.where(ody > 0, 0, np.where(ody < 0, 1, 4))
        flee_act = np.where(np.abs(odx) >= np.abs(ody), h_flee, v_flee)

        # On charger + opponent near → flee (don't stubbornly stay)
        actions = np.where(should_flee, flee_act, seek_act)
        return actions.astype(np.int32)

    def _train_vectorized(self):
        """Training loop for vectorized environments (sequential actions)"""
        print(f"Starting vectorized training with {self.num_envs} environments (sequential mode)...")

        # Reset all environments
        observations, _ = self.env.reset()  # (num_envs, n_agents, obs_dim)

        # Reset episode statistics
        self._episode_rewards.fill(0)
        self._episode_steps.fill(0)
        self._robot_death_step.fill(-1)
        self._monopoly_step.fill(-1)
        self._episode_collisions_matrix.fill(0)
        self._episode_dist_sum.fill(0)
        self._episode_dist_steps.fill(0)
        self._episode_charger_steps.fill(0)
        self._episode_dist_to_charger_sum.fill(0)
        self._episode_dist_to_charger_count.fill(0)
        self._episode_first_collision_step.fill(-1)
        self._episode_off_charger_hits.fill(0)
        self._episode_on_charger_hits.fill(0)
        self._episode_off_charger_kill.fill(False)
        self._prev_dist_r0_r1.fill(-1)
        self._episode_approach_steps.fill(0)
        self._episode_both_alive_steps.fill(0)

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
            robot_order = list(range(self.n_agents))
            if self.args.shuffle_step_order:
                np.random.shuffle(robot_order)
            for robot_id in robot_order:
                n_turns = self.robot_speeds[robot_id]
                is_scripted     = robot_id in self.scripted_robots
                is_random       = robot_id in self.random_robots
                is_safe_random  = robot_id in self.safe_random_robots
                is_flee         = robot_id in self.flee_robots
                is_frozen       = robot_id in self.frozen_robots
                is_seek_charger = robot_id in self.seek_charger_robots
                is_opp_charger  = robot_id in self.opportunistic_charger_robots
                is_learning = not (is_scripted or is_random or is_safe_random or is_flee or is_frozen or is_seek_charger or is_opp_charger)

                for turn_idx in range(n_turns):
                    is_last_turn = (turn_idx == n_turns - 1)
                    # --- Action selection (6 modes) ---
                    if is_scripted:
                        actions = np.full(self.num_envs, 4, dtype=np.int32)  # STAY
                    elif is_random:
                        actions = np.random.randint(0, 5, size=self.num_envs).astype(np.int32)
                    elif is_safe_random:
                        obs = self.env.get_observation(robot_id)
                        actions = self._safe_random_actions(robot_id, obs)
                    elif is_opp_charger:
                        obs = self.env.get_observation(robot_id)
                        actions = self._opportunistic_charger_actions(robot_id, obs)
                    elif is_flee:
                        obs = self.env.get_observation(robot_id)
                        actions = self._flee_actions(robot_id, obs)
                    elif is_frozen:
                        obs = self.env.get_observation(robot_id)
                        amask = self.env.get_valid_action_mask(robot_id) if (
                            self.use_vec_env and isinstance(self.env, BatchRobotVacuumEnv)
                        ) else None
                        actions = self.select_actions_for_robot(robot_id, obs, frozen=True, action_mask=amask)
                    elif is_seek_charger:
                        obs = self.env.get_observation(robot_id)
                        actions = self._seek_charger_actions(robot_id, obs)
                    else:
                        obs = self.env.get_observation(robot_id)
                        amask = self.env.get_valid_action_mask(robot_id) if (
                            self.use_vec_env and isinstance(self.env, BatchRobotVacuumEnv)
                        ) else None
                        actions = self.select_actions_for_robot(robot_id, obs, action_mask=amask)

                    # --- Track alive state before step (for replay buffer filtering) ---
                    if is_learning and self.use_vec_env and isinstance(self.env, BatchRobotVacuumEnv):
                        was_alive = self.env.alive[:, robot_id].copy()
                    else:
                        was_alive = None

                    # --- Execute ---
                    if self.use_vec_env and isinstance(self.env, BatchRobotVacuumEnv):
                        next_obs, rewards, terminated, truncated, infos = self.env.step_single(robot_id, actions, is_last_turn=is_last_turn)
                    else:
                        next_obs, rewards, terminated, truncated, infos = self.env.step_single(robot_id, actions)

                    # --- Bookkeeping (all modes) ---
                    if self.use_vec_env and isinstance(self.env, BatchRobotVacuumEnv):
                        # 全向量化：直接從 BatchRobotVacuumEnv 的 numpy array 讀碰撞數
                        self._episode_rewards[:, robot_id] += rewards
                        col_vec = self.env.active_collisions_with[:, robot_id, :]  # (N, R)
                        self._episode_active_collisions[:, robot_id] += col_vec.sum(axis=1)
                        self._episode_collisions_matrix[:, robot_id, :] += col_vec
                        env_terminations[:, robot_id] |= terminated

                        # --- Pursuit metrics: off-charger hit tracking ---
                        # 只計算「對非盟友的攻擊」：排除同盟友好碰撞，避免能量平均碰撞污染數字
                        if self.n_agents >= 2 and hasattr(self.env, 'charger_positions'):
                            hostile_col_vec = col_vec.copy()
                            if hasattr(self.env, '_allied_pairs'):
                                for j in range(self.n_agents):
                                    pair = (min(robot_id, j), max(robot_id, j))
                                    if pair in self.env._allied_pairs:
                                        hostile_col_vec[:, j] = 0  # 排除友好碰撞
                            hits_given = hostile_col_vec.sum(axis=1)  # (N,) 對敵人的攻擊次數
                            has_hit = hits_given > 0
                            if has_hit.any():
                                cpos = self.env.charger_positions[0]  # (y, x)
                                on_charger = ((self.env.pos[:, robot_id, 0] == cpos[0]) &
                                              (self.env.pos[:, robot_id, 1] == cpos[1]))
                                off_hit = has_hit & ~on_charger
                                on_hit = has_hit & on_charger
                                self._episode_off_charger_hits[off_hit, robot_id] += hits_given[off_hit]
                                self._episode_on_charger_hits[on_hit, robot_id] += hits_given[on_hit]

                        # Track death step for ALL robots
                        new_deaths = terminated & (self._robot_death_step[:, robot_id] == -1)
                        self._robot_death_step[new_deaths, robot_id] = self._episode_steps[new_deaths] + 1

                        # Off-charger kill: r0 killed r1 (or any robot died from r0's hit off-charger)
                        if robot_id == 0 and self.n_agents >= 2 and hasattr(self.env, 'charger_positions'):
                            # Check if any OTHER robot just died from r0's hit
                            for j in range(1, self.n_agents):
                                j_just_died = new_deaths if robot_id == j else (
                                    (self.env.alive[:, j] == False) &
                                    (self._robot_death_step[:, j] == self._episode_steps + 1) &
                                    (col_vec[:, j] > 0)
                                )
                                if j_just_died.any():
                                    cpos = self.env.charger_positions[0]
                                    r0_on_charger = ((self.env.pos[:, 0, 0] == cpos[0]) &
                                                     (self.env.pos[:, 0, 1] == cpos[1]))
                                    self._episode_off_charger_kill[j_just_died & ~r0_on_charger] = True

                        # Monopoly: all weak robots (id >= 1) dead
                        if self.n_agents > 1:
                            weak_dead = (self._robot_death_step[:, 1:] != -1).all(axis=1)
                            new_mono = weak_dead & (self._monopoly_step == -1)
                            self._monopoly_step[new_mono] = self._episode_steps[new_mono]
                    else:
                        for env_idx in range(self.num_envs):
                            self._episode_rewards[env_idx, robot_id] += rewards[env_idx]
                            for j in range(self.n_agents):
                                c = infos[env_idx].get(f'active_collisions_with_{j}', 0)
                                self._episode_active_collisions[env_idx, robot_id] += c
                                self._episode_collisions_matrix[env_idx, robot_id, j] += c
                            if terminated[env_idx]:
                                env_terminations[env_idx, robot_id] = True
                                if self._robot_death_step[env_idx, robot_id] == -1:
                                    self._robot_death_step[env_idx, robot_id] = self._episode_steps[env_idx] + 1
                        # Monopoly check (non-batch path)
                        if self.n_agents > 1:
                            for env_idx in range(self.num_envs):
                                weak_dead = all(self._robot_death_step[env_idx, j] != -1
                                                for j in range(1, self.n_agents))
                                if weak_dead and self._monopoly_step[env_idx] == -1:
                                    self._monopoly_step[env_idx] = self._episode_steps[env_idx]

                    # --- Store transitions (learning mode only) ---
                    # Skip post-death transitions: only store for envs where robot
                    # was alive BEFORE step_single. The death step itself (was_alive=True,
                    # terminated=True) is kept; post-death junk is filtered out.
                    if is_learning:
                        if self.use_vec_env:
                            self.remember_batch(obs, actions, rewards, next_obs, terminated, robot_id, alive_mask=was_alive)
                        else:
                            self.remember(obs[0], actions[0], rewards[0], next_obs[0], terminated[0], 0, robot_id)

            # 5. Advance step count after all robots have acted
            done_mask, done_envs = self.env.advance_step()

            for env_idx in range(self.num_envs):
                self._episode_steps[env_idx] += 1

            # Manhattan distance between robot_0 and robot_1 (2-agent only)
            if self.n_agents >= 2 and hasattr(self.env, 'pos'):
                # batch_env: self.env.pos shape (N, R, 2)
                dy = np.abs(self.env.pos[:, 0, 0] - self.env.pos[:, 1, 0])
                dx = np.abs(self.env.pos[:, 0, 1] - self.env.pos[:, 1, 1])
                cur_dist = (dy + dx).astype(np.float32)
                alive_both = self.env.alive[:, 0] & self.env.alive[:, 1]
                self._episode_dist_sum[alive_both] += cur_dist[alive_both]
                self._episode_dist_steps[alive_both] += 1

                # Pursuit ratio: r0 靠近 r1 的步數比例
                initialized = self._prev_dist_r0_r1 >= 0
                approached = alive_both & initialized & (cur_dist < self._prev_dist_r0_r1)
                self._episode_approach_steps[approached] += 1
                self._episode_both_alive_steps[alive_both & initialized] += 1
                # Update prev distance (only for alive-both envs; reset on death/episode end)
                self._prev_dist_r0_r1[alive_both] = cur_dist[alive_both]

            # Per-robot charger occupancy & distance to charger
            if hasattr(self.env, 'pos') and hasattr(self.env, 'charger_positions'):
                cpos = self.env.charger_positions[0]  # first charger (y, x)
                for rid in range(self.n_agents):
                    alive_mask = self.env.alive[:, rid]
                    if alive_mask.any():
                        dy_c = np.abs(self.env.pos[:, rid, 0] - cpos[0])
                        dx_c = np.abs(self.env.pos[:, rid, 1] - cpos[1])
                        dist_c = dy_c + dx_c
                        on_charger = (dist_c == 0) & alive_mask
                        self._episode_charger_steps[on_charger, rid] += 1
                        self._episode_dist_to_charger_sum[alive_mask, rid] += dist_c[alive_mask]
                        self._episode_dist_to_charger_count[alive_mask, rid] += 1

            # First collision step tracking
            if hasattr(self.env, 'active_collisions_with'):
                total_collisions = self.env.active_collisions_with.sum(axis=(1, 2))  # (N,)
                new_first = (total_collisions > 0) & (self._episode_first_collision_step == -1)
                self._episode_first_collision_step[new_first] = self._episode_steps[new_first]

            # Train (controlled by train_frequency)
            if self.global_step % self.train_frequency == 0:
                # 每個 agent 從自己的 buffer 抽樣訓練 (skip frozen/scripted/random)
                frozen_agent_ids = {f'robot_{i}' for i in self.frozen_robots | self.scripted_robots | self.random_robots | self.safe_random_robots | self.flee_robots | self.seek_charger_robots | self.opportunistic_charger_robots}
                for agent_id in self.agent_ids:
                    if agent_id in frozen_agent_ids:
                        continue
                    agent_memory = self.memories[agent_id]
                    if len(agent_memory) >= max(self.args.replay_start_size, self.args.batch_size):
                        batch = self.sample_batch(agent_id)

                        if batch is not None:
                            if self.use_per:
                                *batch6, indices, weights = batch
                                train_stats = self.agents[agent_id].train_step(tuple(batch6), weights=weights)
                                td_errors = train_stats.get('td_errors')
                                if td_errors is not None:
                                    agent_memory.update_priorities(indices, td_errors)
                            else:
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
                    # Death steps for all robots
                    death_parts = []
                    for i in range(self.n_agents):
                        ds = self._robot_death_step[env_idx, i]
                        death_parts.append(f"r{i}@{ds}" if ds != -1 else f"r{i}_alive")
                    death_str = " ".join(death_parts)
                    mono = self._monopoly_step[env_idx]
                    mono_str = f" mono@{mono}" if mono != -1 else ""
                    # Pursuit info
                    off_h = self._episode_off_charger_hits[env_idx, 0]
                    on_h = self._episode_on_charger_hits[env_idx, 0]
                    okill = "✓" if self._episode_off_charger_kill[env_idx] else ""
                    pursuit_str = f" hunt:{off_h}/{off_h+on_h}{okill}" if (off_h + on_h) > 0 else ""
                    bas = self._episode_both_alive_steps[env_idx]
                    pr_str = f" pr:{self._episode_approach_steps[env_idx]/bas:.2f}" if bas > 0 else ""
                    # Alliance hit stats: show rA->rB hits on enemies + friendly collisions
                    alliance_str = ""
                    if self._alliance_groups:
                        # Collect all allied robot IDs
                        allied_ids = set()
                        for grp in self._alliance_groups:
                            allied_ids |= grp
                        # Enemy IDs = all robots not in any alliance group
                        enemy_ids = [r for r in range(self.n_agents) if r not in allied_ids]
                        parts = []
                        if enemy_ids:
                            for attacker in sorted(allied_ids):
                                for victim in enemy_ids:
                                    hits = int(self._episode_collisions_matrix[env_idx, attacker, victim])
                                    parts.append(f"r{attacker}→r{victim}:{hits}")
                        # Friendly collisions: sum both directions for each allied pair
                        for grp in self._alliance_groups:
                            grp_list = sorted(grp)
                            for gi in range(len(grp_list)):
                                for gj in range(gi + 1, len(grp_list)):
                                    ri, rj = grp_list[gi], grp_list[gj]
                                    fc = (int(self._episode_collisions_matrix[env_idx, ri, rj])
                                          + int(self._episode_collisions_matrix[env_idx, rj, ri]))
                                    parts.append(f"r{ri}↔r{rj}:{fc}")
                        if parts:
                            alliance_str = " | " + " ".join(parts)
                    print(f"[Episode {self.total_episodes}] Steps:{steps} | "
                          f"{rewards_str} | Collisions({collisions_str}) | {death_str}{mono_str}{pursuit_str}{pr_str}{alliance_str} | ε:{current_epsilon:.3f}")

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
                        ds = self._robot_death_step[env_idx, i]
                        log_dict[f"robot_{i}/episode_reward"] = self._episode_rewards[env_idx, i]
                        log_dict[f"robot_{i}/active_collisions"] = int(self._episode_active_collisions[env_idx, i])
                        log_dict[f"robot_{i}/survived"] = int(ds == -1)
                        if ds != -1:
                            log_dict[f"robot_{i}/death_step"] = int(ds)
                        # Per-pair collision: how many times robot_i attacked robot_j
                        for j in range(self.n_agents):
                            if i != j:
                                log_dict[f"robot_{i}/attacks_robot_{j}"] = int(self._episode_collisions_matrix[env_idx, i, j])
                    if mono != -1:
                        log_dict["robot_0/monopoly_step"] = int(mono)
                    # Mean Manhattan distance (only when both robots were alive)
                    if self.n_agents >= 2 and self._episode_dist_steps[env_idx] > 0:
                        log_dict["behavior/mean_distance"] = float(
                            self._episode_dist_sum[env_idx] / self._episode_dist_steps[env_idx]
                        )
                    # Kill order: sorted death steps among weak robots
                    if self.n_agents > 1:
                        weak_deaths = [(i, self._robot_death_step[env_idx, i])
                                       for i in range(1, self.n_agents)
                                       if self._robot_death_step[env_idx, i] != -1]
                        weak_deaths.sort(key=lambda x: x[1])
                        for rank, (rid, ds) in enumerate(weak_deaths):
                            log_dict[f"kill_order/rank{rank+1}_robot"] = rid
                            log_dict[f"kill_order/rank{rank+1}_step"] = int(ds)

                    # --- Behavior metrics ---
                    # Per-robot charger occupancy & avg distance to charger
                    for i in range(self.n_agents):
                        cs = self._episode_charger_steps[env_idx, i]
                        log_dict[f"behavior/r{i}_charger_steps"] = int(cs)
                        if steps > 0:
                            log_dict[f"behavior/r{i}_charger_occupancy"] = float(cs) / steps
                        dc_count = self._episode_dist_to_charger_count[env_idx, i]
                        if dc_count > 0:
                            log_dict[f"behavior/r{i}_avg_dist_to_charger"] = float(
                                self._episode_dist_to_charger_sum[env_idx, i] / dc_count)

                    # First collision step
                    fcs = self._episode_first_collision_step[env_idx]
                    if fcs != -1:
                        log_dict["behavior/first_collision_step"] = int(fcs)

                    # Pursuit metrics
                    for i in range(self.n_agents):
                        off_h = self._episode_off_charger_hits[env_idx, i]
                        on_h = self._episode_on_charger_hits[env_idx, i]
                        total_h = off_h + on_h
                        if total_h > 0:
                            log_dict[f"pursuit/r{i}_off_charger_hits"] = int(off_h)
                            log_dict[f"pursuit/r{i}_on_charger_hits"] = int(on_h)
                            log_dict[f"pursuit/r{i}_off_charger_hit_ratio"] = float(off_h) / total_h
                    if self._episode_off_charger_kill[env_idx]:
                        log_dict["pursuit/off_charger_kill"] = 1
                    elif self._robot_death_step[env_idx, 1] != -1 if self.n_agents >= 2 else False:
                        log_dict["pursuit/off_charger_kill"] = 0  # killed on charger
                    # Pursuit ratio
                    bas = self._episode_both_alive_steps[env_idx]
                    if bas > 0:
                        log_dict["pursuit/r0_pursuit_ratio"] = float(
                            self._episode_approach_steps[env_idx]) / bas

                    # Episode outcome
                    n_survived = sum(1 for i in range(self.n_agents) if self._robot_death_step[env_idx, i] == -1)
                    log_dict["outcome/num_survived"] = n_survived
                    if self.n_agents == 2:
                        r0_alive = self._robot_death_step[env_idx, 0] == -1
                        r1_alive = self._robot_death_step[env_idx, 1] == -1
                        if r0_alive and not r1_alive:
                            self._outcome_r0_wins += 1
                        elif r1_alive and not r0_alive:
                            self._outcome_r1_wins += 1
                        elif r0_alive and r1_alive:
                            self._outcome_both_alive += 1
                        else:
                            self._outcome_both_dead += 1
                        self._outcome_count += 1
                        if self._outcome_count >= 100:
                            total = self._outcome_count
                            log_dict["outcome/r0_win_rate"] = self._outcome_r0_wins / total
                            log_dict["outcome/r1_win_rate"] = self._outcome_r1_wins / total
                            log_dict["outcome/both_alive_rate"] = self._outcome_both_alive / total
                            log_dict["outcome/both_dead_rate"] = self._outcome_both_dead / total
                            # Reset window
                            self._outcome_r0_wins = 0
                            self._outcome_r1_wins = 0
                            self._outcome_both_alive = 0
                            self._outcome_both_dead = 0
                            self._outcome_count = 0

                    wandb.log(log_dict)

                # Reset episode statistics for this environment
                self._episode_rewards[env_idx].fill(0)
                self._episode_steps[env_idx] = 0
                self._episode_active_collisions[env_idx].fill(0)
                self._episode_collisions_matrix[env_idx].fill(0)
                env_terminations[env_idx, :] = False
                self._robot_death_step[env_idx].fill(-1)
                self._monopoly_step[env_idx] = -1
                self._episode_dist_sum[env_idx] = 0
                self._episode_dist_steps[env_idx] = 0
                self._episode_charger_steps[env_idx].fill(0)
                self._episode_dist_to_charger_sum[env_idx].fill(0)
                self._episode_dist_to_charger_count[env_idx].fill(0)
                self._episode_first_collision_step[env_idx] = -1
                self._episode_off_charger_hits[env_idx].fill(0)
                self._episode_on_charger_hits[env_idx].fill(0)
                self._episode_off_charger_kill[env_idx] = False
                self._prev_dist_r0_r1[env_idx] = -1
                self._episode_approach_steps[env_idx] = 0
                self._episode_both_alive_steps[env_idx] = 0

                # Save models periodically (skip the very first step = offset itself)
                new_ep = self.total_episodes - self.episode_offset
                if new_ep > 0 and new_ep % self.args.save_frequency == 0:
                    self.save_models(f"episode_{self.total_episodes}")

            # Update epsilon based on training progress (absolute, accounts for loaded checkpoint)
            if len(done_envs) > 0:
                total_planned = self.episode_offset + self.args.num_episodes
                progress = min(1.0, self.total_episodes / max(total_planned, 1))
                for agent in self.agents.values():
                    agent.set_epsilon_by_progress(progress)
                # Auto-freeze: add robot to frozen_robots once epsilon reaches epsilon_end
                for rid in self.auto_freeze_robots:
                    agent = self.agents.get(f'robot_{rid}')
                    if agent is not None and rid not in self.frozen_robots:
                        if agent.epsilon <= agent.epsilon_end * 2:
                            self.frozen_robots.add(rid)
                            if rid not in self._auto_freeze_logged:
                                print(f"[Auto-Freeze] robot_{rid} frozen at episode {self.total_episodes} (ε={agent.epsilon:.5f})")
                                self._auto_freeze_logged.add(rid)

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
                                if self.use_per:
                                    *batch6, indices, weights = batch
                                    train_stats = self.agents[agent_id].train_step(tuple(batch6), weights=weights)
                                    td_errors = train_stats.get('td_errors')
                                    if td_errors is not None:
                                        agent_memory.update_priorities(indices, td_errors)
                                else:
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

            # Update epsilon based on training progress (absolute, accounts for loaded checkpoint)
            total_planned = getattr(self, 'episode_offset', 0) + self.args.num_episodes
            progress = min(1.0, (episode + 1) / max(total_planned, 1))
            for agent in self.agents.values():
                agent.set_epsilon_by_progress(progress)
            # Auto-freeze: add robot to frozen_robots once epsilon reaches epsilon_end
            for rid in self.auto_freeze_robots:
                agent = self.agents.get(f'robot_{rid}')
                if agent is not None and rid not in self.frozen_robots:
                    if agent.epsilon <= agent.epsilon_end * 2:
                        self.frozen_robots.add(rid)
                        if rid not in self._auto_freeze_logged:
                            print(f"[Auto-Freeze] robot_{rid} frozen at episode {episode + 1} (ε={agent.epsilon:.5f})")
                            self._auto_freeze_logged.add(rid)

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

        # Write training_config.json to model root (once, on first save)
        self._write_training_config()

        print(f"Models saved to: {save_path}")

    def _write_training_config(self):
        """Write env/arch parameters to training_config.json in save_dir for reproducibility."""
        import json as _json
        config_path = os.path.join(self.save_dir, "training_config.json")
        if os.path.exists(config_path):
            return
        a = self.args
        n = self.num_robots
        def _get(attr, default=None):
            return getattr(a, attr, default)
        config = {
            "env_n":              _get("env_n"),
            "num_robots":         n,
            "charger_positions":  _get("charger_positions"),
            "charger_range":      _get("charger_range", 0),
            "exclusive_charging": _get("exclusive_charging", False),
            "no_dust":            _get("no_dust", False),
            "e_move":             _get("e_move"),
            "e_charge":           _get("e_charge"),
            "e_collision":        _get("e_collision"),
            "e_boundary":         _get("e_boundary"),
            "e_decay":            _get("e_decay", 0.0),
            "robot_energies":     [_get(f"robot_{i}_energy") for i in range(n)],
            "robot_speeds":       [_get(f"robot_{i}_speed") for i in range(n)],
            "robot_attack_powers":[_get(f"robot_{i}_attack_power") for i in range(n)],
            "robot_stun_steps":   [_get(f"robot_{i}_stun_steps") for i in range(n)],
            "robot_docking_steps":[_get(f"robot_{i}_docking_steps") for i in range(n)],
            "alliance_groups":    _get("alliance_groups"),
            "alliance_zone":      _get("alliance_zone", False),
            "energy_sharing_mode":        _get("energy_sharing_mode", "none"),
            "energy_sharing_events":      _get("energy_sharing_events", "charge,collision"),
            "energy_sharing_self_weight": _get("energy_sharing_self_weight"),
            "energy_sharing_ally_weight": _get("energy_sharing_ally_weight"),
            "dueling":            _get("dueling", False),
            "noisy":              _get("noisy", False),
            "c51":                _get("c51", False),
            "max_episode_steps":  _get("max_episode_steps"),
        }
        with open(config_path, "w") as f:
            _json.dump(config, f, indent=2)
        print(f"[Config] training_config.json written to {self.save_dir}")


def parse_alliance_groups(s: str, num_robots: int):
    """
    把 "0,1" 解析成 [{0, 1}]。
    多個 alliance group 用 ";" 分隔，例如 "0,1;2,3" → [{0, 1}, {2, 3}]。
    非法格式印 warning 並回傳 None（不 crash）。
    """
    if not s:
        return None
    try:
        groups = []
        for group_str in s.split(';'):
            ids = set(int(x.strip()) for x in group_str.split(',') if x.strip())
            if not ids:
                continue
            for rid in ids:
                if rid < 0 or rid >= num_robots:
                    print(f"[WARNING] alliance-groups: robot id {rid} out of range (num_robots={num_robots}), ignoring group")
                    ids = None
                    break
            if ids:
                groups.append(ids)
        return groups if groups else None
    except Exception as e:
        print(f"[WARNING] Failed to parse --alliance-groups '{s}': {e}. Alliance sharing disabled.")
        return None


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
    parser.add_argument("--e-move", type=float, default=1, help="Energy cost per move")
    parser.add_argument("--e-charge", type=float, default=1.5, help="Energy gain per charge")
    parser.add_argument("--e-collision", type=int, default=3, help="Energy loss per collision")
    parser.add_argument("--e-boundary", type=int, default=50, help="Energy loss when hitting wall")
    parser.add_argument("--energy-cap", type=float, default=None, help="Max energy (None=no cap, overcharge allowed)")
    parser.add_argument("--e-decay", type=float, default=0.0, help="Passive energy drain per step for all alive robots")
    parser.add_argument("--reward-mode", type=str, default="delta-energy",
                        choices=["delta-energy", "hp-ratio"],
                        help="Reward mode: delta-energy (default) or hp-ratio")
    parser.add_argument("--reward-alpha", type=float, default=0.05,
                        help="Reward scaling factor (default 0.05 for delta-energy, 0.2 recommended for hp-ratio)")
    parser.add_argument("--robot-0-attack-power", type=float, default=None,
                        help="Attack power (collision damage dealt) for robot 0. Default: e_collision")
    parser.add_argument("--robot-1-attack-power", type=float, default=None,
                        help="Attack power (collision damage dealt) for robot 1. Default: e_collision")
    parser.add_argument("--robot-2-attack-power", type=float, default=None,
                        help="Attack power (collision damage dealt) for robot 2. Default: e_collision")
    parser.add_argument("--robot-3-attack-power", type=float, default=None,
                        help="Attack power (collision damage dealt) for robot 3. Default: e_collision")
    parser.add_argument("--thief-spawn", action="store_true", default=False,
                        help="Thief scenario spawn: weak (robot 1) near charger, strong (robot 0) far from charger")
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
                        help="充電座獨佔模式：有其他 robot 在範圍內時充電無效")
    parser.add_argument("--docking-steps", type=int, default=0,
                        help="Docking 機制：robot 需連續站在充電座 N 步後才能開始充電（0=即時充電）")
    parser.add_argument("--robot-0-docking-steps", type=int, default=None,
                        help="Robot 0 專用 docking steps（覆蓋 --docking-steps）")
    parser.add_argument("--robot-1-docking-steps", type=int, default=None,
                        help="Robot 1 專用 docking steps（覆蓋 --docking-steps）")
    parser.add_argument("--robot-2-docking-steps", type=int, default=None,
                        help="Robot 2 專用 docking steps（覆蓋 --docking-steps）")
    parser.add_argument("--robot-3-docking-steps", type=int, default=None,
                        help="Robot 3 專用 docking steps（覆蓋 --docking-steps）")
    parser.add_argument("--stun-steps", type=int, default=0,
                        help="被撞後暈眩步數：robot 被碰撞後強制 STAY N 步（0=無暈眩）")
    parser.add_argument("--robot-0-stun-steps", type=int, default=None,
                        help="Robot 0 專用暈眩步數（覆蓋 --stun-steps）")
    parser.add_argument("--robot-1-stun-steps", type=int, default=None,
                        help="Robot 1 專用暈眩步數（覆蓋 --stun-steps）")
    parser.add_argument("--robot-2-stun-steps", type=int, default=None,
                        help="Robot 2 專用暈眩步數（覆蓋 --stun-steps）")
    parser.add_argument("--robot-3-stun-steps", type=int, default=None,
                        help="Robot 3 專用暈眩步數（覆蓋 --stun-steps）")
    parser.add_argument("--shuffle-step-order", action="store_true", default=False,
                        help="每步隨機化 robot 行動順序（消除 sequential stepping 的先後手優勢）")
    parser.add_argument("--charger-range", type=int, default=0,
                        help="充電範圍：0=只有站在充電座上（預設），1=3×3")
    parser.add_argument("--scripted-robots", type=str, default="",
                        help='Comma-separated robot IDs to fix as STAY (no training), e.g. "1" or "1,2"')
    parser.add_argument("--random-robots", type=str, default="",
                        help='Comma-separated robot IDs that use random actions (no training), e.g. "1"')
    parser.add_argument("--safe-random-robots", type=str, default="",
                        help='Comma-separated robot IDs that use wall-avoiding random walk (no training), e.g. "1"')
    parser.add_argument("--flee-robots", type=str, default="",
                        help='Comma-separated robot IDs that use flee heuristic (no training), e.g. "1"')
    parser.add_argument("--frozen-robots", type=str, default="",
                        help='Comma-separated robot IDs that use their loaded model (greedy) but are NOT trained, e.g. "0"')
    parser.add_argument("--auto-freeze-robots", type=str, default="",
                        help='Comma-separated robot IDs to auto-freeze (stop training) once their epsilon reaches epsilon_end, e.g. "1"')
    parser.add_argument("--seek-charger-robots", type=str, default="",
                        help='Comma-separated robot IDs that walk toward charger and STAY (not trained), e.g. "1"')
    parser.add_argument("--opportunistic-charger-robots", type=str, default="",
                        help='Comma-separated robot IDs: seek charger when r0 far, flee when r0 near (not trained)')
    parser.add_argument("--random-start-robots", type=str, default="all",
                        help='Comma-separated robot IDs to randomize start position each episode, "all" for all robots (default), "" to disable')
    parser.add_argument("--robot-start-positions", type=str, default=None,
                        help='Fixed start positions as "y0,x0;y1,x1;..." (overrides corner defaults), e.g. "0,0;3,3"')
    parser.add_argument("--load-model-dir", type=str, default=None,
                        help='Load pre-trained models from this dir to continue training (curriculum phase)')

    # Agent type system (trait.md control experiment)
    parser.add_argument("--agent-types-mode", type=str, default="off", choices=["off", "observe"],
                        help='Agent type visibility: off=no type info (pad 0), observe=type visible in obs')
    parser.add_argument("--triangle-agent-id", type=int, default=None,
                        help='Which agent is triangle (others are circle). None=all circle.')
    parser.add_argument("--heterotype-charge-mode", type=str, default="off",
                        choices=["off", "local-penalty"],
                        help='Heterotype charging penalty: off=no penalty, local-penalty=mixed types reduce charging')
    parser.add_argument("--heterotype-charge-factor", type=float, default=1.0,
                        help='Charging efficiency when mixed types share a charger (0.7=70%%, 1.0=no penalty)')

    # DQN hyperparameters
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n-step", type=int, default=1, help="N-step return")

    # Epsilon configuration
    parser.add_argument("--use-epsilon-decay", action=argparse.BooleanOptionalAction, default=True, help="Use epsilon decay (use --no-use-epsilon-decay to disable)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Fixed epsilon")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon for decay")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate (only used for exponential schedule)")
    parser.add_argument("--epsilon-schedule", type=str, default="linear",
                        choices=["linear", "exponential", "exp_tail", "sigmoid"],
                        help="Epsilon decay schedule: linear (default), exponential, exp_tail, sigmoid")
    parser.add_argument("--exp-tail-k", type=float, default=15.0,
                        help="exp_tail schedule: decay rate k (ε = end + (start-end)*exp(-k*x), default 15)")
    parser.add_argument("--sigmoid-s", type=float, default=14.0,
                        help="sigmoid schedule: steepness s (default 14)")
    parser.add_argument("--sigmoid-c", type=float, default=0.45,
                        help="sigmoid schedule: midpoint c in [0,1] (default 0.45)")

    # Per-robot epsilon overrides (fallback to global --epsilon-* if not set)
    for _ri in range(4):
        parser.add_argument(f"--robot-{_ri}-epsilon-start",    type=float, default=None,
                            help=f"robot_{_ri} epsilon start (overrides --epsilon-start)")
        parser.add_argument(f"--robot-{_ri}-epsilon-end",      type=float, default=None,
                            help=f"robot_{_ri} epsilon end (overrides --epsilon-end)")
        parser.add_argument(f"--robot-{_ri}-epsilon-schedule", type=str,   default=None,
                            choices=["linear", "exponential", "exp_tail", "sigmoid"],
                            help=f"robot_{_ri} epsilon schedule (overrides --epsilon-schedule)")
        parser.add_argument(f"--robot-{_ri}-exp-tail-k",  type=float, default=None,
                            help=f"robot_{_ri} exp_tail k (overrides --exp-tail-k)")
        parser.add_argument(f"--robot-{_ri}-sigmoid-s",   type=float, default=None,
                            help=f"robot_{_ri} sigmoid steepness s (overrides --sigmoid-s)")
        parser.add_argument(f"--robot-{_ri}-sigmoid-c",   type=float, default=None,
                            help=f"robot_{_ri} sigmoid midpoint c (overrides --sigmoid-c)")

    parser.add_argument("--memory-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--replay-start-size", type=int, default=1000, help="Minimum buffer size before training")
    parser.add_argument("--target-update-frequency", type=int, default=1000, help="Target network update frequency")

    # Rainbow DQN components
    parser.add_argument("--dueling", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Dueling DQN (V/A stream split, use --no-dueling to disable)")
    parser.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable NoisyNet (replaces epsilon-greedy, use --no-noisy to disable)")
    parser.add_argument("--c51", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable C51 distributional DQN (use --no-c51 to disable)")
    parser.add_argument("--num-atoms", type=int, default=51,
                        help="Number of atoms for C51 distributional DQN")
    parser.add_argument("--v-min", type=float, default=-100.0,
                        help="C51 minimum value support")
    parser.add_argument("--v-max", type=float, default=100.0,
                        help="C51 maximum value support")

    # Prioritized Experience Replay (PER)
    parser.add_argument("--per", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Prioritized Experience Replay (use --no-per to disable)")
    parser.add_argument("--per-alpha", type=float, default=0.6,
                        help="PER priority exponent α (0=uniform, 1=fully prioritized)")
    parser.add_argument("--per-beta-start", type=float, default=0.4,
                        help="PER IS-weight exponent β starting value (anneals to 1.0)")
    parser.add_argument("--per-beta-steps", type=int, default=100000,
                        help="Global steps over which β anneals from per-beta-start to 1.0")
    parser.add_argument("--recency-lambda", type=float, default=0.0,
                        help="Recency weighting strength for PER (0=disabled, >0 favours newer transitions). "
                             "Effective only when PER is enabled.")

    # Vectorized training parameters
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Subprocess workers for env parallelism (1=disabled, N>1=SubprocVecEnv). "
                             "num_envs must be divisible by num_workers. "
                             "建議: --num-envs 32 --num-workers 8 (8 cores, 4 envs each)")
    parser.add_argument("--train-frequency", type=int, default=4, help="Train every N steps")
    parser.add_argument("--batch-env", action=argparse.BooleanOptionalAction, default=True,
                        help="使用全 numpy 批次環境（BatchRobotVacuumEnv），大幅加速 env_step (use --no-batch-env to disable)")

    # Training settings
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--save-frequency", type=int, default=1000, help="Model save frequency (episodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index (e.g. 0, 3, 4). If not set, uses first available.")
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable torch.compile (use --no-use-torch-compile to disable)")

    # Wandb and logging
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--wandb-project", type=str, default="multi-robot-idqn", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default="idqn-vec", help="Wandb run name")
    parser.add_argument("--wandb-mode", type=str, default="offline", help="Wandb mode")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save models")

    # Auto-evaluation after training
    parser.add_argument("--eval-after-training", action=argparse.BooleanOptionalAction, default=True, help="Run evaluation after training (use --no-eval-after-training to disable)")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Max steps for evaluation")
    parser.add_argument("--plot-evolution-after-training", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-run plot_attack_evolution.py (greedy, model r1) after training completes (use --no-plot-evolution-after-training to disable)")

    # Alliance energy-sharing parameters (all optional, default = disabled)
    parser.add_argument("--alliance-groups", type=str, default=None,
                        help="Alliance groups as comma-separated robot IDs, e.g. '0,1' or '0,1;2,3'")
    parser.add_argument("--alliance-zone", action="store_true", default=False,
                        help="Enforce Chebyshev ≤ 1 between allied robots (zone moves with them)")
    parser.add_argument("--energy-sharing-mode", type=str, default="none",
                        choices=["none", "event_only"],
                        help="Energy sharing mode: none (default) or event_only")
    parser.add_argument("--energy-sharing-events", type=str, default="charge,collision",
                        help="Comma-separated list of events to share: charge, collision, move, decay, boundary")
    parser.add_argument("--energy-sharing-self-weight", type=float, default=2.0 / 3.0,
                        help="Weight for own energy events (default 2/3)")
    parser.add_argument("--energy-sharing-ally-weight", type=float, default=1.0 / 3.0,
                        help="Weight per ally for shared energy events (default 1/3)")

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

        # Add robot start positions if specified
        if getattr(args, 'robot_start_positions', None):
            eval_cmd.extend(["--robot-start-positions", args.robot_start_positions])

        # Add boolean flags
        if args.no_dust:
            eval_cmd.append("--no-dust")
        if args.exclusive_charging:
            eval_cmd.append("--exclusive-charging")
        if args.charger_range != 1:
            eval_cmd.extend(["--charger-range", str(args.charger_range)])
        if args.scripted_robots:
            eval_cmd.extend(["--scripted-robots", args.scripted_robots])
        if args.random_robots:
            eval_cmd.extend(["--random-robots", args.random_robots])
        if args.safe_random_robots:
            eval_cmd.extend(["--safe-random-robots", args.safe_random_robots])
        if args.flee_robots:
            eval_cmd.extend(["--flee-robots", args.flee_robots])

        # Add agent type flags
        if args.agent_types_mode != 'off':
            eval_cmd.extend(["--agent-types-mode", args.agent_types_mode])
        if args.triangle_agent_id is not None:
            eval_cmd.extend(["--triangle-agent-id", str(args.triangle_agent_id)])
        if args.heterotype_charge_mode != 'off':
            eval_cmd.extend(["--heterotype-charge-mode", args.heterotype_charge_mode])
            eval_cmd.extend(["--heterotype-charge-factor", str(args.heterotype_charge_factor)])
        if args.energy_cap is not None:
            eval_cmd.extend(["--energy-cap", str(args.energy_cap)])
        if args.e_decay > 0:
            eval_cmd.extend(["--e-decay", str(args.e_decay)])

        # Add Rainbow DQN flags
        if args.dueling:
            eval_cmd.append("--dueling")
        if args.noisy:
            eval_cmd.append("--noisy")
        if args.c51:
            eval_cmd.append("--c51")
            eval_cmd.extend(["--num-atoms", str(args.num_atoms)])
            eval_cmd.extend(["--v-min", str(args.v_min)])
            eval_cmd.extend(["--v-max", str(args.v_max)])

        # Add wandb settings
        if args.wandb_entity:
            eval_cmd.extend(["--wandb-entity", args.wandb_entity])
        eval_cmd.extend(["--wandb-project", "robot-vacuum-eval"])
        eval_cmd.extend(["--wandb-run-name", f"{args.wandb_run_name}-eval"])
        eval_cmd.extend(["--wandb-mode", args.wandb_mode])

        # Run evaluation
        subprocess.run(eval_cmd)

    # Auto attack-evolution plot after training
    if getattr(args, 'plot_evolution_after_training', False):
        import subprocess
        run_name = args.wandb_run_name or os.path.basename(trainer.save_dir)
        output_dir = os.path.join("./analyze", run_name)
        os.makedirs(output_dir, exist_ok=True)

        evo_cmd = [
            "python", "tools/plot_attack_evolution.py",
            "--model-dir", trainer.save_dir,
            "--r1-policy", "model",
            "--num-points", "60",
            "--episodes-per-ckpt", "50",
            "--output-dir", output_dir,
            # Env params
            "--env-n", str(args.env_n),
            "--e-move", str(args.e_move),
            "--e-charge", str(args.e_charge),
            "--e-decay", str(args.e_decay),
            "--e-collision", str(args.e_collision),
            "--e-boundary", str(args.e_boundary),
            "--charger-positions", args.charger_positions,
            "--charger-range", str(args.charger_range),
            "--robot-0-energy", str(args.robot_0_energy),
            "--robot-1-energy", str(args.robot_1_energy),
            "--robot-0-speed", str(args.robot_0_speed),
            "--robot-1-speed", str(args.robot_1_speed),
            "--robot-0-attack", str(args.robot_0_attack_power),
            "--robot-1-attack", str(args.robot_1_attack_power),
            "--robot-0-docking-steps", str(args.robot_0_docking_steps or 0),
            "--robot-1-docking-steps", str(args.robot_1_docking_steps or 0),
            "--robot-0-stun-steps", str(args.robot_0_stun_steps),
            "--robot-1-stun-steps", str(args.robot_1_stun_steps),
            "--max-steps", str(args.max_episode_steps),
        ]
        if args.exclusive_charging:
            evo_cmd.append("--exclusive-charging")
        if getattr(args, 'dueling', False):
            evo_cmd.append("--dueling")
        if getattr(args, 'noisy', False):
            evo_cmd.append("--noisy")
        if getattr(args, 'c51', False):
            evo_cmd.append("--c51")

        print(f"\n{'='*50}")
        print(f"Running attack evolution plot → {output_dir}")
        print(f"{'='*50}\n")
        subprocess.run(evo_cmd)


if __name__ == "__main__":
    main()
