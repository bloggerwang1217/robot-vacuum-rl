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
from vec_env import VectorizedRobotVacuumEnv


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
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.q_net.state_dict())


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
            'e_move': args.e_move,
            'e_charge': args.e_charge,
            'e_collision': args.e_collision,
            'e_boundary': args.e_boundary,
            'n_steps': args.max_episode_steps,
            'charger_positions': charger_positions
        }

        # Create vectorized environment
        if self.num_envs == 1:
            # 單環境模式（向後兼容）
            self.use_vec_env = False
            self.env = RobotVacuumGymEnv(**env_kwargs)
            observation_dim = self.env.observation_space.shape[0]
        else:
            # Vectorized 模式
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

        # Independent Replay Buffer (每個 agent 獨立)
        self.memories = {agent_id: deque(maxlen=args.memory_size) for agent_id in self.agent_ids}
        
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

        # Training counters
        self.global_step = 0
        self.total_episodes = 0

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
        self._episode_rewards = np.zeros((self.num_envs if self.use_vec_env else 1, self.n_agents))
        self._episode_steps = np.zeros(self.num_envs if self.use_vec_env else 1, dtype=np.int32)
        self._episode_immediate_kills = np.zeros(self.num_envs if self.use_vec_env else 1, dtype=np.int32)

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
                # Episode 結束：將 buffer 中所有剩餘的 transition 都存入
                # 這確保直接導致死亡的動作能被學習到（1-step transition with full penalty）
                buffer_list = list(buffer)
                _, _, _, end_next_state, end_done = buffer_list[-1]

                for start_idx in range(len(buffer_list)):
                    # 從 start_idx 開始計算到結尾的 n-step return
                    n_step_return = 0
                    for offset, (_, _, r, _, _) in enumerate(buffer_list[start_idx:]):
                        n_step_return += (gamma ** offset) * r

                    # 取該位置的 state, action
                    start_state, start_action, _, _, _ = buffer_list[start_idx]

                    # 實際步數 = 從 start_idx 到結尾的長度
                    actual_n_step = len(buffer_list) - start_idx

                    # 存入該 agent 的 replay buffer
                    agent_memory.append((start_state, start_action, n_step_return, end_next_state, end_done, actual_n_step))

                buffer.clear()

            elif len(buffer) == self.n_step:
                # 正常情況：累積滿 n 步
                n_step_return = 0
                for idx, (_, _, r, _, _) in enumerate(buffer):
                    n_step_return += (gamma ** idx) * r

                start_state, start_action, _, _, _ = buffer[0]
                _, _, _, end_next_state, end_done = buffer[-1]

                # 存入該 agent 的 replay buffer
                agent_memory.append((start_state, start_action, n_step_return, end_next_state, end_done, self.n_step))

                buffer.popleft()

    def sample_batch(self, agent_id: str) -> Tuple:
        """Sample a batch from agent's independent memory and convert to tensors"""
        agent_memory = self.memories[agent_id]
        batch_size = min(self.args.batch_size, len(agent_memory))
        if batch_size == 0:
            return None
        batch = random.sample(list(agent_memory), batch_size)
        states, actions, rewards, next_states, dones, actual_n_steps = zip(*batch)

        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        actual_n_steps_np = np.array(actual_n_steps, dtype=np.float32)

        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
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

        while self.total_episodes < self.args.num_episodes:
            # Sequential actions: each robot acts one at a time (across all envs)
            for robot_id in range(self.n_agents):
                agent_id = self.agent_ids[robot_id]

                # 1. Get current observation for this robot (all envs)
                obs = self.env.get_observation(robot_id)  # (num_envs, obs_dim)

                # 2. Select actions for this robot (all envs)
                actions = self.select_actions_for_robot(robot_id, obs)  # (num_envs,)

                # 3. Execute actions for this robot (all envs)
                next_obs, rewards, terminated, truncated, infos = self.env.step_single(robot_id, actions)

                # 4. Store transitions
                for env_idx in range(self.num_envs):
                    self.remember(
                        obs[env_idx],
                        actions[env_idx],
                        rewards[env_idx],
                        next_obs[env_idx],
                        terminated[env_idx],
                        env_idx,
                        robot_id
                    )
                    self._episode_rewards[env_idx, robot_id] += rewards[env_idx]

                    # Update termination status
                    if terminated[env_idx]:
                        env_terminations[env_idx, robot_id] = True

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
                    print(f"[Episode {self.total_episodes}] Steps: {steps} | "
                          f"Mean Reward: {mean_reward:.2f} | Epsilon: {current_epsilon:.3f}")

                    # 計算所有 buffer 的總大小
                    total_buffer_size = sum(len(m) for m in self.memories.values())
                    wandb.log({
                        "episode": self.total_episodes,
                        "episode_length": steps,
                        "mean_episode_reward": mean_reward,
                        "epsilon": current_epsilon,
                        "global_step": self.global_step,
                        "buffer_size": total_buffer_size
                    })

                # Reset episode statistics for this environment
                self._episode_rewards[env_idx].fill(0)
                self._episode_steps[env_idx] = 0
                env_terminations[env_idx, :] = False

                # Save models periodically
                if self.total_episodes % self.args.save_frequency == 0:
                    self.save_models(f"episode_{self.total_episodes}")

            # Update epsilon based on training progress
            if len(done_envs) > 0:
                progress = self.total_episodes / self.args.num_episodes
                for agent in self.agents.values():
                    agent.set_epsilon_by_progress(progress)

            self.global_step += 1

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
    parser.add_argument("--e-move", type=int, default=1, help="Energy cost per move")
    parser.add_argument("--e-charge", type=int, default=5, help="Energy gain per charge")
    parser.add_argument("--e-collision", type=int, default=3, help="Energy loss per collision")
    parser.add_argument("--e-boundary", type=int, default=50, help="Energy loss when hitting wall")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--charger-positions", type=str, default=None,
                       help='Charger positions as "y1,x1;y2,x2;..."')

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
    parser.add_argument("--train-frequency", type=int, default=4, help="Train every N steps")

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
    parser.add_argument("--eval-steps", type=int, default=10000, help="Max steps for evaluation")

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
        if args.num_robots >= 2:
            eval_cmd.extend(["--robot-1-energy", str(args.robot_1_energy)])
        if args.num_robots >= 3:
            eval_cmd.extend(["--robot-2-energy", str(args.robot_2_energy)])
        if args.num_robots >= 4:
            eval_cmd.extend(["--robot-3-energy", str(args.robot_3_energy)])

        # Add charger positions if specified
        if args.charger_positions:
            eval_cmd.extend(["--charger-positions", args.charger_positions])

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
