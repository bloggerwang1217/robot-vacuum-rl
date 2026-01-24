"""
Independent DQN Training Script for Multi-Robot Energy Survival Environment
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

# Import the Gym-wrapped environment (assumed to be implemented by others)
from gym import RobotVacuumGymEnv


class IndependentDQNAgent:
    """
    Independent DQN Agent for Multi-Agent Learning
    Each robot has its own independent DQN model
    """
    def __init__(self,
                 agent_id: str,
                 observation_dim: int,
                 action_dim: int,
                 device: torch.device,
                 args: argparse.Namespace):
        """
        Initialize Independent DQN Agent

        Args:
            agent_id: Unique identifier for the agent (e.g., 'robot_0')
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            device: PyTorch device (cpu/cuda/mps)
            args: Training hyperparameters
        """
        self.agent_id = agent_id
        self.device = device
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # DQN hyperparameters
        self.gamma = args.gamma
        self.batch_size = args.batch_size

        # Epsilon configuration
        self.use_epsilon_decay = args.use_epsilon_decay
        if self.use_epsilon_decay:
            self.epsilon = args.epsilon_start
            self.epsilon_start = args.epsilon_start
            self.epsilon_end = args.epsilon_end
            self.epsilon_decay = args.epsilon_decay
        else:
            self.epsilon = args.epsilon  # Fixed epsilon (no decay)

        # Eval epsilon (for evaluation mode, default 0)
        self.eval_epsilon = getattr(args, 'eval_epsilon', 0.0)

        # Network architecture
        self.q_net = DQN(action_dim, observation_dim).to(device)
        self.target_net = DQN(action_dim, observation_dim).to(device)
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Compile models for faster execution (PyTorch 2.0+)
        # 只在 CUDA 上編譯，避免 CPU/MPS 的相容性問題
        # 注意：對小網路，torch.compile 的 JIT overhead 可能得不償失
        use_compile = getattr(args, 'use_torch_compile', False)
        if use_compile and device.type == 'cuda' and hasattr(torch, 'compile'):
            self.q_net = torch.compile(self.q_net, mode='default')
            self.target_net = torch.compile(self.target_net, mode='default')

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Mixed Precision Training (AMP) - 只在 CUDA 上使用
        self.use_amp = device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Replay buffer (simple deque)
        self.memory = deque(maxlen=args.memory_size)

        # N-step learning
        self.n_step = getattr(args, 'n_step', 1)  # 預設 1-step（原本行為）
        self.n_step_buffer = deque(maxlen=self.n_step) if self.n_step > 1 else None

        # Counters
        self.train_count = 0

        # Pre-allocated tensors for faster inference (避免重複建立 tensor)
        self._obs_buffer = torch.zeros(1, observation_dim, dtype=torch.float32, device=device)
        self._batch_states_buffer = None  # 在 train_step 時動態分配

    def select_action(self, observation: np.ndarray, eval_mode: bool = False, return_q_values: bool = False):
        """
        Select action using epsilon-greedy strategy

        Args:
            observation: Current observation
            eval_mode: Whether in evaluation mode (uses eval_epsilon)
            return_q_values: If True, also return Q-values for all actions

        Returns:
            action (int) if return_q_values=False
            (action, q_values) if return_q_values=True
        """
        epsilon = self.eval_epsilon if eval_mode else self.epsilon

        # 先決定是否 random（避免不必要的 GPU 計算）
        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
            if return_q_values:
                # 仍需計算 Q-values
                self._obs_buffer.copy_(torch.from_numpy(observation))
                with torch.no_grad():
                    q_values = self.q_net(self._obs_buffer)
                return action, q_values.squeeze(0).cpu().numpy()
            return action

        # 使用 pre-allocated buffer 避免重複建立 tensor
        self._obs_buffer.copy_(torch.from_numpy(observation))
        with torch.no_grad():
            q_values = self.q_net(self._obs_buffer)

        action = q_values.argmax().item()

        if return_q_values:
            return action, q_values.squeeze(0).cpu().numpy()
        else:
            return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience to replay buffer

        支援 n-step learning：
        - n_step=1: 標準 1-step DQN（直接存入）
        - n_step>1: 累積 n 步後計算 n-step return 再存入

        Replay buffer 格式: (state, action, reward, next_state, done, actual_n_step)
        - actual_n_step: 實際累積的步數（用於計算正確的 γ^k）

        Episode 結束時的特殊處理：
        - 將 buffer 中所有剩餘的 transition 都存入
        - 確保直接導致死亡的動作能被學習到
        """
        if self.n_step == 1:
            # 原本的 1-step 行為，actual_n_step = 1
            self.memory.append((state, action, reward, next_state, done, 1))
        else:
            # N-step learning
            self.n_step_buffer.append((state, action, reward, next_state, done))

            if done:
                # Episode 結束：將 buffer 中所有剩餘的 transition 都存入
                # 這確保直接導致死亡的動作能被學習到（1-step transition with full penalty）
                buffer_list = list(self.n_step_buffer)
                _, _, _, end_next_state, end_done = buffer_list[-1]

                for start_idx in range(len(buffer_list)):
                    # 從 start_idx 開始計算到結尾的 n-step return
                    n_step_return = 0
                    for offset, (_, _, r, _, _) in enumerate(buffer_list[start_idx:]):
                        n_step_return += (self.gamma ** offset) * r

                    # 取該位置的 state, action
                    start_state, start_action, _, _, _ = buffer_list[start_idx]

                    # 實際步數 = 從 start_idx 到結尾的長度
                    actual_n_step = len(buffer_list) - start_idx

                    # 存入 replay buffer
                    self.memory.append((start_state, start_action, n_step_return, end_next_state, end_done, actual_n_step))

                self.n_step_buffer.clear()

            elif len(self.n_step_buffer) == self.n_step:
                # 正常情況：累積滿 n 步
                # 計算 n-step return（累積折現獎勵）
                n_step_return = 0
                for idx, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                    n_step_return += (self.gamma ** idx) * r

                # 取第一步的 state, action
                start_state, start_action, _, _, _ = self.n_step_buffer[0]

                # 取最後一步的 next_state, done
                _, _, _, end_next_state, end_done = self.n_step_buffer[-1]

                # 存入 replay buffer（包含實際步數）
                self.memory.append((start_state, start_action, n_step_return, end_next_state, end_done, self.n_step))

                # 滑動窗口：移除最舊的一步
                self.n_step_buffer.popleft()

    def train_step(self, replay_start_size: int) -> Dict[str, float]:
        """
        Execute one training step

        Args:
            replay_start_size: Minimum buffer size before training starts

        Returns:
            Training statistics
        """
        if len(self.memory) < replay_start_size:
            return {}

        self.train_count += 1

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, actual_n_steps = zip(*batch)

        # 使用 numpy 預先轉換，減少 Python 層面的開銷
        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        actual_n_steps_np = np.array(actual_n_steps, dtype=np.float32)

        # 一次性傳輸到 GPU（使用 non_blocking=True 允許異步傳輸）
        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
        actual_n_steps_t = torch.from_numpy(actual_n_steps_np).to(self.device, non_blocking=True)

        # Mixed Precision Training - 前向傳播使用 autocast
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                # Compute Q-values
                q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q-values (Double DQN to lower overestimation)
                # N-step learning: 用 γ^k 來折現（k = 實際累積步數，可能 < n_step）
                # reward 已經在 remember() 中累積為 k-step return
                with torch.no_grad():
                    # 計算 per-sample 的 gamma^k
                    gamma_n = self.gamma ** actual_n_steps_t
                    # Select action: use q_net to obtain best action
                    next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                    # Evaluate value: use target_net to evaluate the Q-value
                    next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                    target_q_values = rewards + gamma_n * next_q_values * (1 - dones)

                # Compute loss (MSE)
                loss = nn.MSELoss()(q_values, target_q_values)

            # Optimize with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Original training path (CPU/MPS)
            # Compute Q-values
            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values (Double DQN to lower overestimation)
            # N-step learning: 用 γ^k 來折現（k = 實際累積步數，可能 < n_step）
            # reward 已經在 remember() 中累積為 k-step return
            with torch.no_grad():
                # 計算 per-sample 的 gamma^k
                gamma_n = self.gamma ** actual_n_steps_t
                # Select action: use q_net to obtain best action
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                # Evaluate value: use target_net to evaluate the Q-value
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                target_q_values = rewards + gamma_n * next_q_values * (1 - dones)

            # Compute loss (MSE)
            loss = nn.MSELoss()(q_values, target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_std': q_values.std().item()
        }

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """
        Decay epsilon using exponential decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        """
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save model weights"""
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        """Load model weights"""
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.q_net.state_dict())


class MultiAgentTrainer:
    """
    Multi-Agent Independent DQN Trainer
    Manages training of 4 independent DQN agents
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Random seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Number of robots
        self.num_robots = args.num_robots

        # Prepare individual robot energies (only for the robots that exist)
        all_robot_energies = [
            args.robot_0_energy if args.robot_0_energy is not None else args.initial_energy,
            args.robot_1_energy if args.robot_1_energy is not None else args.initial_energy,
            args.robot_2_energy if args.robot_2_energy is not None else args.initial_energy,
            args.robot_3_energy if args.robot_3_energy is not None else args.initial_energy,
        ]
        robot_energies = all_robot_energies[:self.num_robots]

        # Parse charger positions if provided
        charger_positions = None
        if args.charger_positions is not None:
            try:
                # Parse format: "y1,x1;y2,x2;y3,x3;y4,x4"
                charger_positions = []
                for pos_str in args.charger_positions.split(';'):
                    y, x = map(int, pos_str.split(','))
                    charger_positions.append((y, x))
                print(f"Using custom charger positions: {charger_positions}")
            except Exception as e:
                print(f"Error parsing charger positions: {e}")
                print("Using default (four corners)")
                charger_positions = None

        # Environment
        self.env = RobotVacuumGymEnv(
            n=args.env_n,
            num_robots=self.num_robots,
            initial_energy=args.initial_energy,
            robot_energies=robot_energies,
            e_move=args.e_move,
            e_charge=args.e_charge,
            e_collision=args.e_collision,
            e_boundary=args.e_boundary,
            n_steps=args.max_episode_steps,
            charger_positions=charger_positions
        )

        # Initialize agents (only for the robots that exist)
        self.agent_ids = [f'robot_{i}' for i in range(self.num_robots)]
        self.agents = {}

        observation_dim = self.env.observation_space.shape[0]  # Get actual observation dim from env
        action_dim = 5  # UP, DOWN, LEFT, RIGHT, STAY

        for agent_id in self.agent_ids:
            self.agents[agent_id] = IndependentDQNAgent(
                agent_id=agent_id,
                observation_dim=observation_dim,
                action_dim=action_dim,
                device=self.device,
                args=args
            )

        # Training counters
        self.global_step = 0
        self.episode_count = 0

        # Cumulative death counter for each robot
        self.cumulative_deaths = {agent_id: 0 for agent_id in self.agent_ids}

        # Model saving
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Record keeping for kill analysis
        self.episode_history = []

        # Best model tracking
        self.best_metric = float('-inf')

        # Pre-allocated buffers for batch action selection (效能優化)
        self._obs_batch_np = np.zeros((self.num_robots, observation_dim), dtype=np.float32)
        self._obs_batch_tensor = torch.zeros(self.num_robots, observation_dim, dtype=torch.float32, device=self.device)

    def compute_immediate_kills(self, episode_infos: List[Dict]) -> Tuple[int, Dict[str, int]]:
        """
        Compute the number of "immediate kills" in this episode

        Definition: Robot A collides with robot B at time t, and at time t+1:
        - Exactly one of them dies (the other survives)
        - If both die, it does not count

        Args:
            episode_infos: Infos records for the entire episode (infos dict at each step)

        Returns:
            Tuple of (total_immediate_kills, per_agent_immediate_kills_dict)
        """
        immediate_kills = 0
        per_agent_immediate_kills = {agent_id: 0 for agent_id in self.agent_ids}
        counted_pairs = set()  # Track counted collision pairs to avoid double counting

        for t, infos in enumerate(episode_infos):
            # Check if there's a next step
            if t + 1 >= len(episode_infos):
                break

            next_infos = episode_infos[t + 1]

            for agent_id in self.agent_ids:
                collision_target = infos[agent_id].get('collided_with_agent_id', None)

                if collision_target is not None:
                    # Create a canonical pair identifier (sorted to avoid duplicates)
                    pair = tuple(sorted([agent_id, collision_target]))

                    # Skip if we've already counted this collision pair at this timestep
                    if (t, pair) in counted_pairs:
                        continue

                    # Check both agents were alive at time t
                    agent_alive_at_t = not infos[agent_id].get('is_dead', False)
                    target_alive_at_t = not infos[collision_target].get('is_dead', False)

                    if agent_alive_at_t and target_alive_at_t:
                        # Check status at time t+1
                        agent_alive_at_t1 = not next_infos[agent_id].get('is_dead', False)
                        target_alive_at_t1 = not next_infos[collision_target].get('is_dead', False)

                        # Count as immediate kill if exactly one dies
                        if agent_alive_at_t1 != target_alive_at_t1:
                            immediate_kills += 1
                            counted_pairs.add((t, pair))
                            # Attribute to the survivor ONLY if they were actively moving at time t
                            if agent_alive_at_t1:
                                # agent survived, check if they were moving
                                if infos[agent_id].get('is_mover_this_step', False):
                                    per_agent_immediate_kills[agent_id] += 1
                            else:
                                # target survived, check if they were moving
                                if infos[collision_target].get('is_mover_this_step', False):
                                    per_agent_immediate_kills[collision_target] += 1

        return immediate_kills, per_agent_immediate_kills

    def select_actions_batch(self, observations: Dict[str, np.ndarray], eval_mode: bool = False) -> List[int]:
        """
        批量選擇動作 - 一次處理所有 agents，減少 GPU 傳輸次數

        保證與逐個調用 select_action 的結果完全相同（相同的隨機種子）

        Args:
            observations: 所有 agents 的觀測字典
            eval_mode: 是否為評估模式

        Returns:
            actions: 所有 agents 的動作列表
        """
        actions = [None] * self.num_robots  # 預分配，避免 append
        need_inference_indices = []  # 記錄需要推理的 agent index

        # 第一階段：決定 random actions，收集需要推理的 indices
        for i, agent_id in enumerate(self.agent_ids):
            epsilon = self.agents[agent_id].eval_epsilon if eval_mode else self.agents[agent_id].epsilon

            if random.random() < epsilon:
                actions[i] = random.randint(0, self.agents[agent_id].action_dim - 1)
            else:
                # 直接複製到 pre-allocated numpy buffer
                self._obs_batch_np[len(need_inference_indices)] = observations[agent_id]
                need_inference_indices.append(i)

        # 第二階段：批量推理
        if need_inference_indices:
            n_infer = len(need_inference_indices)
            # 一次性傳輸到 GPU（使用 pre-allocated tensor 的 slice）
            self._obs_batch_tensor[:n_infer].copy_(
                torch.from_numpy(self._obs_batch_np[:n_infer]), non_blocking=True
            )

            with torch.no_grad():
                # 每個 agent 有獨立的網路，需要分別推理
                for idx, agent_idx in enumerate(need_inference_indices):
                    agent_id = self.agent_ids[agent_idx]
                    obs_single = self._obs_batch_tensor[idx:idx+1]
                    q_values = self.agents[agent_id].q_net(obs_single)
                    actions[agent_idx] = q_values.argmax().item()

        return actions

    def train(self):
        """Main training loop (sequential actions: one robot moves at a time)"""
        for episode in range(self.args.num_episodes):
            self.episode_count = episode

            # Reset environment
            observations, infos = self.env.reset()

            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            episode_infos_history = []
            step_count = 0
            done = False

            # Track terminations across the episode
            terminations = {agent_id: False for agent_id in self.agent_ids}

            # Episode loop
            while not done:
                # Sequential actions: each robot acts one at a time
                for robot_id in range(self.num_robots):
                    agent_id = self.agent_ids[robot_id]
                    agent = self.agents[agent_id]

                    # 1. Get current observation (latest state)
                    obs = self.env.get_observation(robot_id)

                    # 2. Select action
                    action = agent.select_action(obs)

                    # 3. Execute action
                    next_obs, reward, terminated, truncated, info = self.env.step_single(robot_id, action)

                    # 4. Store transition
                    agent.remember(obs, action, reward, next_obs, terminated)

                    # 5. Train
                    train_stats = agent.train_step(self.args.replay_start_size)

                    # Log training stats
                    if train_stats and self.global_step % 1000 == 0:
                        wandb.log({
                            f"{agent_id}/loss": train_stats['loss'],
                            f"{agent_id}/q_mean": train_stats['q_mean'],
                            f"{agent_id}/q_std": train_stats['q_std'],
                            "global_step": self.global_step
                        })

                    episode_rewards[agent_id] += reward

                    # Update termination status
                    if terminated:
                        terminations[agent_id] = True

                # 6. Advance step count after all robots have acted
                max_steps_reached, truncations = self.env.advance_step()

                # Update target networks
                if self.global_step % self.args.target_update_frequency == 0:
                    for agent in self.agents.values():
                        agent.update_target_network()

                self.global_step += 1
                step_count += 1

                # Collect infos for kill analysis at the end of this step
                state = self.env.env.get_global_state()
                infos = self.env._get_infos(state)
                episode_infos_history.append(infos)

                # Check episode termination
                # 訓練時：永遠跑到 max steps 或全死
                # 這讓勝利者能學到「殺人後獨佔資源」的長期收益
                alive_count = sum(1 for agent_id in self.agent_ids if not terminations.get(agent_id, False))

                if alive_count == 0 or max_steps_reached:
                    done = True

            # Episode summary
            self.log_episode_summary(episode, episode_rewards, episode_infos_history, step_count)

            # Decay epsilon for all agents
            for agent in self.agents.values():
                agent.decay_epsilon()

            # Periodic model saving
            if (episode + 1) % self.args.save_frequency == 0:
                self.save_models(f"episode_{episode + 1}")

            # Key model checkpointing (based on interesting metrics)
            self.check_and_save_key_models(episode, episode_infos_history)

    def log_episode_summary(self, episode: int, episode_rewards: Dict[str, float],
                           episode_infos_history: List[Dict], step_count: int):
        """
        Log episode statistics to wandb

        Metric design based on PLAN.md section 4.2
        """
        # Compute summary metrics
        all_rewards = list(episode_rewards.values())
        mean_episode_reward = np.mean(all_rewards)
        std_episode_reward = np.std(all_rewards)

        # Get survival rate and final energy from last step's infos
        final_infos = episode_infos_history[-1] if episode_infos_history else {}
        survival_count = sum(1 for agent_id in self.agent_ids
                            if not final_infos.get(agent_id, {}).get('is_dead', False))

        # Get per-agent final energies and positions
        per_agent_energies = {}
        per_agent_positions = {}
        for agent_id in self.agent_ids:
            per_agent_energies[agent_id] = final_infos.get(agent_id, {}).get('energy', 0)
            per_agent_positions[agent_id] = final_infos.get(agent_id, {}).get('position', (0, 0))
        
        final_energies = list(per_agent_energies.values())
        mean_final_energy = np.mean(final_energies)

        # Compute total and per-agent metrics
        # 1. Collisions
        per_agent_collisions = {}
        total_agent_collisions = 0
        for agent_id in self.agent_ids:
            collisions = final_infos.get(agent_id, {}).get('total_agent_collisions', 0)
            per_agent_collisions[agent_id] = collisions
            total_agent_collisions += collisions

        # 2. Charges
        per_agent_charges = {}
        total_charges = 0
        for agent_id in self.agent_ids:
            charges = final_infos.get(agent_id, {}).get('total_charges', 0)
            per_agent_charges[agent_id] = charges
            total_charges += charges
        
        # 3. Non-home charges
        per_agent_non_home_charges = {}
        total_non_home_charges = 0
        for agent_id in self.agent_ids:
            non_home_charges = final_infos.get(agent_id, {}).get('total_non_home_charges', 0)
            per_agent_non_home_charges[agent_id] = non_home_charges
            total_non_home_charges += non_home_charges

        # 4. Kills
        # total_kills, per_agent_kills = self.compute_kills(episode_infos_history) # Removed as per user request
        
        # 5. Immediate kills
        total_immediate_kills, per_agent_immediate_kills = self.compute_immediate_kills(episode_infos_history)

        # 6. Update cumulative deaths and get per-episode deaths
        per_agent_deaths = {}
        for agent_id in self.agent_ids:
            is_dead = final_infos.get(agent_id, {}).get('is_dead', False)
            if is_dead:
                self.cumulative_deaths[agent_id] += 1
            per_agent_deaths[agent_id] = 1 if is_dead else 0

        # Get current epsilon from first agent (all agents have same epsilon)
        current_epsilon = self.agents['robot_0'].epsilon

        # Prepare wandb log dict
        log_dict = {
            "episode": episode,
            "episode_length": step_count,
            "survival_rate": survival_count,
            "mean_episode_reward": mean_episode_reward,
            "std_episode_reward": std_episode_reward,
            "total_agent_collisions_per_episode": total_agent_collisions,
            "total_charges_per_episode": total_charges,
            "total_non_home_charges_per_episode": total_non_home_charges,
            "total_immediate_kills_per_episode": total_immediate_kills,
            "mean_final_energy": mean_final_energy,
            "epsilon": current_epsilon,
            "global_step": self.global_step
        }

        # Add per-agent metrics to wandb
        for agent_id in self.agent_ids:
            log_dict[f"{agent_id}/reward_per_episode"] = episode_rewards[agent_id]
            log_dict[f"{agent_id}/collisions_per_episode"] = per_agent_collisions[agent_id]
            log_dict[f"{agent_id}/charges_per_episode"] = per_agent_charges[agent_id]
            log_dict[f"{agent_id}/non_home_charges_per_episode"] = per_agent_non_home_charges[agent_id]
            log_dict[f"{agent_id}/immediate_kills_per_episode"] = per_agent_immediate_kills[agent_id]
            log_dict[f"{agent_id}/deaths_per_episode"] = per_agent_deaths[agent_id]
            log_dict[f"{agent_id}/cumulative_deaths"] = self.cumulative_deaths[agent_id]
            log_dict[f"{agent_id}/final_energy"] = per_agent_energies[agent_id]
            log_dict[f"{agent_id}/final_position_x"] = per_agent_positions[agent_id][0]
            log_dict[f"{agent_id}/final_position_y"] = per_agent_positions[agent_id][1]

            # Add collided_by metrics
            log_dict[f"{agent_id}/collided_by_robot_0"] = final_infos.get(agent_id, {}).get('collided_by_robot_0', 0)
            log_dict[f"{agent_id}/collided_by_robot_1"] = final_infos.get(agent_id, {}).get('collided_by_robot_1', 0)
            log_dict[f"{agent_id}/collided_by_robot_2"] = final_infos.get(agent_id, {}).get('collided_by_robot_2', 0)
            log_dict[f"{agent_id}/collided_by_robot_3"] = final_infos.get(agent_id, {}).get('collided_by_robot_3', 0)
        
        # Log to wandb
        wandb.log(log_dict)

        # Print summary with totals
        print(f"[Episode {episode}] Steps: {step_count} | Survival: {survival_count}/{len(self.agent_ids)} | "
              f"Mean Reward: {mean_episode_reward:.2f} | Collisions: {total_agent_collisions} | "
              f"Immediate Kills: {total_immediate_kills} | "
              f"Non-Home Charges: {total_non_home_charges}")
        
        # Print per-agent breakdown
        print(f"  Per-Agent Metrics:")
        for agent_id in self.agent_ids:
            death_marker = " 💀" if per_agent_deaths[agent_id] == 1 else ""
            
            # Get collided_by info
            collided_by_0 = final_infos.get(agent_id, {}).get('collided_by_robot_0', 0)
            collided_by_1 = final_infos.get(agent_id, {}).get('collided_by_robot_1', 0)
            collided_by_2 = final_infos.get(agent_id, {}).get('collided_by_robot_2', 0)
            collided_by_3 = final_infos.get(agent_id, {}).get('collided_by_robot_3', 0)
            
            pos = per_agent_positions[agent_id]
            print(f"    {agent_id}{death_marker}: Pos=({pos[0]},{pos[1]}), Energy={per_agent_energies[agent_id]}, "
                  f"Collisions={per_agent_collisions[agent_id]}, "
                  f"Charges={per_agent_charges[agent_id]}, "
                  f"NonHomeCharges={per_agent_non_home_charges[agent_id]}, "
                  f"ImmediateKills={per_agent_immediate_kills[agent_id]}, "
                  f"CumulativeDeaths={self.cumulative_deaths[agent_id]}")
            print(f"      CollidedBy: [0→{collided_by_0}, 1→{collided_by_1}, 2→{collided_by_2}, 3→{collided_by_3}]")

    def check_and_save_key_models(self, episode: int, episode_infos_history: List[Dict]):
        """
        Check if key metrics are reached and save models

        Design based on PLAN.md section 4.3
        """
        # Use immediate kills as the key metric for this episode
        total_immediate_kills, _ = self.compute_immediate_kills(episode_infos_history)

        # Save model if new record is achieved
        if total_immediate_kills > self.best_metric:
            self.best_metric = total_immediate_kills
            save_path = os.path.join(self.save_dir, f"key_models_ep{episode}_kills{total_immediate_kills}")
            self.save_models(save_path)
            print(f"[Key Model Saved] Episode {episode}: New record of {total_immediate_kills} immediate kills!")

    def save_models(self, subfolder: str):
        """
        Save all agents' models

        Args:
            subfolder: Subfolder name for saving
        """
        save_path = os.path.join(self.save_dir, subfolder)
        os.makedirs(save_path, exist_ok=True)

        for agent_id, agent in self.agents.items():
            model_path = os.path.join(save_path, f"{agent_id}.pt")
            agent.save(model_path)

        print(f"Models saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Independent DQN Training for Multi-Robot Environment")

    # Environment parameters
    parser.add_argument("--env-n", type=int, default=3, help="Environment grid size (n×n)")
    parser.add_argument("--num-robots", type=int, default=4, help="Number of robots (1-4)")
    parser.add_argument("--initial-energy", type=int, default=100, help="Initial energy for all robots (used if individual energies not specified)")
    parser.add_argument("--robot-0-energy", type=int, default=None, help="Initial energy for robot 0")
    parser.add_argument("--robot-1-energy", type=int, default=None, help="Initial energy for robot 1")
    parser.add_argument("--robot-2-energy", type=int, default=None, help="Initial energy for robot 2")
    parser.add_argument("--robot-3-energy", type=int, default=None, help="Initial energy for robot 3")
    parser.add_argument("--e-move", type=int, default=1, help="Energy cost per move")
    parser.add_argument("--e-charge", type=int, default=5, help="Energy gain per charge")
    parser.add_argument("--e-collision", type=int, default=3, help="Energy loss per collision (互撞或被推人時的傷害)")
    parser.add_argument("--e-boundary", type=int, default=50, help="Energy loss when hitting wall/boundary (撞牆懲罰)")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--charger-positions", type=str, default=None,
                       help='Charger positions as "y1,x1;y2,x2;..." (e.g., "0,0;0,2;2,0;2,2"). Use -1,-1 to disable a charger. Default: four corners')

    # DQN hyperparameters
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n-step", type=int, default=1,
                       help="N-step return (1=標準 DQN, 2-5=n-step DQN，看更遠的未來)")

    # Epsilon configuration
    parser.add_argument("--use-epsilon-decay", action="store_true",
                       help="Use epsilon decay instead of fixed epsilon")
    parser.add_argument("--epsilon", type=float, default=0.2,
                       help="Fixed epsilon for exploration (used when --use-epsilon-decay is not set)")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                       help="Starting epsilon for decay (used when --use-epsilon-decay is set)")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                       help="Minimum epsilon for decay (used when --use-epsilon-decay is set)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                       help="Epsilon decay rate per episode (used when --use-epsilon-decay is set)")

    parser.add_argument("--memory-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--replay-start-size", type=int, default=1000, help="Minimum replay buffer size before training")
    parser.add_argument("--target-update-frequency", type=int, default=1000, help="Target network update frequency")

    # Training settings
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--save-frequency", type=int, default=1000, help="Model save frequency (episodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-torch-compile", action="store_true",
                       help="Enable torch.compile (may slow down small networks due to JIT overhead)")

    # Wandb and logging
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity (username or team name)")
    parser.add_argument("--wandb-project", type=str, default="multi-robot-idqn", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default="idqn-base", help="Wandb run name")
    parser.add_argument("--wandb-mode", type=str, default="offline", help="Wandb mode (online/offline/disabled)")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save models")

    args = parser.parse_args()

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
    trainer = MultiAgentTrainer(args)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
