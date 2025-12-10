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

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Replay buffer (simple deque)
        self.memory = deque(maxlen=args.memory_size)

        # Counters
        self.train_count = 0

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

        state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = q_values.argmax().item()

        if return_q_values:
            return action, q_values.squeeze(0).cpu().numpy()
        else:
            return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience to replay buffer

        Standard 1-step DQN transition
        """
        self.memory.append((state, action, reward, next_state, done))

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
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (Standard DQN)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

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

        # Prepare individual robot energies
        robot_energies = [
            args.robot_0_energy if args.robot_0_energy is not None else args.initial_energy,
            args.robot_1_energy if args.robot_1_energy is not None else args.initial_energy,
            args.robot_2_energy if args.robot_2_energy is not None else args.initial_energy,
            args.robot_3_energy if args.robot_3_energy is not None else args.initial_energy,
        ]

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
            initial_energy=args.initial_energy,
            robot_energies=robot_energies,
            e_move=args.e_move,
            e_charge=args.e_charge,
            e_collision=args.e_collision,
            n_steps=args.max_episode_steps,
            charger_positions=charger_positions
        )

        # Initialize agents
        self.agent_ids = ['robot_0', 'robot_1', 'robot_2', 'robot_3']
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

    def train(self):
        """Main training loop"""
        for episode in range(self.args.num_episodes):
            self.episode_count = episode

            # Reset environment
            observations, infos = self.env.reset()

            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            episode_infos_history = []
            step_count = 0
            done = False

            # Episode loop
            while not done:
                # Select actions for all agents
                actions = []
                for agent_id in self.agent_ids:
                    obs = observations[agent_id]
                    action = self.agents[agent_id].select_action(obs)
                    actions.append(action)

                # Step environment
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

                # Store infos for kill analysis
                episode_infos_history.append(infos)

                # Store transitions and train each agent
                for i, agent_id in enumerate(self.agent_ids):
                    obs = observations[agent_id]
                    next_obs = next_observations[agent_id]
                    reward = rewards[agent_id]
                    terminated = terminations[agent_id]

                    # Remember transition
                    self.agents[agent_id].remember(obs, actions[i], reward, next_obs, terminated)

                    # Train
                    train_stats = self.agents[agent_id].train_step(self.args.replay_start_size)

                    # Log training stats
                    if train_stats and self.global_step % 1000 == 0:
                        wandb.log({
                            f"{agent_id}/loss": train_stats['loss'],
                            f"{agent_id}/q_mean": train_stats['q_mean'],
                            f"{agent_id}/q_std": train_stats['q_std'],
                            "global_step": self.global_step
                        })

                    episode_rewards[agent_id] += reward

                # Update target networks
                if self.global_step % self.args.target_update_frequency == 0:
                    for agent in self.agents.values():
                        agent.update_target_network()

                # Update state
                observations = next_observations
                self.global_step += 1
                step_count += 1

                # Check episode termination
                # End Episode if：
                # 1. Only one robot survives
                # 2. All robots are dead
                # 3. Reach max_steps (truncation)
                alive_count = sum(1 for agent_id in self.agent_ids if not terminations.get(agent_id, False))

                if alive_count <= 1 or any(truncations.values()):
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
        print(f"[Episode {episode}] Steps: {step_count} | Survival: {survival_count}/4 | "
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
    parser.add_argument("--initial-energy", type=int, default=100, help="Initial energy for all robots (used if individual energies not specified)")
    parser.add_argument("--robot-0-energy", type=int, default=None, help="Initial energy for robot 0")
    parser.add_argument("--robot-1-energy", type=int, default=None, help="Initial energy for robot 1")
    parser.add_argument("--robot-2-energy", type=int, default=None, help="Initial energy for robot 2")
    parser.add_argument("--robot-3-energy", type=int, default=None, help="Initial energy for robot 3")
    parser.add_argument("--e-move", type=int, default=1, help="Energy cost per move")
    parser.add_argument("--e-charge", type=int, default=5, help="Energy gain per charge")
    parser.add_argument("--e-collision", type=int, default=3, help="Energy loss per collision (互撞或被推人時的傷害)")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--charger-positions", type=str, default=None,
                       help='Charger positions as "y1,x1;y2,x2;..." (e.g., "0,0;0,2;2,0;2,2"). Use -1,-1 to disable a charger. Default: four corners')

    # DQN hyperparameters
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

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
