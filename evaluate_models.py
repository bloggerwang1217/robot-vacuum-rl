"""
Evaluation Script for Independent DQN Models
Load trained models and visualize their behavior with pygame rendering
"""

import torch
import numpy as np
import os
import argparse
from typing import Dict, List

# Import the DQN network and agent components
from dqn import DQN, init_weights
from train_dqn import IndependentDQNAgent

# Import the Gym-wrapped environment
from gym import RobotVacuumGymEnv


class ModelEvaluator:
    """
    Evaluator for trained Independent DQN models
    Loads saved models and runs evaluation episodes with visualization
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

        # Environment with rendering
        self.env = RobotVacuumGymEnv(
            n=args.env_n,
            initial_energy=args.initial_energy,
            e_move=args.e_move,
            e_charge=args.e_charge,
            e_collision=args.e_collision,
            n_steps=args.max_episode_steps,
            render_mode="human"  # Enable pygame rendering
        )

        # Initialize agents
        self.agent_ids = ['robot_0', 'robot_1', 'robot_2', 'robot_3']
        self.agents = {}

        observation_dim = 20  # Actual observation dimension used in training
        action_dim = 5  # UP, DOWN, LEFT, RIGHT, STAY

        # Create a dummy args object for agent initialization
        dummy_args = argparse.Namespace(
            gamma=args.gamma,
            epsilon=args.eval_epsilon,  # Use low epsilon for evaluation
            batch_size=32,
            lr=0.0001,
            memory_size=10000,
            use_epsilon_decay=False,  # No epsilon decay during evaluation
            save_dir='.'  # Dummy save directory
        )

        for agent_id in self.agent_ids:
            self.agents[agent_id] = IndependentDQNAgent(
                agent_id=agent_id,
                observation_dim=observation_dim,
                action_dim=action_dim,
                device=self.device,
                args=dummy_args
            )

        # Load models
        self.load_models(args.model_dir)

    def load_models(self, model_dir: str):
        """
        Load trained model weights for all agents

        Args:
            model_dir: Directory containing saved model files
        """
        print(f"Loading models from: {model_dir}")

        for agent_id in self.agent_ids:
            model_path = os.path.join(model_dir, f"{agent_id}.pt")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.agents[agent_id].load(model_path)
            print(f"  Loaded {agent_id} from {model_path}")

        print("All models loaded successfully!")

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Run evaluation episodes with visualization

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation statistics
        """
        all_episode_stats = []

        for episode in range(num_episodes):
            print(f"\n[Evaluation Episode {episode + 1}/{num_episodes}]")

            # Reset environment
            observations, infos = self.env.reset()

            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            episode_infos_history = []
            step_count = 0
            done = False

            # Episode loop
            while not done:
                # Select actions for all agents (in eval mode)
                actions = []
                for agent_id in self.agent_ids:
                    obs = observations[agent_id]
                    action = self.agents[agent_id].select_action(obs, eval_mode=True)
                    actions.append(action)

                # Step environment
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

                # Render the environment (pygame will display automatically)
                self.env.render()

                # Store infos for analysis
                episode_infos_history.append(infos)

                # Accumulate rewards
                for agent_id in self.agent_ids:
                    episode_rewards[agent_id] += rewards[agent_id]

                # Update state
                observations = next_observations
                step_count += 1

                # Check episode termination
                if any(terminations.values()) or any(truncations.values()):
                    done = True

            # Compute episode statistics
            stats = self.compute_episode_stats(episode_rewards, episode_infos_history, step_count)
            all_episode_stats.append(stats)

            # Print episode summary
            print(f"  Steps: {stats['episode_length']}")
            print(f"  Survival: {stats['survival_count']}/4")
            print(f"  Mean Reward: {stats['mean_reward']:.2f}")
            print(f"  Total Collisions: {stats['total_collisions']}")
            print(f"  Total Kills: {stats['total_kills']}")
            print(f"  Total Charges: {stats['total_charges']}")

        # Compute overall statistics
        overall_stats = self.compute_overall_stats(all_episode_stats)

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"Average Episode Length: {overall_stats['avg_episode_length']:.2f}")
        print(f"Average Survival Rate: {overall_stats['avg_survival_rate']:.2f}/4")
        print(f"Average Mean Reward: {overall_stats['avg_mean_reward']:.2f}")
        print(f"Average Total Collisions: {overall_stats['avg_total_collisions']:.2f}")
        print(f"Average Total Kills: {overall_stats['avg_total_kills']:.2f}")
        print(f"Average Total Charges: {overall_stats['avg_total_charges']:.2f}")
        print("=" * 60)

        return overall_stats

    def compute_episode_stats(self, episode_rewards: Dict[str, float],
                             episode_infos_history: List[Dict],
                             step_count: int) -> Dict[str, float]:
        """
        Compute statistics for a single episode

        Args:
            episode_rewards: Accumulated rewards for each agent
            episode_infos_history: Infos records for the entire episode
            step_count: Number of steps in the episode

        Returns:
            Episode statistics dictionary
        """
        # Mean reward
        all_rewards = list(episode_rewards.values())
        mean_reward = np.mean(all_rewards)

        # Survival count
        final_infos = episode_infos_history[-1] if episode_infos_history else {}
        survival_count = sum(1 for agent_id in self.agent_ids
                            if not final_infos.get(agent_id, {}).get('is_dead', False))

        # Total collisions and charges
        total_collisions = sum(final_infos.get(agent_id, {}).get('total_agent_collisions', 0)
                              for agent_id in self.agent_ids)
        total_charges = sum(final_infos.get(agent_id, {}).get('total_charges', 0)
                           for agent_id in self.agent_ids)

        # Compute kills
        total_kills = self.compute_kills(episode_infos_history)

        return {
            'episode_length': step_count,
            'survival_count': survival_count,
            'mean_reward': mean_reward,
            'total_collisions': total_collisions,
            'total_kills': total_kills,
            'total_charges': total_charges
        }

    def compute_kills(self, episode_infos: List[Dict]) -> int:
        """
        Compute the number of "kills" in this episode

        Definition: Robot A collides with robot B at time t, and robot B dies within t+1 to t+5

        Args:
            episode_infos: Infos records for the entire episode

        Returns:
            Total number of kills
        """
        kills = 0

        for t, infos in enumerate(episode_infos):
            for agent_id in self.agent_ids:
                collision_target = infos[agent_id].get('collided_with_agent_id', None)

                if collision_target is not None:
                    # Check if collision target dies within next 5 steps
                    for future_t in range(t + 1, min(t + 6, len(episode_infos))):
                        future_info = episode_infos[future_t][collision_target]
                        if future_info.get('is_dead', False):
                            kills += 1
                            break  # Count only once

        return kills

    def compute_overall_stats(self, all_episode_stats: List[Dict]) -> Dict[str, float]:
        """
        Compute overall statistics across all episodes

        Args:
            all_episode_stats: List of episode statistics

        Returns:
            Overall statistics dictionary
        """
        avg_episode_length = np.mean([s['episode_length'] for s in all_episode_stats])
        avg_survival_rate = np.mean([s['survival_count'] for s in all_episode_stats])
        avg_mean_reward = np.mean([s['mean_reward'] for s in all_episode_stats])
        avg_total_collisions = np.mean([s['total_collisions'] for s in all_episode_stats])
        avg_total_kills = np.mean([s['total_kills'] for s in all_episode_stats])
        avg_total_charges = np.mean([s['total_charges'] for s in all_episode_stats])

        return {
            'avg_episode_length': avg_episode_length,
            'avg_survival_rate': avg_survival_rate,
            'avg_mean_reward': avg_mean_reward,
            'avg_total_collisions': avg_total_collisions,
            'avg_total_kills': avg_total_kills,
            'avg_total_charges': avg_total_charges
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Independent DQN models")

    # Model loading
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing saved model files (robot_0.pt, robot_1.pt, etc.)")

    # Environment parameters (should match training config)
    parser.add_argument("--env-n", type=int, default=3, help="Environment grid size (n×n)")
    parser.add_argument("--initial-energy", type=int, default=100, help="Initial energy for robots")
    parser.add_argument("--e-move", type=int, default=1, help="Energy cost per move")
    parser.add_argument("--e-charge", type=int, default=5, help="Energy gain per charge")
    parser.add_argument("--e-collision", type=int, default=3, help="Energy loss per collision")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Maximum steps per episode")

    # Evaluation settings
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--eval-epsilon", type=float, default=0.05,
                       help="Epsilon for evaluation (low value to show learned policy)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (should match training)")

    args = parser.parse_args()

    # Run evaluation
    evaluator = ModelEvaluator(args)
    evaluator.evaluate(num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
