"""
Evaluation Script for Independent DQN Models
Load trained models and run long-term simulation to observe emergent behaviors
"""

import torch
import numpy as np
import random
import os
import argparse
import wandb
from typing import Dict, List, Tuple

# Import the DQN network and agent components
from dqn import DQN, init_weights
from train_dqn import IndependentDQNAgent

# Import the Gym-wrapped environment
from gym import RobotVacuumGymEnv


class ModelEvaluator:
    """
    Evaluator for trained Independent DQN models
    Loads saved models and runs long-term simulation
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Set seed for reproducibility
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
            print(f"Random seed set to: {args.seed}")

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Prepare individual robot energies
        robot_energies = [
            args.robot_0_energy if args.robot_0_energy is not None else args.initial_energy,
            args.robot_1_energy if args.robot_1_energy is not None else args.initial_energy,
            args.robot_2_energy if args.robot_2_energy is not None else args.initial_energy,
            args.robot_3_energy if args.robot_3_energy is not None else args.initial_energy,
        ]
        self.robot_energies = robot_energies

        # Environment setup (render_mode based on args)
        render_mode = "human" if args.render else None
        self.env = RobotVacuumGymEnv(
            n=args.env_n,
            initial_energy=args.initial_energy,
            robot_energies=robot_energies,
            e_move=args.e_move,
            e_charge=args.e_charge,
            e_collision=args.e_collision,
            e_collision_active_one_sided=args.e_collision_active_one_sided,
            e_collision_active_two_sided=args.e_collision_active_two_sided,
            e_collision_passive=args.e_collision_passive,
            n_steps=args.max_steps,  # Use max_steps for single long episode
            render_mode=render_mode
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
            batch_size=128,
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

    def evaluate(self) -> Dict[str, float]:
        """
        Run single long-term evaluation episode

        Returns:
            Evaluation statistics
        """
        print(f"\n{'=' * 60}")
        print(f"Starting Long-Term Simulation")
        print(f"Max Steps: {self.args.max_steps}")
        print(f"Robot Energies: {self.robot_energies}")
        print(f"Eval Epsilon: {self.args.eval_epsilon}")
        print(f"{'=' * 60}\n")

        # Reset environment
        observations, infos = self.env.reset()

        episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        episode_infos_history = []
        step_count = 0
        done = False

        # Main simulation loop
        while not done:
            # Select actions for all agents (in eval mode)
            actions = []
            for agent_id in self.agent_ids:
                obs = observations[agent_id]
                action = self.agents[agent_id].select_action(obs, eval_mode=True)
                actions.append(action)

            # Step environment
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

            # Render if enabled
            if self.args.render:
                self.env.render()

            # Store infos for analysis
            episode_infos_history.append(infos)

            # Accumulate rewards
            for agent_id in self.agent_ids:
                episode_rewards[agent_id] += rewards[agent_id]

            # Update state
            observations = next_observations
            step_count += 1

            # Log EVERY step for complete dynamics tracking
            self.log_step_summary(step_count, episode_rewards, episode_infos_history)

            # Count surviving agents
            survival_count = sum(1 for agent_id in self.agent_ids
                                if not infos.get(agent_id, {}).get('is_dead', False))

            # Check episode termination conditions:
            # 1. All dead (0 surviving)
            # 2. Only 1 survivor left (no more meaningful interaction)
            # 3. Max steps reached
            max_steps_reached = any(truncations.values())
            
            if survival_count <= 1 or max_steps_reached:
                done = True
                if survival_count == 0:
                    print(f"\n[Step {step_count}] All agents have died. Simulation ended.")
                elif survival_count == 1:
                    # Find the survivor
                    survivor = [agent_id for agent_id in self.agent_ids
                               if not infos.get(agent_id, {}).get('is_dead', False)][0]
                    print(f"\n[Step {step_count}] Only {survivor} survives. Episode complete.")
                else:
                    print(f"\n[Step {step_count}] Max steps reached. Simulation ended.")

        # Final summary
        final_stats = self.log_final_summary(step_count, episode_rewards, episode_infos_history)
        
        return final_stats

    def log_step_summary(self, step: int, episode_rewards: Dict[str, float],
                        episode_infos_history: List[Dict]):
        """
        Log periodic statistics during simulation (every log_interval steps)
        """
        # Get current infos
        current_infos = episode_infos_history[-1] if episode_infos_history else {}
        
        # Survival count
        survival_count = sum(1 for agent_id in self.agent_ids
                            if not current_infos.get(agent_id, {}).get('is_dead', False))
        
        # Current rewards
        all_rewards = list(episode_rewards.values())
        mean_reward = np.mean(all_rewards)
        
        # Total collisions, charges, kills up to now
        total_collisions = sum(current_infos.get(agent_id, {}).get('total_agent_collisions', 0)
                              for agent_id in self.agent_ids)
        total_active_collisions = sum(current_infos.get(agent_id, {}).get('total_active_collisions', 0)
                                      for agent_id in self.agent_ids)
        total_passive_collisions = sum(current_infos.get(agent_id, {}).get('total_passive_collisions', 0)
                                       for agent_id in self.agent_ids)
        total_charges = sum(current_infos.get(agent_id, {}).get('total_charges', 0)
                           for agent_id in self.agent_ids)
        total_non_home_charges = sum(current_infos.get(agent_id, {}).get('total_non_home_charges', 0)
                                     for agent_id in self.agent_ids)
        
        # Compute kills up to now
        # total_kills, per_agent_kills = self.compute_kills(episode_infos_history)
        total_immediate_kills, per_agent_immediate_kills = self.compute_immediate_kills(episode_infos_history)
        
        # Get per-agent info
        per_agent_energies = {}
        per_agent_positions = {}
        per_agent_collisions = {}
        per_agent_active_collisions = {}
        per_agent_passive_collisions = {}
        per_agent_charges = {}
        per_agent_non_home_charges = {}

        for agent_id in self.agent_ids:
            info = current_infos.get(agent_id, {})
            per_agent_energies[agent_id] = info.get('energy', 0)
            per_agent_positions[agent_id] = info.get('position', (0, 0))
            per_agent_collisions[agent_id] = info.get('total_agent_collisions', 0)
            per_agent_active_collisions[agent_id] = info.get('total_active_collisions', 0)
            per_agent_passive_collisions[agent_id] = info.get('total_passive_collisions', 0)
            per_agent_charges[agent_id] = info.get('total_charges', 0)
            per_agent_non_home_charges[agent_id] = info.get('total_non_home_charges', 0)

        # Prepare wandb log dict
        log_dict = {
            "step": step,
            "survival_rate": survival_count,
            "mean_cumulative_reward": mean_reward,
            "total_agent_collisions": total_collisions,
            "total_active_collisions": total_active_collisions,
            "total_passive_collisions": total_passive_collisions,
            "total_charges": total_charges,
            "total_non_home_charges": total_non_home_charges,
            "total_immediate_kills": total_immediate_kills,
        }

        # Add per-agent metrics to wandb
        for agent_id in self.agent_ids:
            log_dict[f"{agent_id}/energy"] = per_agent_energies[agent_id]
            log_dict[f"{agent_id}/position_x"] = per_agent_positions[agent_id][0]
            log_dict[f"{agent_id}/position_y"] = per_agent_positions[agent_id][1]
            log_dict[f"{agent_id}/collisions"] = per_agent_collisions[agent_id]
            log_dict[f"{agent_id}/active_collisions"] = per_agent_active_collisions[agent_id]
            log_dict[f"{agent_id}/passive_collisions"] = per_agent_passive_collisions[agent_id]
            log_dict[f"{agent_id}/charges"] = per_agent_charges[agent_id]
            log_dict[f"{agent_id}/non_home_charges"] = per_agent_non_home_charges[agent_id]
            log_dict[f"{agent_id}/immediate_kills"] = per_agent_immediate_kills[agent_id]
            log_dict[f"{agent_id}/cumulative_reward"] = episode_rewards[agent_id]
            
            # Is dead
            is_dead = current_infos.get(agent_id, {}).get('is_dead', False)
            log_dict[f"{agent_id}/is_dead"] = 1 if is_dead else 0
            
            # Active collisions with each opponent
            log_dict[f"{agent_id}/active_collisions_with_0"] = current_infos.get(agent_id, {}).get('active_collisions_with_0', 0)
            log_dict[f"{agent_id}/active_collisions_with_1"] = current_infos.get(agent_id, {}).get('active_collisions_with_1', 0)
            log_dict[f"{agent_id}/active_collisions_with_2"] = current_infos.get(agent_id, {}).get('active_collisions_with_2', 0)
            log_dict[f"{agent_id}/active_collisions_with_3"] = current_infos.get(agent_id, {}).get('active_collisions_with_3', 0)
            
            # Passive collisions with each opponent (collided_by_robot_X)
            log_dict[f"{agent_id}/collided_by_robot_0"] = current_infos.get(agent_id, {}).get('collided_by_robot_0', 0)
            log_dict[f"{agent_id}/collided_by_robot_1"] = current_infos.get(agent_id, {}).get('collided_by_robot_1', 0)
            log_dict[f"{agent_id}/collided_by_robot_2"] = current_infos.get(agent_id, {}).get('collided_by_robot_2', 0)
            log_dict[f"{agent_id}/collided_by_robot_3"] = current_infos.get(agent_id, {}).get('collided_by_robot_3', 0)
        
        # Log to wandb
        wandb.log(log_dict)

        # Print summary
        print(f"[Step {step:5d}] Survival: {survival_count}/4 | "
              f"Mean Reward: {mean_reward:.2f} | "
              f"Collisions: {total_collisions} (Active: {total_active_collisions}, Passive: {total_passive_collisions}) | "
              f"Immediate Kills: {total_immediate_kills} | "
              f"Non-Home Charges: {total_non_home_charges}")
        
        # Print per-agent breakdown
        print(f"  Per-Agent Status:")
        for agent_id in self.agent_ids:
            is_dead = current_infos.get(agent_id, {}).get('is_dead', False)
            death_marker = " 💀" if is_dead else ""
            pos = per_agent_positions[agent_id]
            
            # Get active/passive collision info
            active_collisions = per_agent_active_collisions[agent_id]
            passive_collisions = per_agent_passive_collisions[agent_id]
            
            print(f"    {agent_id}{death_marker}: Pos=({pos[0]},{pos[1]}), "
                  f"Energy={per_agent_energies[agent_id]}, "
                  f"Collisions={per_agent_collisions[agent_id]} (Active: {active_collisions}, Passive: {passive_collisions}), "
                  f"Charges={per_agent_charges[agent_id]}, "
                  f"NonHomeCharges={per_agent_non_home_charges[agent_id]}, "
                  f"ImmediateKills={per_agent_immediate_kills[agent_id]}")
            
            # Get pairwise collision info
            active_with = [
                current_infos.get(agent_id, {}).get('active_collisions_with_0', 0),
                current_infos.get(agent_id, {}).get('active_collisions_with_1', 0),
                current_infos.get(agent_id, {}).get('active_collisions_with_2', 0),
                current_infos.get(agent_id, {}).get('active_collisions_with_3', 0)
            ]
            collided_by = [
                current_infos.get(agent_id, {}).get('collided_by_robot_0', 0),
                current_infos.get(agent_id, {}).get('collided_by_robot_1', 0),
                current_infos.get(agent_id, {}).get('collided_by_robot_2', 0),
                current_infos.get(agent_id, {}).get('collided_by_robot_3', 0)
            ]
            
            print(f"      ActiveCollisionsWith: [0→{active_with[0]}, 1→{active_with[1]}, 2→{active_with[2]}, 3→{active_with[3]}]")
            print(f"      CollidedByRobot: [0→{collided_by[0]}, 1→{collided_by[1]}, 2→{collided_by[2]}, 3→{collided_by[3]}]")

    def log_final_summary(self, step_count: int, episode_rewards: Dict[str, float],
                         episode_infos_history: List[Dict]) -> Dict[str, float]:
        """
        Log final statistics at end of simulation
        """
        final_infos = episode_infos_history[-1] if episode_infos_history else {}
        
        # Survival count
        survival_count = sum(1 for agent_id in self.agent_ids
                            if not final_infos.get(agent_id, {}).get('is_dead', False))
        
        # Final rewards
        all_rewards = list(episode_rewards.values())
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        # Totals
        total_collisions = sum(final_infos.get(agent_id, {}).get('total_agent_collisions', 0)
                              for agent_id in self.agent_ids)
        total_charges = sum(final_infos.get(agent_id, {}).get('total_charges', 0)
                           for agent_id in self.agent_ids)
        total_non_home_charges = sum(final_infos.get(agent_id, {}).get('total_non_home_charges', 0)
                                     for agent_id in self.agent_ids)
        
        # Kills
        # total_kills, per_agent_kills = self.compute_kills(episode_infos_history) # Removed as per user request
        total_immediate_kills, per_agent_immediate_kills = self.compute_immediate_kills(episode_infos_history)
        
        # Per-agent final info
        per_agent_energies = {}
        per_agent_positions = {}
        per_agent_collisions = {}
        per_agent_charges = {}
        per_agent_non_home_charges = {}
        per_agent_deaths = {}

        for agent_id in self.agent_ids:
            info = final_infos.get(agent_id, {})
            per_agent_energies[agent_id] = info.get('energy', 0)
            per_agent_positions[agent_id] = info.get('position', (0, 0))
            per_agent_collisions[agent_id] = info.get('total_agent_collisions', 0)
            per_agent_charges[agent_id] = info.get('total_charges', 0)
            per_agent_non_home_charges[agent_id] = info.get('total_non_home_charges', 0)
            per_agent_deaths[agent_id] = 1 if info.get('is_dead', False) else 0

        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total Steps: {step_count}")
        print(f"Final Survival: {survival_count}/4")
        print(f"Mean Cumulative Reward: {mean_reward:.2f} (std: {std_reward:.2f})")
        print(f"Total Collisions: {total_collisions}")
        print(f"Total Charges: {total_charges}")
        print(f"Total Non-Home Charges: {total_non_home_charges}")
        print(f"Total Immediate Kills: {total_immediate_kills}")
        print("-" * 60)
        print("Per-Agent Final Statistics:")
        for agent_id in self.agent_ids:
            death_marker = " 💀 DEAD" if per_agent_deaths[agent_id] == 1 else " ✓ ALIVE"
            pos = per_agent_positions[agent_id]
            
            # Get collided_by info
            collided_by_0 = final_infos.get(agent_id, {}).get('collided_by_robot_0', 0)
            collided_by_1 = final_infos.get(agent_id, {}).get('collided_by_robot_1', 0)
            collided_by_2 = final_infos.get(agent_id, {}).get('collided_by_robot_2', 0)
            collided_by_3 = final_infos.get(agent_id, {}).get('collided_by_robot_3', 0)
            
            print(f"  {agent_id}{death_marker}")
            print(f"    Position: ({pos[0]}, {pos[1]})")
            print(f"    Final Energy: {per_agent_energies[agent_id]}")
            print(f"    Cumulative Reward: {episode_rewards[agent_id]:.2f}")
            print(f"    Collisions: {per_agent_collisions[agent_id]}")
            print(f"    Charges: {per_agent_charges[agent_id]}")
            print(f"    Non-Home Charges: {per_agent_non_home_charges[agent_id]}")
            print(f"    Immediate Kills: {per_agent_immediate_kills[agent_id]}")
            print(f"    Collided By: [R0→{collided_by_0}, R1→{collided_by_1}, R2→{collided_by_2}, R3→{collided_by_3}]")
        print("=" * 60)
        
        # Log final to wandb
        final_log = {
            "final/total_steps": step_count,
            "final/survival_count": survival_count,
            "final/mean_cumulative_reward": mean_reward,
            "final/std_cumulative_reward": std_reward,
            "final/total_collisions": total_collisions,
            "final/total_charges": total_charges,
            "final/total_non_home_charges": total_non_home_charges,
            "final/total_immediate_kills": total_immediate_kills,
        }

        for agent_id in self.agent_ids:
            pos = per_agent_positions[agent_id]
            final_log[f"final/{agent_id}/energy"] = per_agent_energies[agent_id]
            final_log[f"final/{agent_id}/position_x"] = pos[0]
            final_log[f"final/{agent_id}/position_y"] = pos[1]
            final_log[f"final/{agent_id}/cumulative_reward"] = episode_rewards[agent_id]
            final_log[f"final/{agent_id}/collisions"] = per_agent_collisions[agent_id]
            final_log[f"final/{agent_id}/charges"] = per_agent_charges[agent_id]
            final_log[f"final/{agent_id}/non_home_charges"] = per_agent_non_home_charges[agent_id]
            final_log[f"final/{agent_id}/immediate_kills"] = per_agent_immediate_kills[agent_id]
            final_log[f"final/{agent_id}/is_dead"] = per_agent_deaths[agent_id]
        
        wandb.log(final_log)

        return {
            'total_steps': step_count,
            'survival_count': survival_count,
            'mean_cumulative_reward': mean_reward,
            'total_collisions': total_collisions,
            'total_immediate_kills': total_immediate_kills,
        }


    def compute_immediate_kills(self, episode_infos: List[Dict]) -> Tuple[int, Dict[str, int]]:
        """
        Compute the number of "immediate kills" in this episode

        Definition: Robot A collides with robot B at time t, and at time t+1:
        - Exactly one of them dies (the other survives)
        - If both die, it does not count

        Args:
            episode_infos: Infos records for the entire episode

        Returns:
            Tuple of (total_immediate_kills, per_agent_immediate_kills_dict)
        """
        immediate_kills = 0
        per_agent_immediate_kills = {agent_id: 0 for agent_id in self.agent_ids}

        for t, infos in enumerate(episode_infos):
            if t + 1 >= len(episode_infos):
                break

            for agent_id in self.agent_ids:
                collision_target = infos[agent_id].get('collided_with_agent_id', None)

                if collision_target is not None:
                    # Get death status at t+1
                    agent_dead_next = episode_infos[t + 1][agent_id].get('is_dead', False)
                    target_dead_next = episode_infos[t + 1][collision_target].get('is_dead', False)

                    # Check if the collision target was alive at the time of collision
                    target_alive_at_collision = not infos[collision_target].get('is_dead', False)

                    # Immediate kill: target dies, agent survives, target was alive when collision happened
                    if target_dead_next and not agent_dead_next and target_alive_at_collision:
                        # Only attribute kill if agent was actively moving at time t
                        if infos[agent_id].get('is_mover_this_step', False):
                            immediate_kills += 1
                            per_agent_immediate_kills[agent_id] += 1

        return immediate_kills, per_agent_immediate_kills


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Independent DQN models with long-term simulation")

    # Model loading
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing saved model files (robot_0.pt, robot_1.pt, etc.)")

    # Environment parameters (should match training config)
    parser.add_argument("--env-n", type=int, default=3, help="Environment grid size (n×n)")
    parser.add_argument("--initial-energy", type=int, default=100, help="Initial energy for all robots (used if individual energies not specified)")
    parser.add_argument("--robot-0-energy", type=int, default=None, help="Initial energy for robot 0")
    parser.add_argument("--robot-1-energy", type=int, default=None, help="Initial energy for robot 1")
    parser.add_argument("--robot-2-energy", type=int, default=None, help="Initial energy for robot 2")
    parser.add_argument("--robot-3-energy", type=int, default=None, help="Initial energy for robot 3")
    parser.add_argument("--e-move", type=int, default=1, help="Energy cost per move")
    parser.add_argument("--e-charge", type=int, default=5, help="Energy gain per charge")
    parser.add_argument("--e-collision", type=int, default=3, help="Default energy loss per collision (used as fallback)")
    parser.add_argument("--e-collision-active-one-sided", type=int, default=None, help="Damage for active robot in one-sided collision")
    parser.add_argument("--e-collision-active-two-sided", type=int, default=None, help="Damage for active robot in two-sided collision")
    parser.add_argument("--e-collision-passive", type=int, default=None, help="Damage for passive robot in one-sided collision")

    # Simulation settings
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum steps for single long episode")
    parser.add_argument("--eval-epsilon", type=float, default=0.0,
                       help="Epsilon for evaluation (0 for deterministic inference)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (should match training)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (None for non-deterministic)")

    # Rendering (default: no rendering for long simulations)
    parser.add_argument("--render", action="store_true",
                       help="Enable pygame rendering (default: disabled)")

    # Wandb settings
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity (username or team name)")
    parser.add_argument("--wandb-project", type=str, default="robot-vacuum-eval", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default="long-simulation", help="Wandb run name")
    parser.add_argument("--wandb-mode", type=str, default="online", help="Wandb mode (online/offline/disabled)")

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

    # Run evaluation
    evaluator = ModelEvaluator(args)
    evaluator.evaluate()

    wandb.finish()


if __name__ == "__main__":
    main()
