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
        self.robot_energies = robot_energies

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

        # Environment setup (render_mode based on args)
        render_mode = "human" if args.render else None
        self.env = RobotVacuumGymEnv(
            n=args.env_n,
            num_robots=self.num_robots,
            initial_energy=args.initial_energy,
            robot_energies=robot_energies,
            e_move=args.e_move,
            e_charge=args.e_charge,
            e_collision=args.e_collision,
            e_boundary=args.e_boundary,
            n_steps=args.max_steps,  # Use max_steps for single long episode
            render_mode=render_mode,
            charger_positions=charger_positions
        )

        # Initialize agents (only for the robots that exist)
        self.agent_ids = [f'robot_{i}' for i in range(self.num_robots)]
        self.agents = {}

        observation_dim = self.env.observation_space.shape[0]  # Get actual observation dim from env
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
        Run single long-term evaluation episode (sequential actions: one robot moves at a time)

        Returns:
            Evaluation statistics
        """
        print(f"\n{'=' * 60}")
        print(f"Starting Long-Term Simulation (Sequential Mode)")
        print(f"Max Steps: {self.args.max_steps}")
        print(f"Robot Energies: {self.robot_energies}")
        print(f"Eval Epsilon: {self.args.eval_epsilon}")
        print(f"{'=' * 60}\n")

        # Reset environment with seed if provided
        observations, infos = self.env.reset(seed=self.args.seed)

        episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        episode_infos_history = []
        episode_actions_history = []
        episode_q_values_history = []  # Store Q-values for debugging
        step_count = 0

        # 延長模式追蹤（只剩 1 個 agent 時延長 100 步）
        self._extension_started = False
        self._extension_steps = 0
        done = False

        # Track terminations across the episode
        terminations = {agent_id: False for agent_id in self.agent_ids}

        # Sub-step history for replay (記錄每個 robot 行動前後的狀態)
        episode_substeps_history = []

        # Main simulation loop
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        while not done:
            # Sequential actions: each robot acts one at a time
            actions = []
            step_q_values = {}  # Store Q-values for this step
            step_substeps = []  # 這個 step 的所有 sub-steps

            for robot_id in range(self.num_robots):
                agent_id = self.agent_ids[robot_id]
                agent = self.agents[agent_id]

                # 記錄行動前的狀態
                state_before = self.env.env.get_global_state()
                robots_before = {
                    f"robot_{i}": {
                        "position": [state_before['robots'][i]['x'], state_before['robots'][i]['y']],
                        "energy": state_before['robots'][i]['energy'],
                        "is_dead": not state_before['robots'][i]['is_active']
                    }
                    for i in range(self.num_robots)
                }

                # 1. Get current observation (latest state)
                obs = self.env.get_observation(robot_id)

                # 2. Select action with Q-values
                action, q_values = agent.select_action(obs, eval_mode=True, return_q_values=True)
                actions.append(action)

                # Store Q-values
                agent_q_values = {
                    action_names[i]: float(q_values[i])
                    for i in range(len(action_names))
                }
                step_q_values[agent_id] = agent_q_values

                # Print Q-values for this step (debug output)
                q_str = ", ".join([f"{action_names[i]}:{q_values[i]:.3f}" for i in range(len(action_names))])
                print(f"  {agent_id} Q-values: [{q_str}] Selected: {action_names[action]}")

                # 3. Execute action
                next_obs, reward, terminated, truncated, info = self.env.step_single(robot_id, action)

                # 記錄行動後的狀態
                state_after = self.env.env.get_global_state()
                robots_after = {
                    f"robot_{i}": {
                        "position": [state_after['robots'][i]['x'], state_after['robots'][i]['y']],
                        "energy": state_after['robots'][i]['energy'],
                        "is_dead": not state_after['robots'][i]['is_active']
                    }
                    for i in range(self.num_robots)
                }

                # 記錄這個 sub-step
                substep_data = {
                    "robot_id": robot_id,
                    "agent_id": agent_id,
                    "action": action_names[action],
                    "q_values": agent_q_values,
                    "robots_before": robots_before,
                    "robots_after": robots_after,
                    "reward": reward,
                    "terminated": terminated
                }
                step_substeps.append(substep_data)

                # Accumulate rewards
                episode_rewards[agent_id] += reward

                # Update termination status
                if terminated:
                    terminations[agent_id] = True

            # 4. Advance step count after all robots have acted
            max_steps_reached, truncations = self.env.advance_step()
            step_count += 1

            # Render if enabled
            if self.args.render:
                self.env.render()

            # Collect infos for analysis and replay
            state = self.env.env.get_global_state()
            infos = self.env._get_infos(state)

            # Store actions, infos, q_values, and substeps for analysis and replay
            episode_actions_history.append(actions)
            episode_infos_history.append(infos)
            episode_q_values_history.append(step_q_values)
            episode_substeps_history.append(step_substeps)

            # Log EVERY step for complete dynamics tracking
            self.log_step_summary(step_count, episode_rewards, episode_infos_history)

            # Count surviving agents
            survival_count = sum(1 for agent_id in self.agent_ids
                                if not terminations.get(agent_id, False))

            # 推論時：只剩 1 個 agent 後延長 100 步，然後結束
            # 用來觀察勝利者的後續行為
            if survival_count == 0:
                done = True
                print(f"\n[Step {step_count}] All agents have died. Simulation ended.")
            elif survival_count == 1 and self.num_robots > 1 and not self._extension_started:
                # 開始延長 100 步
                self._extension_started = True
                self._extension_steps = 0
                survivor = [agent_id for agent_id in self.agent_ids
                           if not terminations.get(agent_id, False)][0]
                print(f"\n[Step {step_count}] Only {survivor} survives. Extending for 100 more steps...")
            elif self._extension_started:
                self._extension_steps += 1
                if self._extension_steps >= 100:
                    done = True
                    print(f"\n[Step {step_count}] Extension complete (100 steps). Simulation ended.")
            elif max_steps_reached:
                done = True
                print(f"\n[Step {step_count}] Max steps reached. Simulation ended.")

        # Final summary
        final_stats = self.log_final_summary(step_count, episode_rewards, episode_infos_history)

        # Save replay data for visualization (包含 sub-steps)
        replay_file = self.save_replay(episode_actions_history, episode_infos_history, episode_q_values_history, episode_substeps_history, step_count)
        print(f"\nReplay data saved to: {replay_file}")

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

    def save_replay(self, episode_actions: List[List[int]], episode_infos: List[Dict],
                   episode_q_values: List[Dict], episode_substeps: List[List[Dict]], total_steps: int) -> str:
        """
        Save episode replay data as JSON for visualization with pygame

        Args:
            episode_actions: List of action lists (each step has N actions, one per robot)
            episode_infos: List of info dicts from environment
            episode_q_values: List of Q-value dicts for each step
            episode_substeps: List of substep lists (每個 step 包含每個 robot 的 sub-step 資料)
            total_steps: Total number of steps in episode

        Returns:
            Path to saved replay file
        """
        import json
        from pathlib import Path

        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

        # Build replay data structure
        replay_data = {
            "config": {
                "grid_size": self.args.env_n,
                "num_robots": self.num_robots,
                "charger_positions": self._parse_charger_positions(),
                "robot_initial_energies": {
                    f"robot_{i}": self.robot_energies[i]
                    for i in range(self.num_robots)
                },
                "parameters": {
                    "e_move": self.args.e_move,
                    "e_collision": self.args.e_collision,
                    "e_boundary": self.args.e_boundary,
                    "e_charge": self.args.e_charge,
                }
            },
            "steps": []
        }

        # Process each step
        for step_idx in range(total_steps):
            actions = episode_actions[step_idx]
            infos = episode_infos[step_idx]
            q_values = episode_q_values[step_idx]
            substeps = episode_substeps[step_idx]

            # Build step data
            step_data = {
                "step": step_idx,
                "actions": {
                    f"robot_{i}": action_names[actions[i]]
                    for i in range(self.num_robots)
                },
                "q_values": q_values,
                "robots": {},
                "sub_steps": substeps  # 新增：每個 robot 的 sub-step 資料
            }

            # Extract robot state at this step (final state after all robots acted)
            for agent_id in self.agent_ids:
                agent_info = infos.get(agent_id, {})
                pos = agent_info.get('position', (0, 0))

                step_data["robots"][agent_id] = {
                    "position": list(pos),
                    "energy": agent_info.get('energy', 0),
                    "is_dead": agent_info.get('is_dead', False),
                }

            # Extract events from this step (collisions, charges, deaths)
            events = []

            # Check for collisions
            for agent_id in self.agent_ids:
                agent_info = infos.get(agent_id, {})
                collision_target = agent_info.get('collided_with_agent_id', None)

                if collision_target is not None:
                    damage = self.args.e_collision
                    events.append({
                        "type": "collision",
                        "attacker": agent_id,
                        "victim": collision_target,
                        "damage": damage
                    })

            # Check for deaths (death happens when energy <= 0)
            if step_idx > 0:
                prev_infos = episode_infos[step_idx - 1]
                for agent_id in self.agent_ids:
                    prev_dead = prev_infos.get(agent_id, {}).get('is_dead', False)
                    curr_dead = infos.get(agent_id, {}).get('is_dead', False)

                    if not prev_dead and curr_dead:
                        events.append({
                            "type": "death",
                            "robot": agent_id,
                            "cause": "energy_depleted"
                        })

            step_data["events"] = events
            replay_data["steps"].append(step_data)

        # Save to JSON file
        # Determine output path based on wandb run name or default
        run_name = self.args.wandb_run_name if hasattr(self.args, 'wandb_run_name') else "replay"
        replay_dir = Path(self.args.model_dir).parent  # Save in same directory as model
        replay_file = replay_dir / f"{run_name}_replay.json"

        with open(replay_file, 'w') as f:
            json.dump(replay_data, f, indent=2)

        return str(replay_file)

    def _parse_charger_positions(self) -> List[List[int]]:
        """Parse charger positions from args"""
        if self.args.charger_positions is None:
            return [[0, 0], [0, self.args.env_n-1],
                    [self.args.env_n-1, 0], [self.args.env_n-1, self.args.env_n-1]]

        try:
            positions = []
            for pos_str in self.args.charger_positions.split(';'):
                y, x = map(int, pos_str.split(','))
                positions.append([y, x])
            return positions
        except:
            # Default to four corners
            return [[0, 0], [0, self.args.env_n-1],
                    [self.args.env_n-1, 0], [self.args.env_n-1, self.args.env_n-1]]


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
    parser.add_argument("--charger-positions", type=str, default=None,
                       help='Charger positions as "y1,x1;y2,x2;..." (e.g., "0,0;0,2;2,0;2,2"). Use -1,-1 to disable a charger. Default: four corners')

    # Simulation settings
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps for single long episode")
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
