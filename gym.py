"""
Gymnasium-compatible wrapper for Multi-Robot Energy Survival Environment
將 robot_vacuum_env.py 封裝成符合 Gymnasium 標準的多智能體環境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
from robot_vacuum_env import RobotVacuumEnv


class RobotVacuumGymEnv:
    """
    Gymnasium-compatible Multi-Agent Environment

    將 RobotVacuumEnv 封裝成符合 Gymnasium 標準的接口
    每個機器人都有獨立的觀測、獎勵和終止狀態
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """
        初始化環境

        Args:
            config: 環境配置字典 (可選)
            **kwargs: 或者使用關鍵字參數 (n, initial_energy, robot_energies, e_move, e_charge, e_collision, n_steps)
        """
        # 支援兩種初始化方式
        if config is None:
            initial_energy = kwargs.get('initial_energy', 100)
            robot_energies = kwargs.get('robot_energies', None)
            # 如果沒有指定個別血量，使用統一血量
            if robot_energies is None:
                robot_energies = [initial_energy] * 4
            
            config = {
                'n': kwargs.get('n', 3),
                'initial_energy': initial_energy,
                'robot_energies': robot_energies,
                'e_move': kwargs.get('e_move', 1),
                'e_charge': kwargs.get('e_charge', 5),
                'e_collision': kwargs.get('e_collision', 3),
                'e_boundary': kwargs.get('e_boundary', 50),
                'e_collision_active_one_sided': kwargs.get('e_collision_active_one_sided', None),
                'e_collision_active_two_sided': kwargs.get('e_collision_active_two_sided', None),
                'e_collision_passive': kwargs.get('e_collision_passive', None),
                'n_steps': kwargs.get('n_steps', 500),
                'epsilon': kwargs.get('epsilon', 0.2),
                'charger_positions': kwargs.get('charger_positions', None)
            }

        # 創建底層環境
        self.env = RobotVacuumEnv(config)
        self.config = config

        # 環境參數
        self.n = config.get('n', 3)
        self.initial_energy = config['initial_energy']
        self.n_robots = 4

        # 定義動作空間 (每個機器人: 0-4)
        self.action_space = spaces.Discrete(5)

        # 定義觀測空間 (每個機器人)
        # [自身位置2 + 自身能量1 + 其他機器人3*3 + 充電座2*N]
        # = 3 + 9 + 2*num_chargers = 12 + 2*num_chargers
        num_chargers = len(self.env.charger_positions)
        obs_dim = 12 + 2 * num_chargers
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 機器人 ID 列表
        self.agent_ids = [f'robot_{i}' for i in range(self.n_robots)]

        # 追蹤上一步的狀態 (用於計算獎勵)
        self.prev_robots = None

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置環境

        Returns:
            observations: 每個機器人的觀測字典
            infos: 每個機器人的額外資訊字典
        """
        # 重置底層環境
        state = self.env.reset()

        # 儲存當前狀態
        self.prev_robots = [robot.copy() for robot in state['robots']]

        # 生成觀測
        observations = self._get_observations(state)

        # 生成 infos
        infos = self._get_infos(state)

        return observations, infos

    def step(self, actions: List[int]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],        # rewards
        Dict[str, bool],         # terminations
        Dict[str, bool],         # truncations
        Dict[str, Any]           # infos
    ]:
        """
        執行一步

        Args:
            actions: 4個機器人的動作列表

        Returns:
            observations: 每個機器人的觀測
            rewards: 每個機器人的獎勵
            terminations: 每個機器人是否終止
            truncations: 每個機器人是否截斷
            infos: 每個機器人的額外資訊
        """
        # 執行動作
        state, done = self.env.step(actions)

        # 生成觀測
        observations = self._get_observations(state)

        # 計算獎勵
        rewards = self._calculate_rewards(state)

        # 判斷終止條件
        terminations = {}
        truncations = {}
        for i in range(self.n_robots):
            agent_id = self.agent_ids[i]
            robot = state['robots'][i]

            # 機器人死亡 -> termination
            terminations[agent_id] = not robot['is_active']

            # 達到最大步數 -> truncation
            truncations[agent_id] = done

        # 生成 infos
        infos = self._get_infos(state)

        # 更新上一步狀態
        self.prev_robots = [robot.copy() for robot in state['robots']]

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        為每個機器人生成觀測向量

        觀測向量結構 (20 維):
        - [0:2]: 自身位置 (x, y) 正規化到 [0, 1]
        - [2:3]: 自身能量正規化到 [0, 1]
        - [3:12]: 其他3個機器人的相對狀態 (dx, dy, energy) * 3
        - [12:20]: 4個充電座的相對位置 (dx, dy) * 4
        """
        observations = {}
        robots = state['robots']

        for i in range(self.n_robots):
            agent_id = self.agent_ids[i]
            robot = robots[i]

            obs = []

            # 1. 自身位置 (正規化到 [0, 1])
            obs.append(robot['x'] / (self.n - 1) if self.n > 1 else 0.5)
            obs.append(robot['y'] / (self.n - 1) if self.n > 1 else 0.5)

            # 2. 自身能量 (正規化到 [0, 1])
            obs.append(robot['energy'] / self.initial_energy)

            # 3. 其他機器人的相對狀態
            for j in range(self.n_robots):
                if i == j:
                    continue
                other = robots[j]

                # 相對位置 (正規化到 [-1, 1])
                dx = (other['x'] - robot['x']) / (self.n - 1) if self.n > 1 else 0.0
                dy = (other['y'] - robot['y']) / (self.n - 1) if self.n > 1 else 0.0

                # 能量 (正規化到 [0, 1])
                energy = other['energy'] / self.initial_energy

                obs.extend([dx, dy, energy])

            # 4. 充電座相對位置
            for charger_y, charger_x in self.env.charger_positions:
                # 相對位置 (正規化到 [-1, 1])
                dx = (charger_x - robot['x']) / (self.n - 1) if self.n > 1 else 0.0
                dy = (charger_y - robot['y']) / (self.n - 1) if self.n > 1 else 0.0
                obs.extend([dx, dy])

            observations[agent_id] = np.array(obs, dtype=np.float32)

        return observations

    def _calculate_rewards(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        計算每個機器人的獎勵

        獎勵結構:
        1. 能量變化獎勵: energy_delta * 0.05
        2. 充電獎勵: +10.0
        3. 死亡懲罰: -100.0
        4. 存活獎勵: +0.1
        """
        rewards = {}
        robots = state['robots']

        for i in range(self.n_robots):
            agent_id = self.agent_ids[i]
            robot = robots[i]
            prev_robot = self.prev_robots[i]

            reward = 0.0

            # 1. 能量變化獎勵
            energy_delta = robot['energy'] - prev_robot['energy']
            reward += energy_delta * 0.05

            # 2. 充電獎勵 (提升 10 倍: 0.5 → 5.0)
            if robot['charge_count'] > prev_robot['charge_count']:
                reward += 10.0

            # 3. 死亡懲罰
            if not robot['is_active'] and prev_robot['is_active']:
                reward -= 100.0

            # 4. 存活獎勵 (提升 10 倍: 0.01 → 0.1)
            if robot['is_active']:
                reward += 0.1

            rewards[agent_id] = reward

        return rewards

    def _get_infos(self, state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        生成診斷資訊
        """
        infos = {}
        robots = state['robots']

        for i in range(self.n_robots):
            agent_id = self.agent_ids[i]
            robot = robots[i]

            # Convert collided_with_agent_id from integer index to agent_id string
            collided_with = None
            if robot['collided_with_agent_id'] is not None:
                collided_with = self.agent_ids[robot['collided_with_agent_id']]

            # 從 collision_events 計算被各個機器人碰撞的次數
            collided_by_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for event in robot['collision_events']:
                opponent_id = event['opponent_id']
                if opponent_id >= 0:  # 忽略邊界碰撞 (-1)
                    collided_by_counts[opponent_id] += 1

            infos[agent_id] = {
                'energy': robot['energy'],
                'position': (robot['x'], robot['y']),
                'is_active': robot['is_active'],
                'is_dead': not robot['is_active'],  # For kill analysis
                'is_mover_this_step': robot['is_mover_this_step'],  # 本回合是否主動移動 (用於 kill 歸屬分析)
                'charge_count': robot['charge_count'],  # Keep for backwards compatibility
                'total_charges': robot['charge_count'],  # Cumulative charges (same as charge_count)
                'total_non_home_charges': robot['non_home_charge_count'],  # Cumulative non-home charges
                'total_agent_collisions': robot['active_collision_count'] + robot['passive_collision_count'],  # 總碰撞次數 (主動+被動)
                'total_active_collisions': robot['active_collision_count'],  # 主動碰撞總次數
                'total_passive_collisions': robot['passive_collision_count'],  # 被碰撞總次數
                'active_collisions_with_0': robot['active_collisions_with'][0],  # 主動碰撞 robot 0 次數
                'active_collisions_with_1': robot['active_collisions_with'][1],  # 主動碰撞 robot 1 次數
                'active_collisions_with_2': robot['active_collisions_with'][2],  # 主動碰撞 robot 2 次數
                'active_collisions_with_3': robot['active_collisions_with'][3],  # 主動碰撞 robot 3 次數
                'collided_by_robot_0': collided_by_counts[0],  # 被 robot 0 碰撞次數 (從 collision_events 計算)
                'collided_by_robot_1': collided_by_counts[1],  # 被 robot 1 碰撞次數 (從 collision_events 計算)
                'collided_by_robot_2': collided_by_counts[2],  # 被 robot 2 碰撞次數 (從 collision_events 計算)
                'collided_by_robot_3': collided_by_counts[3],  # 被 robot 3 碰撞次數 (從 collision_events 計算)
                'collided_with_agent_id': collided_with,  # For kill analysis (converted to agent_id string)
                'step': state['current_step']
            }

        return infos

    def render(self):
        """渲染環境 (委託給底層環境)"""
        self.env.render()

    def close(self):
        """關閉環境"""
        if hasattr(self.env, 'screen') and self.env.screen is not None:
            import pygame
            pygame.quit()
