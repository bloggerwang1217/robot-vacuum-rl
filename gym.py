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
            **kwargs: 或者使用關鍵字參數 (n, num_robots, initial_energy, robot_energies, e_move, e_charge, e_collision, n_steps)
        """
        # 支援兩種初始化方式
        if config is None:
            num_robots = kwargs.get('num_robots', 4)
            initial_energy = kwargs.get('initial_energy', 100)
            robot_energies = kwargs.get('robot_energies', None)
            # 如果沒有指定個別血量，使用統一血量
            if robot_energies is None:
                robot_energies = [initial_energy] * num_robots

            config = {
                'n': kwargs.get('n', 3),
                'num_robots': num_robots,
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
        self.n_robots = config.get('num_robots', 4)

        # 定義動作空間 (每個機器人: 0-4)
        self.action_space = spaces.Discrete(5)

        # 定義觀測空間 (每個機器人)
        # [自身位置2 + 自身能量1 + 其他機器人(n_robots-1)*3 + 充電座2*N + 可移動方向4]
        # = 3 + (n_robots-1)*3 + 2*num_chargers + 4 = 7 + (n_robots-1)*3 + 2*num_chargers
        num_chargers = len(self.env.charger_positions)
        obs_dim = 7 + (self.n_robots - 1) * 3 + 2 * num_chargers
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

        觀測向量結構 (16 + 2*N 維，N=充電座數量):
        - [0:2]: 自身位置 (x, y) 正規化到 [0, 1]
        - [2:3]: 自身能量正規化到 [0, 1]
        - [3:12]: 其他3個機器人的相對狀態 (dx, dy, energy) * 3
        - [12:12+2N]: N個充電座的相對位置 (dx, dy) * N
        - [12+2N:16+2N]: 可移動方向 (can_up, can_down, can_left, can_right)
        """
        observations = {}
        robots = state['robots']

        # 計算全域最大血量（用於正規化所有能量）
        global_max_energy = max(robot['max_energy'] for robot in robots)

        for i in range(self.n_robots):
            agent_id = self.agent_ids[i]
            robot = robots[i]

            obs = []

            # 1. 自身位置 (正規化到 [0, 1])
            obs.append(robot['x'] / (self.n - 1) if self.n > 1 else 0.5)
            obs.append(robot['y'] / (self.n - 1) if self.n > 1 else 0.5)

            # 2. 自身能量 (用全域最大血量正規化)
            obs.append(robot['energy'] / global_max_energy)

            # 3. 其他機器人的相對狀態
            for j in range(self.n_robots):
                if i == j:
                    continue
                other = robots[j]

                # 相對位置 (正規化到 [-1, 1])
                dx = (other['x'] - robot['x']) / (self.n - 1) if self.n > 1 else 0.0
                dy = (other['y'] - robot['y']) / (self.n - 1) if self.n > 1 else 0.0

                # 能量 (用全域最大血量正規化)
                energy = other['energy'] / global_max_energy

                obs.extend([dx, dy, energy])

            # 4. 充電座相對位置
            for charger_y, charger_x in self.env.charger_positions:
                # 相對位置 (正規化到 [-1, 1])
                dx = (charger_x - robot['x']) / (self.n - 1) if self.n > 1 else 0.0
                dy = (charger_y - robot['y']) / (self.n - 1) if self.n > 1 else 0.0
                obs.extend([dx, dy])

            # 5. 可移動方向 (邊界資訊)
            obs.append(1.0 if robot['y'] > 0 else 0.0)              # can_move_up
            obs.append(1.0 if robot['y'] < self.n - 1 else 0.0)     # can_move_down
            obs.append(1.0 if robot['x'] > 0 else 0.0)              # can_move_left
            obs.append(1.0 if robot['x'] < self.n - 1 else 0.0)     # can_move_right

            observations[agent_id] = np.array(obs, dtype=np.float32)

        return observations

    def _calculate_rewards(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        計算每個機器人的獎勵

        獎勵結構:
        1. 能量變化獎勵: energy_delta * 0.05
        2. 充電獎勵: +20.0
        3. 死亡懲罰: -100.0
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

            # 2. 充電獎勵（固定 20.0，避免 Q-value overestimation）
            if robot['charge_count'] > prev_robot['charge_count']:
                reward += 20.0

            # 3. 死亡懲罰
            if not robot['is_active'] and prev_robot['is_active']:
                reward -= 100.0

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
            collided_by_counts = {j: 0 for j in range(self.n_robots)}
            for event in robot['collision_events']:
                opponent_id = event['opponent_id']
                if opponent_id >= 0 and opponent_id < self.n_robots:  # 忽略邊界碰撞 (-1) 和不存在的機器人
                    collided_by_counts[opponent_id] += 1

            info_dict = {
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
                'collided_with_agent_id': collided_with,  # For kill analysis (converted to agent_id string)
                'step': state['current_step']
            }

            # 動態添加碰撞統計（根據實際機器人數量）
            for j in range(self.n_robots):
                info_dict[f'active_collisions_with_{j}'] = robot['active_collisions_with'].get(j, 0)
                info_dict[f'collided_by_robot_{j}'] = collided_by_counts.get(j, 0)

            infos[agent_id] = info_dict

        return infos

    def get_observation(self, robot_id: int) -> np.ndarray:
        """
        獲取指定 agent 的當前觀測

        Args:
            robot_id: 機器人 ID (0 到 n_robots-1)

        Returns:
            observation: 該 agent 的觀測向量
        """
        state = self.env.get_global_state()
        observations = self._get_observations(state)
        agent_id = self.agent_ids[robot_id]
        return observations[agent_id]

    def step_single(self, robot_id: int, action: int) -> Tuple[
        np.ndarray,     # observation (該 agent)
        float,          # reward (該 agent)
        bool,           # terminated (該 agent)
        bool,           # truncated (該 agent，在 advance_step 之前總是 False)
        Dict[str, Any]  # info (該 agent)
    ]:
        """
        執行單個機器人的動作

        Args:
            robot_id: 機器人 ID (0 到 n_robots-1)
            action: 動作 (0-4)

        Returns:
            observation: 該 agent 執行動作後的觀測
            reward: 該 agent 的獎勵
            terminated: 該 agent 是否死亡
            truncated: 是否達到最大步數（在 advance_step 之前總是 False）
            info: 該 agent 的額外資訊
        """
        agent_id = self.agent_ids[robot_id]

        # 記錄動作前的狀態（用於計算獎勵）
        prev_robot = self.prev_robots[robot_id].copy()

        # 執行動作
        state = self.env.step_single(robot_id, action)

        # 更新該 robot 的 prev_robots（用於下次獎勵計算）
        self.prev_robots[robot_id] = state['robots'][robot_id].copy()

        # 獲取該 agent 的觀測
        observations = self._get_observations(state)
        observation = observations[agent_id]

        # 計算該 agent 的獎勵
        robot = state['robots'][robot_id]
        reward = self._calculate_single_reward(robot, prev_robot)

        # 判斷終止條件
        terminated = not robot['is_active']
        truncated = False  # 在 advance_step 之前總是 False

        # 生成 info
        infos = self._get_infos(state)
        info = infos[agent_id]

        return observation, reward, terminated, truncated, info

    def _calculate_single_reward(self, robot: Dict[str, Any], prev_robot: Dict[str, Any]) -> float:
        """
        計算單個機器人的獎勵

        Args:
            robot: 當前機器人狀態
            prev_robot: 動作前的機器人狀態

        Returns:
            reward: 獎勵值
        """
        reward = 0.0

        # 1. 能量變化獎勵
        energy_delta = robot['energy'] - prev_robot['energy']
        reward += energy_delta * 0.05

        # 2. 充電獎勵（固定 20.0）
        if robot['charge_count'] > prev_robot['charge_count']:
            reward += 20.0

        # 3. 死亡懲罰
        if not robot['is_active'] and prev_robot['is_active']:
            reward -= 100.0

        return reward

    def advance_step(self) -> Tuple[bool, Dict[str, bool]]:
        """
        所有 agent 都行動完後，推進回合計數

        Returns:
            done: 是否達到最大步數
            truncations: 所有 agent 的截斷狀態
        """
        done = self.env.advance_step()

        truncations = {}
        for agent_id in self.agent_ids:
            truncations[agent_id] = done

        return done, truncations

    def render(self):
        """渲染環境 (委託給底層環境)"""
        self.env.render()

    def close(self):
        """關閉環境"""
        if hasattr(self.env, 'screen') and self.env.screen is not None:
            import pygame
            pygame.quit()
