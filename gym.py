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
                'charger_positions': kwargs.get('charger_positions', None),
                'dust_max': kwargs.get('dust_max', 10.0),
                'dust_rate': kwargs.get('dust_rate', 0.5),
                'dust_epsilon': kwargs.get('dust_epsilon', 0.5),
                'charger_dust_max_ratio': kwargs.get('charger_dust_max_ratio', 0.3),
                'charger_dust_rate_ratio': kwargs.get('charger_dust_rate_ratio', 0.5),
                'dust_reward_scale': kwargs.get('dust_reward_scale', 0.05),
                'dust_enabled': kwargs.get('dust_enabled', True),
                'exclusive_charging': kwargs.get('exclusive_charging', False),
                'charger_range': kwargs.get('charger_range', 1),
                'robot_speeds': kwargs.get('robot_speeds', None),
                'random_start_robots': kwargs.get('random_start_robots', set()),
                'robot_start_positions': kwargs.get('robot_start_positions', {}),
                'agent_types_mode': kwargs.get('agent_types_mode', 'off'),
                'triangle_agent_id': kwargs.get('triangle_agent_id', None),
                'heterotype_charge_mode': kwargs.get('heterotype_charge_mode', 'off'),
                'heterotype_charge_factor': kwargs.get('heterotype_charge_factor', 1.0),
                'energy_cap': kwargs.get('energy_cap', None),
                'e_decay': kwargs.get('e_decay', 0.0),
                'robot_attack_powers': kwargs.get('robot_attack_powers', None),
                'thief_spawn': kwargs.get('thief_spawn', False),
                'legacy_obs': kwargs.get('legacy_obs', False),
                'alliance_groups': kwargs.get('alliance_groups', None),
                'alliance_zone': kwargs.get('alliance_zone', False),
                'energy_sharing_mode': kwargs.get('energy_sharing_mode', 'none'),
                'energy_sharing_events': kwargs.get('energy_sharing_events', ['charge', 'collision']),
                'energy_sharing_self_weight': kwargs.get('energy_sharing_self_weight', 2.0 / 3.0),
                'energy_sharing_ally_weight': kwargs.get('energy_sharing_ally_weight', 1.0 / 3.0),
                'stun_steps': kwargs.get('stun_steps', 0),
                'robot_stun_steps': kwargs.get('robot_stun_steps', None),
                'docking_steps': kwargs.get('docking_steps', 0),
                'robot_docking_steps': kwargs.get('robot_docking_steps', None),
            }

        # 創建底層環境
        self.env = RobotVacuumEnv(config)
        self.config = config

        # Alliance energy-sharing 配置（預設完全關閉）
        self._alliance_groups = config.get('alliance_groups', None)  # list of sets, e.g. [{0, 1}]
        self._energy_sharing_mode = config.get('energy_sharing_mode', 'none')  # 'none' or 'event_only'
        self._energy_sharing_events = config.get('energy_sharing_events', ['charge', 'collision'])
        self._energy_sharing_self_weight = config.get('energy_sharing_self_weight', 2.0 / 3.0)
        self._energy_sharing_ally_weight = config.get('energy_sharing_ally_weight', 1.0 / 3.0)

        # Buffer for energy_events during step_single calls (evaluation path)
        self._step_energy_events: Dict[int, Dict[str, float]] = {}
        # Buffer for corrected rewards after advance_step (evaluation path)
        self._corrected_rewards: Dict[str, float] = {}
        # Last step's energy sharing adjustments {robot_id: delta} (only set when alliance active)
        self._last_sharing_adjustments: Dict[int, float] = {}

        # 環境參數
        self.n = config.get('n', 3)
        self.initial_energy = config['initial_energy']
        self.n_robots = config.get('num_robots', 4)
        robot_energies = config.get('robot_energies', None)
        if robot_energies:
            self.initial_energies = list(robot_energies)
        else:
            self.initial_energies = [self.initial_energy] * self.n_robots
        self.dust_enabled = config.get('dust_enabled', True)
        self.dust_reward_scale = config.get('dust_reward_scale', 0.05)

        # Agent type system (trait.md)
        self.agent_types_mode = config.get('agent_types_mode', 'off')  # 'off' or 'observe'
        # legacy obs format: no self_type, no other_type (obs_dim = 3+4+(N-1)*3+C*2 [+n²])
        self.legacy_obs = config.get('legacy_obs', False)
        triangle_id = config.get('triangle_agent_id', None)
        # Build type array: 0=circle, 1=triangle
        self.agent_types = np.zeros(self.n_robots, dtype=np.float32)
        if triangle_id is not None and 0 <= triangle_id < self.n_robots:
            self.agent_types[triangle_id] = 1.0

        # 定義動作空間 (每個機器人: 0-4)
        self.action_space = spaces.Discrete(5)

        # 定義觀測空間 (每個機器人)
        # [自身位置2 + 自身能量1 + (自身type 1) + 牆壁4 + 其他機器人(n_robots-1)*(3+1) + 充電座2*N + 全地圖灰塵n*n (若啟用)]
        # type 欄位在 mode=off 時填 0 (padding)，mode=observe 時填實際 type
        num_chargers = len(self.env.charger_positions)
        # type dims only included when agent_types_mode == 'observe'
        if self.agent_types_mode == 'observe':
            obs_dim = 3 + 1 + 4 + (self.n_robots - 1) * 4 + 2 * num_chargers  # +1 self_type, +1 per other
        else:
            obs_dim = 3 + 4 + (self.n_robots - 1) * 3 + 2 * num_chargers  # no type dims
        if self.dust_enabled:
            obs_dim += self.n * self.n
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

    def _apply_energy_sharing(
        self,
        energy_events: List[Dict[str, float]],
        alliance_groups,
        mode: str,
        events: List[str],
        w_self: float,
        w_ally: float,
    ) -> Dict[int, float]:
        """
        計算 alliance energy sharing 後各 robot 需要的能量調整量。

        Args:
            energy_events: 每個 robot 的 event-level 能量記錄 (list of dicts, index = robot_id)
            alliance_groups: list of sets，例如 [{0, 1}]；None 或空 list 時不做共享
            mode: 'none' 或 'event_only'
            events: 要共享的事件種類，例如 ['charge', 'collision']
            w_self: 自己保留的比例（預設 2/3）
            w_ally: 給每個 ally 的比例（預設 1/3，2人 alliance 時 ally=1）

        Returns:
            adjustments: {robot_id: delta_energy}，需要加到各 robot 的能量上
        """
        adjustments = {i: 0.0 for i in range(self.n_robots)}

        if mode == 'none' or not alliance_groups:
            return adjustments

        # Get alive status for each robot
        state = self.env.get_global_state()
        alive = [r['is_active'] for r in state['robots']]

        if mode == 'event_only':
            for group in alliance_groups:
                group_list = sorted(group)
                if len(group_list) <= 1:
                    continue
                for robot_id in group_list:
                    # Dead robots don't share (they shouldn't have events anyway)
                    if not alive[robot_id]:
                        continue
                    # Count only alive allies for sharing denominator
                    alive_allies = [a for a in group_list if a != robot_id and alive[a]]
                    n_alive_allies = len(alive_allies)
                    if n_alive_allies <= 0:
                        continue  # no alive allies to share with; keep full amount
                    # 對每個指定事件，計算共享量
                    for event_key in events:
                        if robot_id >= len(energy_events):
                            continue
                        ev_val = energy_events[robot_id].get(event_key, 0.0)
                        if ev_val == 0.0:
                            continue
                        # 原本全給自己，現在自己保留 w_self，剩下分給 alive ally
                        keep = ev_val * w_self
                        share_total = ev_val - keep  # = ev_val * (1 - w_self)
                        # 調整自己：from full ev_val to keep
                        adjustments[robot_id] += keep - ev_val  # negative if sharing out positive events
                        # 平均分給每個 alive ally
                        share_per_ally = share_total / n_alive_allies
                        for ally_id in alive_allies:
                            adjustments[ally_id] += share_per_ally

        return adjustments

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

        # 清除 alliance sharing buffer
        self._step_energy_events = {}
        self._corrected_rewards = {}

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

        觀測向量結構 (3 + 1 + 4 + (N-1)*4 + C*2 維，N=機器人數量，C=充電座數量):
        - [0:2]: 自身位置 (x, y) 正規化到 [0, 1]
        - [2:3]: 自身能量正規化到 [0, 1]
        - [3:4]: 自身 type (0=circle, 1=triangle; mode=off 時為 0)
        - [4:8]: 牆壁指示器 [wall_up, wall_down, wall_left, wall_right] (0 or 1)
        - [8:8+(N-1)*4]: 其他機器人的相對狀態 (dx, dy, energy, type) * (N-1)
        - [8+(N-1)*4:]: C個充電座的相對位置 (dx, dy) * C
        """
        observations = {}
        robots = state['robots']

        # 計算全域正規化基準：取靜態上限與場上當前最高實際能量的最大值
        # 允許超充時，若有機器人超過 max_energy，正規化分母隨之上調，確保輸入永遠在 [0, 1]
        static_max = max(robot['max_energy'] for robot in robots)
        actual_max = max(
            (robot['energy'] for robot in robots if robot['is_active']),
            default=static_max
        )
        global_max_energy = max(static_max, actual_max)

        for i in range(self.n_robots):
            agent_id = self.agent_ids[i]
            robot = robots[i]

            obs = []

            # 1. 自身位置 (正規化到 [0, 1])
            obs.append(robot['x'] / (self.n - 1) if self.n > 1 else 0.5)
            obs.append(robot['y'] / (self.n - 1) if self.n > 1 else 0.5)

            # 2. 自身能量 (用全域最大血量正規化)
            obs.append(robot['energy'] / global_max_energy)

            # 3. 自身 type (只在 mode=observe 時加入)
            if self.agent_types_mode == 'observe':
                obs.append(self.agent_types[i])

            # 4. 牆壁指示器 (1.0 = 該方向緊鄰邊界，不能移動)
            obs.append(1.0 if robot['y'] == 0         else 0.0)  # wall_up
            obs.append(1.0 if robot['y'] == self.n - 1 else 0.0)  # wall_down
            obs.append(1.0 if robot['x'] == 0         else 0.0)  # wall_left
            obs.append(1.0 if robot['x'] == self.n - 1 else 0.0)  # wall_right

            # 5. 其他機器人的相對狀態
            for j in range(self.n_robots):
                if i == j:
                    continue
                other = robots[j]

                # 相對位置 (正規化到 [-1, 1])
                dx = (other['x'] - robot['x']) / (self.n - 1) if self.n > 1 else 0.0
                dy = (other['y'] - robot['y']) / (self.n - 1) if self.n > 1 else 0.0

                # 能量 (用全域最大血量正規化)
                energy = other['energy'] / global_max_energy

                if self.agent_types_mode == 'observe':
                    other_type = self.agent_types[j]
                    obs.extend([dx, dy, energy, other_type])
                else:
                    obs.extend([dx, dy, energy])

            # 6. 充電座相對位置
            for charger_y, charger_x in self.env.charger_positions:
                # 相對位置 (正規化到 [-1, 1])
                dx = (charger_x - robot['x']) / (self.n - 1) if self.n > 1 else 0.0
                dy = (charger_y - robot['y']) / (self.n - 1) if self.n > 1 else 0.0
                obs.extend([dx, dy])

            # 7. 全地圖灰塵（每格用各自上限正規化到 [0, 1]，row-major y→x）
            if self.dust_enabled:
                dust_grid = state['dust_grid']
                d_max_normal = self.env.dust_max
                d_max_charger = self.env.dust_max * self.env.charger_dust_max_ratio
                for dy in range(self.n):
                    for dx in range(self.n):
                        d_max = d_max_charger if self.env.is_charger_grid[dy, dx] else d_max_normal
                        obs.append(dust_grid[dy, dx] / d_max)

            observations[agent_id] = np.array(obs, dtype=np.float32)

        return observations

    def _calculate_rewards(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        計算每個機器人的獎勵 — delta-energy reward

        獎勵結構:
        1. 能量變化獎勵: ΔEnergy × 0.05
        2. 死亡懲罰: -100.0
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

            # 2. 死亡懲罰
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
                'stun_remaining': robot.get('stun_counter', 0),  # 剩餘 stun 步數（replay 用）
                'step': state['current_step']
            }

            # 動態添加碰撞統計（根據實際機器人數量）
            for j in range(self.n_robots):
                info_dict[f'active_collisions_with_{j}'] = robot['active_collisions_with'].get(j, 0)
                info_dict[f'collided_by_robot_{j}'] = collided_by_counts.get(j, 0)

            infos[agent_id] = info_dict

        # Step-level charger log (heterotype experiment)
        if 'charger_log' in state:
            infos['charger_log'] = state['charger_log']

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

    def step_single(self, robot_id: int, action: int, is_last_turn: bool = True) -> Tuple[
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
            is_last_turn: 是否為該 robot 本 step 的最後一個 sub-turn（True 才充電）

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
        state = self.env.step_single(robot_id, action, is_last_turn=is_last_turn)

        # 更新該 robot 的 prev_robots（用於下次獎勵計算）
        self.prev_robots[robot_id] = state['robots'][robot_id].copy()

        # 獲取該 agent 的觀測
        observations = self._get_observations(state)
        observation = observations[agent_id]

        # 計算該 agent 的 reward
        robot = state['robots'][robot_id]
        reward = self._calculate_single_reward(robot, prev_robot, robot_idx=robot_id)

        # 判斷終止條件
        terminated = not robot['is_active']
        truncated = False  # 在 advance_step 之前總是 False

        # 生成 info
        infos = self._get_infos(state)
        info = infos[agent_id]

        return observation, reward, terminated, truncated, info

    def _calculate_single_reward(self, robot: Dict[str, Any], prev_robot: Dict[str, Any], robot_idx: int = 0) -> float:
        """
        計算單個機器人的獎勵 — delta-energy reward，與 batch_env.py 一致

        Args:
            robot: 當前機器人狀態
            prev_robot: 動作前的機器人狀態
            robot_idx: 未使用，保留介面相容

        Returns:
            reward: 獎勵值
        """
        reward = 0.0

        # 1. 能量變化獎勵
        energy_delta = robot['energy'] - prev_robot['energy']
        reward += energy_delta * 0.05

        # 2. 死亡懲罰
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
