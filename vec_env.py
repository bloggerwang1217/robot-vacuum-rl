"""
Vectorized Environment for Multi-Robot Vacuum RL
支援 N 個環境並行執行，大幅提升訓練效率
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from gym import RobotVacuumGymEnv


class VectorizedRobotVacuumEnv:
    """
    包裝 N 個 RobotVacuumGymEnv，支援並行 step
    
    優點：
    - 一次 GPU 推理處理 N*4 個 observations
    - 減少 Python 層面的 overhead
    - 自動處理 episode 結束時的 auto-reset
    """
    
    def __init__(self, num_envs: int, env_kwargs: Dict[str, Any]):
        """
        初始化 N 個環境

        Args:
            num_envs: 並行環境數量
            env_kwargs: 傳給每個 RobotVacuumGymEnv 的參數
        """
        self.num_envs = num_envs
        self.envs = [RobotVacuumGymEnv(**env_kwargs) for _ in range(num_envs)]
        
        # 取第一個環境的 observation/action space
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.agent_ids = self.envs[0].agent_ids
        self.n_agents = len(self.agent_ids)
        
        # Pre-allocate arrays for efficiency
        obs_dim = self.observation_space.shape[0]
        self._obs_buffer = np.zeros((num_envs, self.n_agents, obs_dim), dtype=np.float32)
        self._rewards_buffer = np.zeros((num_envs, self.n_agents), dtype=np.float32)
        self._terms_buffer = np.zeros((num_envs, self.n_agents), dtype=bool)
        self._truncs_buffer = np.zeros((num_envs, self.n_agents), dtype=bool)
        
        # Track episode counts per environment
        self.episode_counts = np.zeros(num_envs, dtype=np.int32)
        
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        重置所有環境
        
        Returns:
            observations: shape (num_envs, 4, obs_dim) 的 numpy array
            infos: 長度 num_envs 的 list，每個元素是 {agent_id: info_dict}
        """
        all_infos = []
        
        for env_idx, env in enumerate(self.envs):
            obs_dict, info = env.reset()
            # 直接寫入 pre-allocated buffer
            for agent_idx, agent_id in enumerate(self.agent_ids):
                self._obs_buffer[env_idx, agent_idx] = obs_dict[agent_id]
            all_infos.append(info)
            self.episode_counts[env_idx] = 0
        
        return self._obs_buffer.copy(), all_infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict], List[int]]:
        """
        並行執行所有環境的 step，自動處理 episode 結束
        
        Args:
            actions: shape (num_envs, 4) 的動作 array
            
        Returns:
            observations: shape (num_envs, 4, obs_dim)
            rewards: shape (num_envs, 4)
            terminations: shape (num_envs, 4) - bool
            truncations: shape (num_envs, 4) - bool
            infos: 長度 num_envs 的 list
            done_envs: 已完成 episode 的環境索引列表（用於統計）
        """
        all_infos = []
        done_envs = []
        
        for env_idx, env in enumerate(self.envs):
            env_actions = actions[env_idx].tolist()  # (4,) -> list
            obs_dict, rewards_dict, term_dict, trunc_dict, info = env.step(env_actions)
            
            # 直接寫入 pre-allocated buffers
            for agent_idx, agent_id in enumerate(self.agent_ids):
                self._obs_buffer[env_idx, agent_idx] = obs_dict[agent_id]
                self._rewards_buffer[env_idx, agent_idx] = rewards_dict[agent_id]
                self._terms_buffer[env_idx, agent_idx] = term_dict[agent_id]
                self._truncs_buffer[env_idx, agent_idx] = trunc_dict[agent_id]
            
            all_infos.append(info)
            
            # 檢查 episode 是否結束
            # 訓練時：永遠跑到 max steps 或全死
            alive_count = sum(1 for agent_id in self.agent_ids if not term_dict[agent_id])
            is_truncated = any(trunc_dict.values())

            should_end = alive_count == 0 or is_truncated

            if should_end:
                done_envs.append(env_idx)
                self.episode_counts[env_idx] += 1
                # Auto-reset
                reset_obs_dict, _ = env.reset()
                for agent_idx, agent_id in enumerate(self.agent_ids):
                    self._obs_buffer[env_idx, agent_idx] = reset_obs_dict[agent_id]
        
        return (
            self._obs_buffer.copy(),
            self._rewards_buffer.copy(),
            self._terms_buffer.copy(),
            self._truncs_buffer.copy(),
            all_infos,
            done_envs
        )
    
    def get_episode_counts(self) -> np.ndarray:
        """返回每個環境完成的 episode 數量"""
        return self.episode_counts.copy()
    
    def get_total_episodes(self) -> int:
        """返回所有環境完成的總 episode 數量"""
        return int(self.episode_counts.sum())

    def get_observation(self, robot_id: int) -> np.ndarray:
        """
        獲取指定 robot 在所有環境中的當前觀測

        Args:
            robot_id: 機器人 ID (0 到 n_agents-1)

        Returns:
            observations: shape (num_envs, obs_dim) 的 numpy array
        """
        for env_idx, env in enumerate(self.envs):
            obs = env.get_observation(robot_id)
            self._obs_buffer[env_idx, robot_id] = obs
        return self._obs_buffer[:, robot_id].copy()

    def step_single(self, robot_id: int, actions: np.ndarray) -> Tuple[
        np.ndarray,  # next_observations: (num_envs, obs_dim)
        np.ndarray,  # rewards: (num_envs,)
        np.ndarray,  # terminated: (num_envs,) bool
        np.ndarray,  # truncated: (num_envs,) bool - always False before advance_step
        List[Dict]   # infos: list of info dicts
    ]:
        """
        對指定 robot 執行動作（所有環境並行）

        Args:
            robot_id: 機器人 ID (0 到 n_agents-1)
            actions: shape (num_envs,) 的動作 array

        Returns:
            observations: 該 robot 執行動作後的觀測
            rewards: 該 robot 的獎勵
            terminated: 該 robot 是否死亡
            truncated: 是否達到最大步數（在 advance_step 之前總是 False）
            infos: 所有環境的 info
        """
        all_infos = []

        for env_idx, env in enumerate(self.envs):
            action = int(actions[env_idx])
            next_obs, reward, terminated, truncated, info = env.step_single(robot_id, action)

            self._obs_buffer[env_idx, robot_id] = next_obs
            self._rewards_buffer[env_idx, robot_id] = reward
            self._terms_buffer[env_idx, robot_id] = terminated
            self._truncs_buffer[env_idx, robot_id] = truncated

            all_infos.append(info)

        return (
            self._obs_buffer[:, robot_id].copy(),
            self._rewards_buffer[:, robot_id].copy(),
            self._terms_buffer[:, robot_id].copy(),
            self._truncs_buffer[:, robot_id].copy(),
            all_infos
        )

    def advance_step(self) -> Tuple[np.ndarray, List[int]]:
        """
        所有 robot 都行動完後，推進回合計數（所有環境）

        Returns:
            done_mask: shape (num_envs,) 表示哪些環境達到 max steps
            done_envs: 已完成 episode 的環境索引列表
        """
        done_envs = []
        done_mask = np.zeros(self.num_envs, dtype=bool)

        for env_idx, env in enumerate(self.envs):
            max_steps_reached, truncations = env.advance_step()
            done_mask[env_idx] = max_steps_reached

            # 檢查 episode 是否結束（全死或達到 max steps）
            alive_count = sum(1 for agent_id in self.agent_ids
                            if not self._terms_buffer[env_idx, self.agent_ids.index(agent_id)])
            should_end = alive_count == 0 or max_steps_reached

            if should_end:
                done_envs.append(env_idx)
                self.episode_counts[env_idx] += 1
                # Auto-reset
                reset_obs_dict, _ = env.reset()
                for agent_idx, agent_id in enumerate(self.agent_ids):
                    self._obs_buffer[env_idx, agent_idx] = reset_obs_dict[agent_id]
                # 重置 termination buffer
                self._terms_buffer[env_idx, :] = False

        return done_mask, done_envs
