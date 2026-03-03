"""
Vectorized Environment for Multi-Robot Vacuum RL
支援 N 個環境並行執行，大幅提升訓練效率
"""

import numpy as np
import multiprocessing as mp
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
        done_envs = []
        done_mask = np.zeros(self.num_envs, dtype=bool)

        for env_idx, env in enumerate(self.envs):
            max_steps_reached, truncations = env.advance_step()
            done_mask[env_idx] = max_steps_reached

            alive_count = sum(1 for agent_id in self.agent_ids
                            if not self._terms_buffer[env_idx, self.agent_ids.index(agent_id)])
            should_end = alive_count == 0 or max_steps_reached

            if should_end:
                done_envs.append(env_idx)
                self.episode_counts[env_idx] += 1
                reset_obs_dict, _ = env.reset()
                for agent_idx, agent_id in enumerate(self.agent_ids):
                    self._obs_buffer[env_idx, agent_idx] = reset_obs_dict[agent_id]
                self._terms_buffer[env_idx, :] = False

        return done_mask, done_envs


# ---------------------------------------------------------------------------
# Subprocess worker + SubprocVecEnv
# ---------------------------------------------------------------------------

def _env_worker(pipe: mp.connection.Connection, env_kwargs: Dict, num_local_envs: int):
    """
    Worker process: 持有一個 mini VectorizedRobotVacuumEnv，
    透過 Pipe 接收指令並回傳結果。
    在 Linux 上用 fork context 啟動，不需重新 import。
    """
    local_env = VectorizedRobotVacuumEnv(num_local_envs, env_kwargs)
    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break

        if cmd == 'reset':
            obs_arr, infos = local_env.reset()
            pipe.send((obs_arr, infos))
        elif cmd == 'get_obs':
            robot_id = data
            obs = local_env.get_observation(robot_id)
            pipe.send(obs)
        elif cmd == 'step_single':
            robot_id, actions = data
            result = local_env.step_single(robot_id, actions)
            pipe.send(result)
        elif cmd == 'advance_step':
            result = local_env.advance_step()
            pipe.send(result)
        elif cmd == 'close':
            break


class SubprocVecEnv:
    """
    Multiprocessing 版 VectorizedRobotVacuumEnv。

    開 num_workers 個 subprocess，每個負責 num_envs // num_workers 個 env。
    介面與 VectorizedRobotVacuumEnv 完全相同，train_dqn_vec.py 不需修改。

    用法（透過 --num-workers 傳入）：
        SubprocVecEnv(num_envs=32, env_kwargs=..., num_workers=8)
        → 8 個 process 各跑 4 個 env，同時執行，加速約 4-8x
    """

    def __init__(self, num_envs: int, env_kwargs: Dict[str, Any], num_workers: int):
        assert num_envs >= num_workers, \
            f"num_envs ({num_envs}) 必須 >= num_workers ({num_workers})"

        self.num_envs = num_envs
        self.num_workers = num_workers

        # 把 num_envs 均分給 num_workers（餘數給前幾個 worker）
        base = num_envs // num_workers
        rem  = num_envs % num_workers
        self.worker_sizes   = [base + (1 if i < rem else 0) for i in range(num_workers)]
        self.worker_offsets = [sum(self.worker_sizes[:i])   for i in range(num_workers)]

        # 建一個暫時 env 取 space / agent_ids（在 fork 前，CUDA 初始化前）
        _tmp = RobotVacuumGymEnv(**env_kwargs)
        self.observation_space = _tmp.observation_space
        self.action_space      = _tmp.action_space
        self.agent_ids         = _tmp.agent_ids
        self.n_agents          = len(self.agent_ids)
        del _tmp

        # Fork worker processes（Linux fork：不需重新 import，比 spawn 快）
        ctx = mp.get_context('fork')
        self._pipes = []
        self._procs = []
        for i in range(num_workers):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_env_worker,
                args=(child_conn, env_kwargs, self.worker_sizes[i]),
                daemon=True,
            )
            proc.start()
            child_conn.close()   # parent 不需要 child 端
            self._pipes.append(parent_conn)
            self._procs.append(proc)

        # Pre-allocate buffers（和 VectorizedRobotVacuumEnv 相同）
        obs_dim = self.observation_space.shape[0]
        self._obs_buffer     = np.zeros((num_envs, self.n_agents, obs_dim), dtype=np.float32)
        self._rewards_buffer = np.zeros((num_envs, self.n_agents), dtype=np.float32)
        self._terms_buffer   = np.zeros((num_envs, self.n_agents), dtype=bool)
        self._truncs_buffer  = np.zeros((num_envs, self.n_agents), dtype=bool)
        self.episode_counts  = np.zeros(num_envs, dtype=np.int32)

        print(f"[SubprocVecEnv] {num_workers} workers × "
              f"{self.worker_sizes[0]} envs = {num_envs} total envs")

    # ------------------------------------------------------------------
    # Public interface（與 VectorizedRobotVacuumEnv 完全相同）
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        for pipe in self._pipes:
            pipe.send(('reset', None))

        all_infos = []
        for w, pipe in enumerate(self._pipes):
            obs_arr, infos = pipe.recv()          # (n_local, n_agents, obs_dim)
            off, n = self.worker_offsets[w], self.worker_sizes[w]
            self._obs_buffer[off:off+n] = obs_arr
            all_infos.extend(infos)

        self.episode_counts[:] = 0
        return self._obs_buffer.copy(), all_infos

    def get_observation(self, robot_id: int) -> np.ndarray:
        for pipe in self._pipes:
            pipe.send(('get_obs', robot_id))

        for w, pipe in enumerate(self._pipes):
            obs_slice = pipe.recv()               # (n_local, obs_dim)
            off, n = self.worker_offsets[w], self.worker_sizes[w]
            self._obs_buffer[off:off+n, robot_id] = obs_slice

        return self._obs_buffer[:, robot_id].copy()

    def step_single(self, robot_id: int, actions: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]
    ]:
        # 同時廣播給所有 worker
        for w, pipe in enumerate(self._pipes):
            off, n = self.worker_offsets[w], self.worker_sizes[w]
            pipe.send(('step_single', (robot_id, actions[off:off+n])))

        # 收集結果（workers 已在並行跑）
        all_infos = []
        for w, pipe in enumerate(self._pipes):
            next_obs, rewards, terminated, truncated, infos = pipe.recv()
            off, n = self.worker_offsets[w], self.worker_sizes[w]
            self._obs_buffer[off:off+n,     robot_id] = next_obs
            self._rewards_buffer[off:off+n, robot_id] = rewards
            self._terms_buffer[off:off+n,   robot_id] = terminated
            self._truncs_buffer[off:off+n,  robot_id] = truncated
            all_infos.extend(infos)

        return (
            self._obs_buffer[:,     robot_id].copy(),
            self._rewards_buffer[:, robot_id].copy(),
            self._terms_buffer[:,   robot_id].copy(),
            self._truncs_buffer[:,  robot_id].copy(),
            all_infos,
        )

    def advance_step(self) -> Tuple[np.ndarray, List[int]]:
        for pipe in self._pipes:
            pipe.send(('advance_step', None))

        done_mask = np.zeros(self.num_envs, dtype=bool)
        done_envs: List[int] = []

        for w, pipe in enumerate(self._pipes):
            local_mask, local_done = pipe.recv()   # local_done: list of local indices
            off, n = self.worker_offsets[w], self.worker_sizes[w]
            done_mask[off:off+n] = local_mask
            for local_idx in local_done:
                global_idx = off + local_idx
                done_envs.append(global_idx)
                self.episode_counts[global_idx] += 1
                self._terms_buffer[global_idx, :] = False   # 重置已 auto-reset 的 env

        return done_mask, done_envs

    def get_episode_counts(self) -> np.ndarray:
        return self.episode_counts.copy()

    def get_total_episodes(self) -> int:
        return int(self.episode_counts.sum())

    def close(self):
        for pipe in self._pipes:
            try:
                pipe.send(('close', None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
