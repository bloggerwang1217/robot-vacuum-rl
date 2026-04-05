"""
BatchRobotVacuumEnv
===================
全 numpy 向量化的 N-env 批次環境。

以 shape (N, ...) 的 numpy array 取代 N 個 Python dict 物件，
所有物理運算一次處理全部 N 個 env，消除 Python 逐 env 迴圈。

介面與 VectorizedRobotVacuumEnv 完全相同，train_dqn_vec.py 可直接替換。
"""

import numpy as np
import random
from typing import List, Tuple, Any, Dict

# ── Action constants ───────────────────────────────────────────────────────
_UP, _DOWN, _LEFT, _RIGHT, _STAY = 0, 1, 2, 3, 4
# dy, dx per action (index 0-4)
_ACT_DY = np.array([-1,  1,  0,  0, 0], dtype=np.int32)
_ACT_DX = np.array([ 0,  0, -1,  1, 0], dtype=np.int32)


class BatchRobotVacuumEnv:
    """
    N 個 RobotVacuumEnv 的全 numpy 向量化版本。

    狀態 arrays (N = num_envs, R = num_robots):
      pos:    (N, R, 2) int32   — axis-2: [y, x]
      energy: (N, R)   float32
      alive:  (N, R)   bool
      steps:  (N,)     int32
      dust:   (N, n, n) float32  (若啟用)
    """

    def __init__(self, num_envs: int, env_kwargs: Dict[str, Any]):
        N = num_envs
        self.N        = N
        self.n        = env_kwargs['n']
        self.R        = env_kwargs['num_robots']
        self.n_steps  = env_kwargs['n_steps']
        n, R          = self.n, self.R

        # ── Scalar config ──────────────────────────────────────────────────
        self.e_move       = float(env_kwargs.get('e_move', 1))
        self.e_charge     = float(env_kwargs.get('e_charge', 5))
        self.e_collision  = float(env_kwargs.get('e_collision', 3))
        self.e_boundary   = float(env_kwargs.get('e_boundary', 50))
        self.exclusive_charging = bool(env_kwargs.get('exclusive_charging', False))
        self.charger_range = int(env_kwargs.get('charger_range', 1))
        self.heterotype_charge_mode = env_kwargs.get('heterotype_charge_mode', 'off')
        self.heterotype_charge_factor = float(env_kwargs.get('heterotype_charge_factor', 1.0))
        self.energy_cap = env_kwargs.get('energy_cap', None)
        if self.energy_cap is not None:
            self.energy_cap = float(self.energy_cap)
        self.e_decay = float(env_kwargs.get('e_decay', 0.0))
        self.reward_mode = env_kwargs.get('reward_mode', 'delta-energy')
        self.reward_alpha = float(env_kwargs.get('reward_alpha', 0.05))
        speeds = env_kwargs.get('robot_speeds', None) or [1] * R
        self.robot_speeds = list(speeds)

        # ── Docking config (consecutive steps on charger before charging) ──
        default_dock = int(env_kwargs.get('docking_steps', 0))
        robot_docking = env_kwargs.get('robot_docking_steps', None)
        if robot_docking is not None:
            self.docking_steps = np.array(robot_docking, dtype=np.int32)  # (R,)
        else:
            self.docking_steps = np.full(R, default_dock, dtype=np.int32)  # (R,)
        self.docking_enabled = np.any(self.docking_steps > 0)

        # ── Stun config (forced STAY after taking collision damage) ────────
        default_stun = int(env_kwargs.get('stun_steps', 0))
        robot_stun = env_kwargs.get('robot_stun_steps', None)
        if robot_stun is not None:
            self.stun_steps = np.array(robot_stun, dtype=np.int32)  # (R,)
        else:
            self.stun_steps = np.full(R, default_stun, dtype=np.int32)  # (R,)
        self.stun_enabled = np.any(self.stun_steps > 0)

        # ── Per-robot attack power (damage dealt when this robot collides) ─
        attack_powers = env_kwargs.get('robot_attack_powers', None)
        if attack_powers is not None:
            self.attack_powers = np.array(attack_powers, dtype=np.float32)  # (R,)
        else:
            self.attack_powers = np.full(R, self.e_collision, dtype=np.float32)  # (R,)

        # ── Initial energies ───────────────────────────────────────────────
        init_e = env_kwargs.get('initial_energy', 100)
        robot_energies = env_kwargs.get('robot_energies', None) or [init_e] * R
        self.init_energies   = np.array(robot_energies, dtype=np.float32)   # (R,)
        self.global_max_energy = float(max(robot_energies))

        # ── Charger positions ──────────────────────────────────────────────
        configured = env_kwargs.get('charger_positions', None)
        defaults   = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
        raw        = configured if configured else defaults
        valid      = [(y, x) for y, x in raw
                      if not (y == -1 and x == -1)
                      and 0 <= y < n and 0 <= x < n]
        if not valid:
            raise ValueError("No valid charger positions")
        self.charger_positions = valid
        self.charger_yx        = np.array(valid, dtype=np.int32)   # (C, 2)
        self.C                 = len(valid)

        self.is_charger_grid = np.zeros((n, n), dtype=bool)
        for cy, cx in valid:
            self.is_charger_grid[cy, cx] = True

        # ── Dust config ────────────────────────────────────────────────────
        self.dust_enabled = bool(env_kwargs.get('dust_enabled', True))
        if self.dust_enabled:
            dust_max  = float(env_kwargs.get('dust_max',  10.0))
            dust_rate = float(env_kwargs.get('dust_rate',  0.5))
            self.dust_eps     = float(env_kwargs.get('dust_epsilon', 0.5))
            cdmr = float(env_kwargs.get('charger_dust_max_ratio',  0.3))
            cdrr = float(env_kwargs.get('charger_dust_rate_ratio', 0.5))
            self.dust_max_grid  = np.where(self.is_charger_grid,
                                           dust_max  * cdmr, dust_max ).astype(np.float32)
            self.dust_rate_grid = np.where(self.is_charger_grid,
                                           dust_rate * cdrr, dust_rate).astype(np.float32)
            self.dust_reward_scale = float(env_kwargs.get('dust_reward_scale', 0.05))

        # ── Agent type system (trait.md) ──────────────────────────────────
        self.agent_types_mode = env_kwargs.get('agent_types_mode', 'off')
        triangle_id = env_kwargs.get('triangle_agent_id', None)
        self.agent_types = np.zeros(R, dtype=np.float32)
        if triangle_id is not None and 0 <= triangle_id < R:
            self.agent_types[triangle_id] = 1.0

        # ── Observation dim ────────────────────────────────────────────────
        if self.agent_types_mode == 'observe':
            obs_dim = 3 + 1 + 4 + (R - 1) * 4 + 2 * self.C  # +1 self_type, +1 per other
        else:
            obs_dim = 3 + 4 + (R - 1) * 3 + 2 * self.C  # no type dims
        if self.dust_enabled:
            obs_dim += n * n
        self.obs_dim = obs_dim

        # Gymnasium compatibility stubs
        import gymnasium
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.agent_ids    = [f'robot_{i}' for i in range(R)]
        self.n_agents     = R

        # ── Random-start config ────────────────────────────────────────────
        self.random_start_robots = set(env_kwargs.get('random_start_robots', set()) or set())
        all_corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
        corner_q = list(all_corners)
        manual_starts = env_kwargs.get('robot_start_positions', {})
        self._fixed_starts: Dict[int, Tuple[int, int]] = {}
        for i, pos in manual_starts.items():
            self._fixed_starts[i] = pos
            if pos in corner_q:
                corner_q.remove(pos)
        for i in range(R):
            if i not in self._fixed_starts and i not in self.random_start_robots:
                self._fixed_starts[i] = corner_q.pop(0)

        # ── Thief spawn config ────────────────────────────────────────────
        self.thief_spawn = bool(env_kwargs.get('thief_spawn', False))

        # ── State arrays (allocated once) ──────────────────────────────────
        self.pos    = np.zeros((N, R, 2), dtype=np.int32)    # [y, x]
        self.energy = np.zeros((N, R),    dtype=np.float32)
        self.alive  = np.ones((N, R),     dtype=bool)
        self.steps  = np.zeros(N,         dtype=np.int32)

        # Per-step tracking
        self.prev_energy           = np.zeros((N, R), dtype=np.float32)
        self.active_collisions_with = np.zeros((N, R, R), dtype=np.int32)
        self.dust_collected         = np.zeros((N, R),    dtype=np.float32)
        self.died_this_step         = np.zeros((N, R),    dtype=bool)

        # Docking state: consecutive steps each robot has been on a charger
        self.docking_counter = np.zeros((N, R), dtype=np.int32)

        # Stun state: remaining forced-STAY steps after collision damage
        self.stun_counter = np.zeros((N, R), dtype=np.int32)
        self.stun_just_set = np.zeros((N, R), dtype=bool)  # prevent same-step decrement

        if self.dust_enabled:
            self.dust = np.zeros((N, n, n), dtype=np.float32)

        # Pre-allocated obs buffer
        self._obs_buf = np.zeros((N, obs_dim), dtype=np.float32)

        # episode_counts for compat
        self.episode_counts = np.zeros(N, dtype=np.int32)

    # ── Public interface ───────────────────────────────────────────────────

    def reset(self):
        """Reset all N envs."""
        for env_idx in range(self.N):
            self._reset_single(env_idx)
        return None, None   # caller doesn't use return value

    def get_observation(self, robot_id: int) -> np.ndarray:
        """Returns (N, obs_dim)."""
        return self._build_obs(robot_id)

    def step_single(self, robot_id: int, actions: np.ndarray, is_last_turn: bool = True):
        """
        Execute robot_id's action across all N envs simultaneously.

        Args:
            robot_id: which robot acts
            actions:  (N,) int32 — 0=UP,1=DOWN,2=LEFT,3=RIGHT,4=STAY
            is_last_turn: if False, skip decay and charging (for speed>1 sub-steps)

        Returns:
            next_obs:   (N, obs_dim)
            rewards:    (N,)  float32
            terminated: (N,)  bool
            truncated:  (N,)  bool  (always False here)
            infos:      list of N dicts
        """
        N, n, R = self.N, self.n, self.R
        rid = robot_id

        # ── Stun: force STAY if stunned (decrement in advance_step) ──────
        if self.stun_enabled:
            stunned = self.stun_counter[:, rid] > 0
            actions = actions.copy()
            actions[stunned] = _STAY

        # ── Reset per-step tracking ────────────────────────────────────────
        # NOTE: prev_energy and died_this_step are NOT reset here.
        # They carry over from the end of this robot's PREVIOUS step, so that
        # deaths caused by another robot's action (e.g., robot_0 kills robot_1)
        # are correctly captured when robot_1's own step runs next.
        self.active_collisions_with[:, rid, :] = 0
        self.dust_collected[:, rid]   = 0.0

        # ── Passive energy decay (only on last turn — not multiplied by speed) ──
        if self.e_decay > 0 and is_last_turn:
            alive_mask = self.alive[:, rid]
            self.energy[:, rid] -= alive_mask.astype(np.float32) * self.e_decay
            self.energy[:, rid] = np.maximum(self.energy[:, rid], 0.0)
            just_died = alive_mask & (self.energy[:, rid] <= 0)
            self.alive[just_died, rid] = False
            self.died_this_step[:, rid] |= just_died

        is_alive = self.alive[:, rid]               # (N,)
        is_move  = is_alive & (actions != _STAY)    # (N,)

        # ── Planned positions ──────────────────────────────────────────────
        dy = _ACT_DY[actions]   # (N,)
        dx = _ACT_DX[actions]   # (N,)
        py = self.pos[:, rid, 0] + dy   # (N,)
        px = self.pos[:, rid, 1] + dx   # (N,)

        # Move energy cost
        self.energy[:, rid] -= is_move.astype(np.float32) * self.e_move

        # ── Boundary ──────────────────────────────────────────────────────
        is_boundary = is_move & ((py < 0) | (py >= n) | (px < 0) | (px >= n))
        self.energy[:, rid] -= is_boundary.astype(np.float32) * self.e_boundary
        # Boundary robots stay; remove from can_move set
        can_move = is_move & ~is_boundary

        # ── Robot-robot collisions ─────────────────────────────────────────
        is_any_collision = np.zeros(N, dtype=bool)

        for j in range(R):
            if j == rid:
                continue

            hit_j = (can_move
                     & self.alive[:, j]
                     & (py == self.pos[:, j, 0])
                     & (px == self.pos[:, j, 1]))    # (N,)

            if not np.any(hit_j):
                continue

            is_any_collision |= hit_j

            # Knockback direction = same as attacker's move direction
            kby = self.pos[:, j, 0] + dy   # (N,)
            kbx = self.pos[:, j, 1] + dx   # (N,)

            kb_inbounds = (kby >= 0) & (kby < n) & (kbx >= 0) & (kbx < n)

            # Check if knockback cell is blocked by another robot
            kb_blocked = np.zeros(N, dtype=bool)
            for k in range(R):
                if k == rid or k == j:
                    continue
                kb_blocked |= (self.alive[:, k]
                               & (kby == self.pos[:, k, 0])
                               & (kbx == self.pos[:, k, 1]))

            can_push    = hit_j & kb_inbounds & ~kb_blocked
            stationary  = hit_j & ~can_push

            # knockback_success: attacker moves in, victim pushed out
            self.pos[can_push, rid, 0] = py[can_push]
            self.pos[can_push, rid, 1] = px[can_push]
            self.pos[can_push, j,   0] = kby[can_push]
            self.pos[can_push, j,   1] = kbx[can_push]
            self.energy[can_push, j]  -= self.attack_powers[rid]
            self.active_collisions_with[can_push, rid, j] += 1
            # Knockback resets victim's docking counter (pushed off charger)
            if self.docking_enabled:
                self.docking_counter[can_push, j] = 0
            # Stun victim on knockback
            if self.stun_enabled:
                self.stun_counter[can_push, j] = self.stun_steps[j]
                self.stun_just_set[can_push, j] = True

            # stationary_blocked: attacker stays, victim takes damage
            self.energy[stationary, j] -= self.attack_powers[rid]
            self.active_collisions_with[stationary, rid, j] += 1
            # Reset victim's docking counter on stationary hit too
            if self.docking_enabled:
                self.docking_counter[stationary, j] = 0
            # Stun victim on stationary hit too
            if self.stun_enabled:
                self.stun_counter[stationary, j] = self.stun_steps[j]
                self.stun_just_set[stationary, j] = True

        # ── Normal move (no collision) ─────────────────────────────────────
        free_move = can_move & ~is_any_collision
        self.pos[free_move, rid, 0] = py[free_move]
        self.pos[free_move, rid, 1] = px[free_move]

        # ── Energy floor & death (all robots, victim may have died) ────────
        for r in range(R):
            was_alive = self.alive[:, r].copy()
            self.energy[:, r] = np.maximum(self.energy[:, r], 0.0)
            just_died = was_alive & (self.energy[:, r] <= 0)
            self.alive[just_died, r] = False
            # Track death for ALL robots, not just rid.
            # When robot_j is killed during robot_rid's step, died_this_step[:,j]
            # stays True until robot_j's own next step applies the penalty.
            self.died_this_step[:, r] |= just_died

        # ── Docking counter update (only on last turn) ────────────────────
        if is_last_turn:
            ry = self.pos[:, rid, 0]    # (N,) current y
            rx = self.pos[:, rid, 1]    # (N,) current x

            # Check if robot is on any charger (and not stunned)
            cr = self.charger_range
            on_any_charger = np.zeros(N, dtype=bool)
            for cy, cx in self.charger_positions:
                on_any_charger |= (self.alive[:, rid]
                                   & (np.abs(ry - cy) <= cr)
                                   & (np.abs(rx - cx) <= cr))
            # Stunned robots cannot accumulate docking progress
            if self.stun_enabled:
                on_any_charger = on_any_charger & (self.stun_counter[:, rid] == 0)

            # Increment counter for robots on charger, reset for those not
            self.docking_counter[:, rid] = np.where(
                on_any_charger,
                self.docking_counter[:, rid] + 1,
                0)

        # ── Charging (only on last turn — not multiplied by speed) ────────
        if is_last_turn:
            cr = self.charger_range
            for cy, cx in self.charger_positions:
                in_range = (self.alive[:, rid]
                            & (np.abs(ry - cy) <= cr)
                            & (np.abs(rx - cx) <= cr))   # (N,)
                if not np.any(in_range):
                    continue

                # Docking gate: only charge if docking counter meets threshold
                if self.docking_enabled:
                    in_range = in_range & (self.docking_counter[:, rid] >= self.docking_steps[rid])

                if not np.any(in_range):
                    continue

                if self.exclusive_charging:
                    # Charge only when sole occupant in this charger's range
                    n_in_range = np.zeros(N, dtype=np.int32)
                    for k in range(R):
                        n_in_range += (self.alive[:, k]
                                       & (np.abs(self.pos[:, k, 0] - cy) <= cr)
                                       & (np.abs(self.pos[:, k, 1] - cx) <= cr)).astype(np.int32)
                    can_charge = in_range & (n_in_range <= 1)
                    self.energy[can_charge, rid] += self.e_charge
                else:
                    # Charge divided among all robots in charger range
                    count = np.zeros(N, dtype=np.float32)
                    for k in range(R):
                        count += (self.alive[:, k]
                                  & (np.abs(self.pos[:, k, 0] - cy) <= cr)
                                  & (np.abs(self.pos[:, k, 1] - cx) <= cr)).astype(np.float32)
                    charge_amt = self.e_charge / np.maximum(count, 1.0)   # (N,)
                    # Heterotype penalty: mixed types in same charger range
                    if self.heterotype_charge_mode == 'local-penalty':
                        has_circle = np.zeros(N, dtype=bool)
                        has_triangle = np.zeros(N, dtype=bool)
                        for k in range(R):
                            k_in_range = (self.alive[:, k]
                                          & (np.abs(self.pos[:, k, 0] - cy) <= cr)
                                          & (np.abs(self.pos[:, k, 1] - cx) <= cr))
                            if self.agent_types[k] == 0:
                                has_circle |= k_in_range
                            else:
                                has_triangle |= k_in_range
                        is_mixed = has_circle & has_triangle  # (N,)
                        charge_amt = np.where(is_mixed, charge_amt * self.heterotype_charge_factor, charge_amt)
                    self.energy[in_range, rid] += charge_amt[in_range]

        # ── Dust harvest ──────────────────────────────────────────────────
        if self.dust_enabled:
            alive_mask   = self.alive[:, rid]           # (N,)
            env_idx_live = np.where(alive_mask)[0]
            if len(env_idx_live) > 0:
                ys = self.pos[env_idx_live, rid, 0]
                xs = self.pos[env_idx_live, rid, 1]
                harvested = self.dust[env_idx_live, ys, xs]
                self.dust_collected[env_idx_live, rid] = harvested
                self.dust[env_idx_live, ys, xs] = 0.0

        # ── Rewards (computed BEFORE energy cap, so charging at full energy
        #    still produces a positive delta-energy reward) ────────────────
        if self.reward_mode == 'hp-ratio':
            # HP-ratio reward: (current_energy / energy_cap) * alpha per step
            cap = self.energy_cap if self.energy_cap is not None else 100.0
            rewards = (self.energy[:, rid] / cap).astype(np.float32) * self.reward_alpha
            rewards -= self.died_this_step[:, rid].astype(np.float32) * 100.0
        else:
            # Delta-energy reward: immediate signal for energy changes
            energy_delta = self.energy[:, rid] - self.prev_energy[:, rid]
            rewards = energy_delta.astype(np.float32) * self.reward_alpha
            rewards -= self.died_this_step[:, rid].astype(np.float32) * 100.0

        # ── Energy cap (AFTER reward computation) ────────────────────────
        # Per-robot cap: use global energy_cap if set, otherwise cap at each
        # robot's initial energy to prevent unbounded overcharging.
        if self.energy_cap is not None:
            self.energy[:, rid] = np.minimum(self.energy[:, rid], self.energy_cap)
        else:
            self.energy[:, rid] = np.minimum(self.energy[:, rid], self.init_energies[rid])

        # ── Outputs ───────────────────────────────────────────────────────
        terminated = ~self.alive[:, rid]                # (N,) bool
        truncated  = np.zeros(N, dtype=bool)
        next_obs   = self._build_obs(rid)               # (N, obs_dim)
        infos      = None  # Skipped: training loop reads collisions directly from numpy arrays

        # ── Save state for next cycle (AFTER reward computation & cap) ───
        # prev_energy is saved HERE (end of step) so that if this robot is
        # killed by another robot before its next turn, the correct
        # pre-death energy is used for the delta calculation.
        self.died_this_step[:, rid] = False
        self.prev_energy[:, rid]    = self.energy[:, rid]

        return next_obs, rewards, terminated, truncated, infos

    def advance_step(self) -> Tuple[np.ndarray, List[int]]:
        """
        Advance step counter, update dust, detect & auto-reset done envs.

        Returns:
            done_mask: (N,) bool
            done_envs: list of env indices that finished this step
        """
        self.steps += 1

        # ── Stun: decrement counters once per game step ────────────────
        #    Skip decrement for stuns set THIS step (prevent off-by-one)
        if self.stun_enabled:
            can_decrement = ~self.stun_just_set
            self.stun_counter = np.where(
                can_decrement,
                np.maximum(self.stun_counter - 1, 0),
                self.stun_counter,
            )
            self.stun_just_set[:] = False

        # Note: passive energy decay is now applied per-robot inside step_single()
        # so decay deaths are properly captured in rewards.

        if self.dust_enabled:
            self._update_dust()

        max_steps  = self.steps >= self.n_steps         # (N,) bool
        all_dead   = ~self.alive.any(axis=1)            # (N,) bool
        done_mask  = max_steps | all_dead

        done_envs = list(np.where(done_mask)[0])
        for env_idx in done_envs:
            self.episode_counts[env_idx] += 1
            self._reset_single(env_idx)

        return done_mask, done_envs

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_obs(self, robot_id: int) -> np.ndarray:
        """Vectorized obs for all N envs. Returns (N, obs_dim)."""
        N, n, R, C = self.N, self.n, self.R, self.C
        rid  = robot_id
        inv  = 1.0 / (n - 1) if n > 1 else 1.0

        # Dynamic energy normalization (per env)
        energies_alive = np.where(self.alive, self.energy, 0.0)    # (N, R)
        actual_max     = energies_alive.max(axis=1)                 # (N,)
        global_max     = np.maximum(actual_max, self.global_max_energy)  # (N,)
        inv_max        = 1.0 / global_max                           # (N,)

        obs = self._obs_buf   # reuse pre-allocated buffer
        col = 0

        # 1. Self position (x, y) normalized [0,1]
        obs[:, col  ] = self.pos[:, rid, 1] * inv   # x
        obs[:, col+1] = self.pos[:, rid, 0] * inv   # y
        col += 2

        # 2. Self energy
        obs[:, col] = self.energy[:, rid] * inv_max
        col += 1

        # 3. Self type (only when mode=observe)
        if self.agent_types_mode == 'observe':
            obs[:, col] = self.agent_types[rid]
            col += 1

        # 4. Wall indicators
        obs[:, col  ] = (self.pos[:, rid, 0] == 0    ).astype(np.float32)  # wall_up
        obs[:, col+1] = (self.pos[:, rid, 0] == n - 1).astype(np.float32)  # wall_down
        obs[:, col+2] = (self.pos[:, rid, 1] == 0    ).astype(np.float32)  # wall_left
        obs[:, col+3] = (self.pos[:, rid, 1] == n - 1).astype(np.float32)  # wall_right
        col += 4

        # 5. Other robots (dx, dy, energy[, type])
        for j in range(R):
            if j == rid:
                continue
            obs[:, col  ] = (self.pos[:, j, 1] - self.pos[:, rid, 1]).astype(np.float32) * inv
            obs[:, col+1] = (self.pos[:, j, 0] - self.pos[:, rid, 0]).astype(np.float32) * inv
            obs[:, col+2] = self.energy[:, j] * inv_max
            col += 3
            if self.agent_types_mode == 'observe':
                obs[:, col] = self.agent_types[j]
                col += 1

        # 6. Charger offsets (dx, dy)
        for cy, cx in self.charger_positions:
            obs[:, col  ] = (cx - self.pos[:, rid, 1]).astype(np.float32) * inv
            obs[:, col+1] = (cy - self.pos[:, rid, 0]).astype(np.float32) * inv
            col += 2

        # 7. Dust grid (row-major y→x, normalized per cell)
        if self.dust_enabled:
            dust_norm = self.dust / self.dust_max_grid[np.newaxis, :, :]   # (N, n, n)
            obs[:, col:col + n * n] = dust_norm.reshape(N, n * n)
            col += n * n

        return obs.copy()  # Must copy: buffer is shared across calls, callers hold references across _build_obs invocations

    def _build_infos(self, robot_id: int) -> List[Dict]:
        """Build list of N info dicts. Only includes fields used by training loop."""
        rid = robot_id
        R   = self.R
        infos = []
        for e in range(self.N):
            d = {f'active_collisions_with_{j}': int(self.active_collisions_with[e, rid, j])
                 for j in range(R) if j != rid}
            infos.append(d)
        return infos

    def _reset_single(self, env_idx: int):
        """Reset one env in-place."""
        n = self.n
        if self.thief_spawn:
            # Thief scenario: weak (robot 1) near charger, strong (robot 0) far
            cy, cx = self.charger_positions[0]
            # Weak (robot 1): random adjacent to charger (4-connected)
            adjacent = [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]
            adjacent = [(y, x) for y, x in adjacent if 0 <= y < n and 0 <= x < n]
            wy, wx = random.choice(adjacent)
            self.pos[env_idx, 1, 0] = wy
            self.pos[env_idx, 1, 1] = wx
            # Strong (robot 0): random, not on/adjacent to charger
            forbidden = set(adjacent + [(cy, cx)])
            candidates = [(r, c) for r in range(n) for c in range(n)
                           if (r, c) not in forbidden and (r, c) != (wy, wx)]
            sy, sx = random.choice(candidates)
            self.pos[env_idx, 0, 0] = sy
            self.pos[env_idx, 0, 1] = sx
        else:
            for i, (y, x) in self._fixed_starts.items():
                self.pos[env_idx, i, 0] = y
                self.pos[env_idx, i, 1] = x

            if self.random_start_robots:
                all_cells = [(r, c) for r in range(n) for c in range(n)]
                occupied  = {(self.pos[env_idx, i, 0], self.pos[env_idx, i, 1])
                             for i in range(self.R) if i not in self.random_start_robots}
                for i in self.random_start_robots:
                    candidates = [p for p in all_cells if p not in occupied]
                    y, x = random.choice(candidates)
                    self.pos[env_idx, i, 0] = y
                    self.pos[env_idx, i, 1] = x
                    occupied.add((y, x))

        self.energy[env_idx]      = self.init_energies
        self.alive[env_idx]       = True
        self.steps[env_idx]       = 0
        self.prev_energy[env_idx] = self.init_energies
        self.active_collisions_with[env_idx] = 0
        self.died_this_step[env_idx]         = False
        self.dust_collected[env_idx]         = 0.0
        self.docking_counter[env_idx]        = 0
        self.stun_counter[env_idx]           = 0
        self.stun_just_set[env_idx]          = False
        if self.dust_enabled:
            self.dust[env_idx] = 0.0

    def _update_dust(self):
        """Vectorized sigmoid dust growth: dD = rate*(D+eps)*(1 - D/Dmax)"""
        growth = (self.dust_rate_grid[np.newaxis]
                  * (self.dust + self.dust_eps)
                  * (1.0 - self.dust / self.dust_max_grid[np.newaxis]))
        self.dust = np.clip(
            self.dust + growth, 0.0, self.dust_max_grid[np.newaxis]
        ).astype(np.float32)
