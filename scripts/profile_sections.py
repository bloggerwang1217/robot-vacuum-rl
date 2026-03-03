"""
真實訓練迴圈各區段計時腳本。

直接複製 hot loop，在每個區段插 perf_counter，跑 N 個 global_step 後印出各段累計耗時。

用法：
  source ~/.venv/bin/activate
  cd ~/robot-vacuum-rl
  python scripts/profile_sections.py
"""

import sys, os
# 讓 import train_dqn_vec 能找到同層的模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

# ── patch sys.argv 讓 train_dqn_vec.main() 裡的 parse_args() 吃到我們的參數 ──
sys.argv = [
    "train_dqn_vec.py",
    "--env-n", "6",
    "--num-robots", "2",
    "--robot-0-energy", "20", "--robot-1-energy", "100",
    "--robot-0-speed", "1",  "--robot-1-speed", "2",
    "--charger-positions", "3,3",
    "--exclusive-charging", "--no-dust",
    "--e-move", "0", "--e-charge", "10", "--e-collision", "30", "--e-boundary", "0",
    "--n-step", "20", "--gamma", "0.999",
    "--max-episode-steps", "1000",
    "--num-episodes", "99999",
    "--num-envs", "256",
    "--wandb-mode", "disabled",
    "--no-eval-after-training",
    "--safe-random-robots", "0",
    "--num-workers", "1",
    "--batch-env",
    "--use-torch-compile",
]

import train_dqn_vec as T

PROFILE_STEPS = 500


def make_trainer():
    import argparse
    # 直接呼叫 train_dqn_vec.main() 裡的 parser 建構邏輯
    # 因為 sys.argv 已 patch，直接建立 parser 並 parse_args() 即可
    # 我們手動複製一份 parser（與 main() 完全相同的 add_argument 清單）
    # 最省事：直接 import 後 exec main() source 到 parse_args 那行
    import inspect, textwrap
    src = inspect.getsource(T.main)
    lines = src.splitlines()
    parser_lines = []
    in_body = False
    for line in lines:
        # main() body 從第一個非 def 行開始
        if not in_body:
            if line.strip().startswith('parser = argparse'):
                in_body = True
        if in_body:
            parser_lines.append(line)
            if line.strip().startswith('args = parser.parse_args()'):
                break

    code = textwrap.dedent('\n'.join(parser_lines))
    ns = {'argparse': argparse}
    exec(code, ns)
    args = ns['args']

    trainer = T.VectorizedMultiAgentTrainer(args)
    trainer.env.reset()
    return trainer


def run_profiled(trainer):
    t = trainer

    timers = {
        "get_obs":       0.0,
        "select_action": 0.0,
        "env_step":      0.0,
        "bookkeeping":   0.0,
        "remember":      0.0,
        "train_step":    0.0,
        "advance_step":  0.0,
        "other":         0.0,
    }

    env_terminations = np.zeros((t.num_envs, t.n_agents), dtype=bool)
    global_step = 0

    while global_step < PROFILE_STEPS:

        for robot_id in range(t.n_agents):
            n_turns     = t.robot_speeds[robot_id]
            is_scripted = robot_id in t.scripted_robots
            is_random   = robot_id in t.random_robots
            is_safe_rnd = robot_id in t.safe_random_robots
            is_flee     = robot_id in t.flee_robots
            is_learning = not (is_scripted or is_random or is_safe_rnd or is_flee)

            for _ in range(n_turns):

                # ── get_obs ────────────────────────────────────────────────
                obs = None
                if not is_scripted and not is_random:
                    t0 = time.perf_counter()
                    obs = t.env.get_observation(robot_id)
                    timers["get_obs"] += time.perf_counter() - t0

                # ── select_action ──────────────────────────────────────────
                t0 = time.perf_counter()
                if is_scripted:
                    actions = np.full(t.num_envs, 4, dtype=np.int32)
                elif is_random:
                    actions = np.random.randint(0, 5, size=t.num_envs).astype(np.int32)
                elif is_safe_rnd:
                    actions = t._safe_random_actions(robot_id, obs)
                elif is_flee:
                    actions = t._flee_actions(robot_id, obs)
                else:
                    actions = t.select_actions_for_robot(robot_id, obs)
                timers["select_action"] += time.perf_counter() - t0

                # ── env step ───────────────────────────────────────────────
                t0 = time.perf_counter()
                next_obs, rewards, terminated, truncated, infos = t.env.step_single(robot_id, actions)
                timers["env_step"] += time.perf_counter() - t0

                # ── bookkeeping ────────────────────────────────────────────
                t0 = time.perf_counter()
                from batch_env import BatchRobotVacuumEnv as _BEnv
                if t.use_vec_env and isinstance(t.env, _BEnv):
                    t._episode_rewards[:, robot_id] += rewards
                    t._episode_active_collisions[:, robot_id] = \
                        t.env.active_collisions_with[:, robot_id, :].sum(axis=1)
                    env_terminations[:, robot_id] |= terminated
                    if robot_id == 0:
                        new_deaths = terminated & (t._robot0_death_step == -1)
                        t._robot0_death_step[new_deaths] = t._episode_steps[new_deaths] + 1
                else:
                    for env_idx in range(t.num_envs):
                        t._episode_rewards[env_idx, robot_id] += rewards[env_idx]
                        t._episode_active_collisions[env_idx, robot_id] = sum(
                            infos[env_idx].get(f'active_collisions_with_{j}', 0)
                            for j in range(t.n_agents)
                        )
                        if terminated[env_idx]:
                            env_terminations[env_idx, robot_id] = True
                            if robot_id == 0 and t._robot0_death_step[env_idx] == -1:
                                t._robot0_death_step[env_idx] = t._episode_steps[env_idx] + 1
                timers["bookkeeping"] += time.perf_counter() - t0

                # ── remember ───────────────────────────────────────────────
                if is_learning:
                    t0 = time.perf_counter()
                    if t.use_vec_env:
                        t.remember_batch(obs, actions, rewards, next_obs, terminated, robot_id)
                    else:
                        t.remember(obs[0], actions[0], rewards[0], next_obs[0], terminated[0], 0, robot_id)
                    timers["remember"] += time.perf_counter() - t0

        # ── advance_step ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        done_mask, done_envs = t.env.advance_step()
        timers["advance_step"] += time.perf_counter() - t0

        for env_idx in range(t.num_envs):
            t._episode_steps[env_idx] += 1

        # ── train_step ────────────────────────────────────────────────────
        if global_step % t.train_frequency == 0:
            t0 = time.perf_counter()
            for agent_id in t.agent_ids:
                mem = t.memories[agent_id]
                if len(mem) >= max(t.args.replay_start_size, t.args.batch_size):
                    batch = t.sample_batch(agent_id)
                    if batch is not None:
                        t.agents[agent_id].train_step(batch)
            timers["train_step"] += time.perf_counter() - t0

        # ── target update + episode reset (折入 other) ────────────────────
        t0 = time.perf_counter()
        if global_step % t.args.target_update_frequency == 0:
            for agent in t.agents.values():
                agent.update_target_network()
        for env_idx in done_envs:
            t.total_episodes += 1
            t._episode_rewards[env_idx].fill(0)
            t._episode_steps[env_idx] = 0
            t._episode_active_collisions[env_idx].fill(0)
            t._robot0_death_step[env_idx] = -1
            env_terminations[env_idx, :] = False
        timers["other"] += time.perf_counter() - t0

        global_step += 1

    # ── 印結果 ────────────────────────────────────────────────────────────────
    total = sum(timers.values())
    print(f"\n{'='*57}")
    print(f"  Timing: {PROFILE_STEPS} steps × {t.num_envs} envs, n-step={t.n_step}")
    print(f"{'='*57}")
    print(f"  {'Section':<18}  {'ms/step':>9}  {'total(s)':>9}  {'%':>6}")
    print(f"  {'-'*50}")
    for name, elapsed in sorted(timers.items(), key=lambda x: -x[1]):
        ms_per_step = elapsed / PROFILE_STEPS * 1000
        pct = elapsed / total * 100 if total > 0 else 0
        print(f"  {name:<18}  {ms_per_step:>9.3f}  {elapsed:>9.3f}s  {pct:>5.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<18}  {total/PROFILE_STEPS*1000:>9.3f}  {total:>9.3f}s  100.0%")
    print(f"{'='*57}\n")


if __name__ == "__main__":
    print("Building trainer...")
    trainer = make_trainer()

    print("Warming up GPU (20 steps)...")
    for _ in range(20):
        for rid in range(trainer.n_agents):
            obs = trainer.env.get_observation(rid)
            if rid in trainer.safe_random_robots:
                actions = trainer._safe_random_actions(rid, obs)
            else:
                actions = trainer.select_actions_for_robot(rid, obs)
            trainer.env.step_single(rid, actions)
        trainer.env.advance_step()

    print(f"Profiling {PROFILE_STEPS} steps...\n")
    run_profiled(trainer)
