#!/usr/bin/env python3
"""
Quick eval: check if robot_1 has learned charger-seeking behavior.
Metrics:
  - charger_occupancy: fraction of steps r1 is ON the charger cell
  - avg_dist_to_charger: average Manhattan distance from r1 to charger
  - charger_visit_rate: fraction of episodes r1 ever reaches the charger
"""

import argparse
import os
import random
import numpy as np
import torch

from batch_env import BatchRobotVacuumEnv
from dqn import build_network


CONFIGS = {
    "v1_r2": {
        "model_dir": "./models/stun5_joint_v1_r2",
        "robot_0_epsilon_schedule": "sigmoid",
        "dueling": False, "noisy": False, "c51": False,
    },
    "v2_r2": {
        "model_dir": "./models/stun5_joint_v2_r2",
        "robot_0_epsilon_schedule": "sigmoid",
        "dueling": False, "noisy": False, "c51": False,
    },
    "v3_r2": {
        "model_dir": "./models/stun5_joint_v3_r2",
        "robot_0_epsilon_schedule": "sigmoid",
        "dueling": False, "noisy": False, "c51": False,
    },
    "v4_r2": {
        "model_dir": "./models/stun5_joint_v4_r2",
        "robot_0_epsilon_schedule": "sigmoid",
        "dueling": True, "noisy": True, "c51": True,
    },
}

ENV_KWARGS = {
    "n": 5,
    "num_robots": 2,
    "n_steps": 500,
    "e_move": 1.0,
    "e_charge": 8.0,
    "e_collision": 30.0,
    "e_boundary": 0.0,
    "e_decay": 0.5,
    "exclusive_charging": True,
    "charger_range": 0,
    "charger_positions": [(2, 2)],
    "dust_enabled": False,
    "initial_energy": 100,
    "robot_energies": [100, 100],
    "robot_speeds": [2, 1],
    "robot_attack_powers": [30.0, 2.0],
    "robot_docking_steps": [0, 0],
    "robot_stun_steps": [5, 1],
    "random_start_robots": {0, 1},
    "reward_mode": "delta-energy",
    "reward_alpha": 0.05,
    "agent_types_mode": "off",
}

CHARGER_POS = (2, 2)
N_EPISODES = 100
MAX_STEPS = 500


def find_latest_checkpoint(model_dir):
    eps = []
    for name in os.listdir(model_dir):
        if not name.startswith("episode_"):
            continue
        try:
            ep = int(name.split("_")[-1])
        except ValueError:
            continue
        if os.path.exists(os.path.join(model_dir, name, "robot_1.pt")):
            eps.append(ep)
    return max(eps) if eps else None


def load_net(path, obs_dim, dueling=False, noisy=False, c51=False):
    net = build_network(num_actions=5, input_dim=obs_dim, dueling=dueling, noisy=noisy, c51=c51)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def seek_charger_action_heuristic(env):
    """Heuristic baseline: move toward charger."""
    obs = env.get_observation(1)
    charger_col = 10
    cdx, cdy = obs[0, charger_col], obs[0, charger_col + 1]
    if abs(cdx) < 0.01 and abs(cdy) < 0.01:
        return 4  # stay (already at charger)
    if abs(cdx) >= abs(cdy):
        return 3 if cdx > 0.01 else (2 if cdx < -0.01 else 4)
    return 1 if cdy > 0.01 else (0 if cdy < -0.01 else 4)


def eval_r1(name, cfg):
    model_dir = cfg["model_dir"]
    ep = find_latest_checkpoint(model_dir)
    if ep is None:
        print(f"{name}: No checkpoint with robot_1.pt found.")
        return

    ckpt_path = os.path.join(model_dir, f"episode_{ep}", "robot_1.pt")
    env = BatchRobotVacuumEnv(num_envs=1, env_kwargs=ENV_KWARGS)
    r1_net = load_net(ckpt_path, env.obs_dim, cfg["dueling"], cfg["noisy"], cfg["c51"])

    # Also load r0 net (greedy) to create realistic opponent pressure
    r0_path = os.path.join(model_dir, f"episode_{ep}", "robot_0.pt")
    r0_net = load_net(r0_path, env.obs_dim, cfg["dueling"], cfg["noisy"], cfg["c51"])

    total_steps = 0
    r1_charger_steps = 0
    r1_total_dist = 0
    r1_visit_count = 0   # episodes where r1 reaches charger
    r1_alive_episodes = 0
    r1_died_count = 0

    cy, cx = CHARGER_POS

    for ep_i in range(N_EPISODES):
        env.reset()
        np.random.seed(ep_i * 13 + 7)
        random.seed(ep_i * 13 + 7)

        visited_charger = False

        for _ in range(MAX_STEPS):
            if not env.alive[0].any():
                break

            robot_order = [0, 1]
            random.shuffle(robot_order)

            for robot_id in robot_order:
                speed = [2, 1][robot_id]
                for turn_idx in range(speed):
                    is_last = (turn_idx == speed - 1)
                    if not env.alive[0, robot_id]:
                        continue

                    if robot_id == 0:
                        obs = env.get_observation(0)
                        with torch.no_grad():
                            q = r0_net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())
                    else:
                        obs = env.get_observation(1)
                        with torch.no_grad():
                            q = r1_net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())

                    env.step_single(robot_id, np.array([action], dtype=np.int32), is_last_turn=is_last)

            # After both robots moved: record r1 stats
            if env.alive[0, 1]:
                r1y = int(env.pos[0, 1, 0])
                r1x = int(env.pos[0, 1, 1])
                dist = abs(r1y - cy) + abs(r1x - cx)
                r1_total_dist += dist
                total_steps += 1
                if r1y == cy and r1x == cx:
                    r1_charger_steps += 1
                    visited_charger = True

            done_mask, _ = env.advance_step()
            if done_mask[0]:
                break

        r1_alive_episodes += 1
        if visited_charger:
            r1_visit_count += 1
        if not env.alive[0, 1]:
            r1_died_count += 1

    occupancy = r1_charger_steps / max(total_steps, 1)
    avg_dist = r1_total_dist / max(total_steps, 1)
    visit_rate = r1_visit_count / N_EPISODES
    death_rate = r1_died_count / N_EPISODES

    print(f"\n{'='*55}")
    print(f"  {name}  (checkpoint: {ep:,})")
    print(f"{'='*55}")
    print(f"  Charger occupancy:   {occupancy:.1%}  ({r1_charger_steps}/{total_steps} steps on charger)")
    print(f"  Avg dist to charger: {avg_dist:.2f} cells")
    print(f"  Charger visit rate:  {visit_rate:.1%}  (episodes r1 reached charger)")
    print(f"  R1 death rate:       {death_rate:.1%}")

    # Heuristic baseline for reference (quick, 10 episodes)
    env2 = BatchRobotVacuumEnv(num_envs=1, env_kwargs=ENV_KWARGS)
    h_charger = 0; h_total = 0
    for ep_i in range(20):
        env2.reset()
        np.random.seed(ep_i * 13 + 7)
        random.seed(ep_i * 13 + 7)
        for _ in range(MAX_STEPS):
            if not env2.alive[0].any(): break
            robot_order = [0, 1]; random.shuffle(robot_order)
            for robot_id in robot_order:
                speed = [2, 1][robot_id]
                for turn_idx in range(speed):
                    is_last = (turn_idx == speed - 1)
                    if not env2.alive[0, robot_id]: continue
                    if robot_id == 0:
                        obs = env2.get_observation(0)
                        with torch.no_grad():
                            q = r0_net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())
                    else:
                        action = seek_charger_action_heuristic(env2)
                    env2.step_single(robot_id, np.array([action], dtype=np.int32), is_last_turn=is_last)
            if env2.alive[0, 1]:
                r1y = int(env2.pos[0, 1, 0]); r1x = int(env2.pos[0, 1, 1])
                if r1y == cy and r1x == cx: h_charger += 1
                h_total += 1
            done_mask, _ = env2.advance_step()
            if done_mask[0]: break
    heuristic_occ = h_charger / max(h_total, 1)
    print(f"  [Heuristic baseline occupancy: {heuristic_occ:.1%}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["v1_r2", "v2_r2", "v3_r2", "v4_r2"])
    args = parser.parse_args()

    for name in args.variants:
        if name in CONFIGS:
            eval_r1(name, CONFIGS[name])
        else:
            print(f"Unknown variant: {name}")
