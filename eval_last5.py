#!/usr/bin/env python3
"""
Eval last N checkpoints for v1_r2, v2_r2, v3_r2.
Uses exact training params + pure greedy + r1-policy=model.
"""

import os, random, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from batch_env import BatchRobotVacuumEnv
from dqn import build_network

# ── Training params (same for v1-v3 r2) ─────────────────────────────────────
ENV_KWARGS = dict(
    n=5, num_robots=2, n_steps=500,
    e_move=1.0, e_charge=8.0, e_collision=30.0, e_boundary=0.0, e_decay=0.5,
    exclusive_charging=True, charger_range=0,
    charger_positions=[(2, 2)],
    dust_enabled=False,
    initial_energy=100,
    robot_energies=[100, 100],
    robot_speeds=[2, 1],
    robot_attack_powers=[30.0, 2.0],
    robot_docking_steps=[0, 0],
    robot_stun_steps=[5, 1],
    random_start_robots={0, 1},
    reward_mode="delta-energy",
    reward_alpha=0.05,
    agent_types_mode="off",
)

CHARGER_YX = {(2, 2)}
EPISODES_PER_CKPT = 100
MAX_STEPS = 500
N_LAST = 5


def find_last_n(model_dir, n):
    eps = []
    for name in os.listdir(model_dir):
        if not name.startswith("episode_"):
            continue
        try:
            ep = int(name.split("_")[-1])
        except ValueError:
            continue
        if os.path.exists(os.path.join(model_dir, name, "robot_0.pt")):
            eps.append(ep)
    eps.sort()
    return eps[-n:]


def load_net(path, obs_dim, dueling=False, noisy=False, c51=False):
    net = build_network(num_actions=5, input_dim=obs_dim,
                        dueling=dueling, noisy=noisy, c51=c51)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    return net


def eval_ckpt(env, ckpt_dir, obs_dim):
    r0_net = load_net(os.path.join(ckpt_dir, "robot_0.pt"), obs_dim)
    r1_path = os.path.join(ckpt_dir, "robot_1.pt")
    r1_net  = load_net(r1_path, obs_dim) if os.path.exists(r1_path) else None

    total_steps = 0
    total_attack_steps = 0   # steps r0 lands >=1 hit
    total_hits = 0
    total_double = 0
    total_single = 0
    total_offensive = 0      # hits where r0 NOT on charger
    r1_kill = 0

    # charger occupancy stats
    r1_on_charger = 0
    r1_alive_steps = 0

    for ep_i in range(EPISODES_PER_CKPT):
        env.reset()
        np.random.seed(ep_i * 37 + 1000)
        random.seed(ep_i * 37 + 1000)

        for _ in range(MAX_STEPS):
            if not env.alive[0].any():
                break
            total_steps += 1

            robot_order = [0, 1]
            random.shuffle(robot_order)

            for robot_id in robot_order:
                speed = [2, 1][robot_id]
                hits_step = 0
                off_step  = 0

                for turn_idx in range(speed):
                    is_last = (turn_idx == speed - 1)
                    if not env.alive[0, robot_id]:
                        continue

                    r1_e_before    = float(env.energy[0, 1])
                    r1_alive_before = bool(env.alive[0, 1])

                    net = r0_net if robot_id == 0 else r1_net
                    if net is None:
                        action = random.randint(0, 4)
                    else:
                        obs = env.get_observation(robot_id)
                        with torch.no_grad():
                            q = net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())

                    env.step_single(robot_id,
                                    np.array([action], dtype=np.int32),
                                    is_last_turn=is_last)

                    if robot_id == 0 and r1_alive_before:
                        r1_e_after = float(env.energy[0, 1])
                        if r1_e_after < r1_e_before - 1.0:
                            hits_step += 1
                            r0y = int(env.pos[0, 0, 0])
                            r0x = int(env.pos[0, 0, 1])
                            if (r0y, r0x) not in CHARGER_YX:
                                off_step += 1

                if robot_id == 0:
                    total_hits      += hits_step
                    total_offensive += off_step
                    if hits_step >= 1:
                        total_attack_steps += 1
                    if hits_step == 2:
                        total_double += 1
                    elif hits_step == 1:
                        total_single += 1

            # r1 position after full step
            if env.alive[0, 1]:
                r1y = int(env.pos[0, 1, 0])
                r1x = int(env.pos[0, 1, 1])
                r1_alive_steps += 1
                if (r1y, r1x) in CHARGER_YX:
                    r1_on_charger += 1

            done_mask, _ = env.advance_step()
            if done_mask[0]:
                break

        if not env.alive[0, 1]:
            r1_kill += 1

    hit_total = total_double + total_single
    return dict(
        attack_rate   = total_attack_steps / max(total_steps, 1),
        offensive_rate= total_offensive    / max(total_hits, 1),
        double_rate   = total_double       / max(hit_total, 1),
        single_rate   = total_single       / max(hit_total, 1),
        kill_rate     = r1_kill            / EPISODES_PER_CKPT,
        r1_occ        = r1_on_charger      / max(r1_alive_steps, 1),
    )


def run(name, model_dir):
    checkpoints = find_last_n(model_dir, N_LAST)
    if not checkpoints:
        print(f"{name}: no checkpoints found"); return

    env = BatchRobotVacuumEnv(num_envs=1, env_kwargs=ENV_KWARGS)

    print(f"\n{'='*70}")
    print(f"  {name}  —  last {N_LAST} checkpoints  (greedy, r1=model)")
    print(f"{'='*70}")
    print(f"{'Episode':>10} | {'atk_rate':>8} | {'offens%':>7} | {'double%':>7} | {'retreat%':>8} | {'kill%':>6} | {'r1_occ%':>7}")
    print(f"{'-'*70}")

    for ep in checkpoints:
        ckpt_dir = os.path.join(model_dir, f"episode_{ep}")
        s = eval_ckpt(env, ckpt_dir, env.obs_dim)
        print(
            f"{ep:>10,} | {s['attack_rate']:>8.3f} | "
            f"{s['offensive_rate']:>6.1%} | {s['double_rate']:>6.1%} | "
            f"{s['single_rate']:>7.1%} | {s['kill_rate']:>5.1%} | "
            f"{s['r1_occ']:>6.1%}"
        )


VARIANTS = {
    "v1_r2": "./models/stun5_joint_v1_r2",
    "v2_r2": "./models/stun5_joint_v2_r2",
    "v3_r2": "./models/stun5_joint_v3_r2",
}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    args = p.parse_args()
    for name in args.variants:
        run(name, VARIANTS[name])
