#!/usr/bin/env python3
"""
分析 r1 被撞離充電座後，在 N 步內是否回到充電座（是/否），
畫出每個 checkpoint 的「N步內回充率」。

用法：
  python analyze_knockoff_return.py \
    --model-dir ./models/stun5_r1_seek_v2 \
    --return-window 3 \
    --output ./analyze/stun5_r1_seek_v2_knockoff_return3.png
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from batch_env import BatchRobotVacuumEnv
from dqn import build_network


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--return-window", type=int, default=3,
                   help="Steps within which r1 must return (default: 3)")
    p.add_argument("--episodes-per-ckpt", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-points", type=int, default=200)
    p.add_argument("--output", type=str, default=None)

    p.add_argument("--env-n", type=int, default=5)
    p.add_argument("--charger-positions", type=str, default="2,2")
    p.add_argument("--charger-range", type=int, default=0)
    p.add_argument("--robot-0-speed", type=int, default=2)
    p.add_argument("--robot-1-speed", type=int, default=1)
    p.add_argument("--robot-0-attack", type=float, default=30.0)
    p.add_argument("--robot-1-attack", type=float, default=2.0)
    p.add_argument("--robot-0-docking-steps", type=int, default=2)
    p.add_argument("--robot-1-docking-steps", type=int, default=0)
    p.add_argument("--robot-0-stun-steps", type=int, default=5)
    p.add_argument("--robot-1-stun-steps", type=int, default=1)
    p.add_argument("--robot-0-energy", type=int, default=100)
    p.add_argument("--robot-1-energy", type=int, default=100)
    p.add_argument("--e-move", type=float, default=1.0)
    p.add_argument("--e-charge", type=float, default=8.0)
    p.add_argument("--e-decay", type=float, default=0.5)
    p.add_argument("--e-boundary", type=float, default=0.0)
    p.add_argument("--e-collision", type=float, default=30.0)
    return p.parse_args()


def parse_charger_positions(s):
    positions = []
    for pair in s.split(";"):
        y, x = pair.strip().split(",")
        positions.append((int(y), int(x)))
    return positions


def make_env(args):
    return BatchRobotVacuumEnv(num_envs=1, env_kwargs={
        "n": args.env_n,
        "num_robots": 2,
        "n_steps": args.max_steps,
        "e_move": args.e_move,
        "e_charge": args.e_charge,
        "e_collision": args.e_collision,
        "e_boundary": args.e_boundary,
        "e_decay": args.e_decay,
        "exclusive_charging": True,
        "charger_range": args.charger_range,
        "charger_positions": parse_charger_positions(args.charger_positions),
        "dust_enabled": False,
        "initial_energy": args.robot_0_energy,
        "robot_energies": [args.robot_0_energy, args.robot_1_energy],
        "robot_speeds": [args.robot_0_speed, args.robot_1_speed],
        "robot_attack_powers": [args.robot_0_attack, args.robot_1_attack],
        "robot_docking_steps": [args.robot_0_docking_steps, args.robot_1_docking_steps],
        "robot_stun_steps": [args.robot_0_stun_steps, args.robot_1_stun_steps],
        "random_start_robots": {0, 1},
        "reward_mode": "delta-energy",
        "reward_alpha": 0.05,
        "agent_types_mode": "off",
    })


def discover_checkpoints(model_dir):
    eps = []
    for name in os.listdir(model_dir):
        if not name.startswith("episode_"):
            continue
        try:
            ep = int(name.split("_")[-1])
        except ValueError:
            continue
        # only need robot_1 for this analysis
        path = os.path.join(model_dir, name, "robot_1.pt")
        if os.path.exists(path):
            eps.append(ep)
    eps.sort()
    return eps


def sample_evenly(values, n):
    if len(values) <= n:
        return values
    idx = np.round(np.linspace(0, len(values) - 1, n)).astype(int)
    seen, out = set(), []
    for i in idx:
        v = values[i]
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def load_net(path, obs_dim):
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # Auto-detect architecture from saved weights
    is_noisy = any("weight_mu" in k for k in state_dict)
    is_dueling = any("value_stream" in k for k in state_dict)
    is_c51 = any("atoms" in k for k in state_dict)
    net = build_network(num_actions=5, input_dim=obs_dim,
                        dueling=is_dueling, noisy=is_noisy, c51=is_c51)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def eval_checkpoint(args, ckpt_ep, env, charger_yx, return_window):
    r1_path = os.path.join(args.model_dir, f"episode_{ckpt_ep}", "robot_1.pt")
    r1_net = load_net(r1_path, env.obs_dim)

    total_knockoffs = 0
    returned_in_window = 0

    for ep in range(args.episodes_per_ckpt):
        np.random.seed(ep * 37 + 1000)
        random.seed(ep * 37 + 1000)
        env.reset()

        # Track r1 position history per step
        r1_on_charger_prev = False
        steps_since_knockoff = None  # None = not in knockoff tracking

        for _ in range(args.max_steps):
            if not env.alive[0].any():
                break

            # r0: random walk (we only care about r1 behavior)
            robot_order = [0, 1]
            random.shuffle(robot_order)

            for robot_id in robot_order:
                speed = [args.robot_0_speed, args.robot_1_speed][robot_id]
                for turn_idx in range(speed):
                    is_last = (turn_idx == speed - 1)
                    if not env.alive[0, robot_id]:
                        continue
                    if robot_id == 0:
                        action = random.randint(0, 4)
                    else:
                        obs = env.get_observation(1)
                        with torch.no_grad():
                            q = r1_net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())
                    env.step_single(robot_id, np.array([action], dtype=np.int32), is_last_turn=is_last)

            env.advance_step()

            if not env.alive[0, 1]:
                break

            r1y = int(env.pos[0, 1, 0])
            r1x = int(env.pos[0, 1, 1])
            r1_on_charger = (r1y, r1x) in charger_yx

            # Detect knockoff: was on charger last step, not on charger now
            if r1_on_charger_prev and not r1_on_charger:
                total_knockoffs += 1
                steps_since_knockoff = 0

            # Track steps since knockoff
            if steps_since_knockoff is not None:
                steps_since_knockoff += 1
                if r1_on_charger:
                    # Returned!
                    if steps_since_knockoff <= return_window:
                        returned_in_window += 1
                    steps_since_knockoff = None  # reset
                elif steps_since_knockoff > return_window:
                    # Didn't return in time — count as no-return
                    steps_since_knockoff = None

            r1_on_charger_prev = r1_on_charger

    return_rate = returned_in_window / max(total_knockoffs, 1)
    return return_rate, total_knockoffs


def main():
    args = parse_args()
    charger_pos = parse_charger_positions(args.charger_positions)
    charger_yx = set((y, x) for (y, x) in charger_pos)

    all_eps = discover_checkpoints(args.model_dir)
    if not all_eps:
        raise SystemExit(f"No checkpoints with robot_1.pt found under {args.model_dir}")

    sampled = sample_evenly(all_eps, args.num_points)
    print(f"Found {len(all_eps)} checkpoints. Evaluating {len(sampled)} points.")

    env = make_env(args)
    episodes, rates = [], []

    for i, ep in enumerate(sampled, 1):
        rate, knockoffs = eval_checkpoint(args, ep, env, charger_yx, args.return_window)
        episodes.append(ep)
        rates.append(rate)
        print(f"[{i:03d}/{len(sampled)}] ep={ep:>8d} | return_in_{args.return_window}={rate:.1%} | knockoffs={knockoffs}")

    # Plot
    x = np.array(episodes) / 1e6
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, [r * 100 for r in rates], "b-o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Training Episodes (millions)")
    ax.set_ylabel(f"Return-in-{args.return_window} Rate (%)")
    ax.set_title(
        f"{os.path.basename(args.model_dir)}: "
        f"r1 Knockoff Return Rate (within {args.return_window} steps)"
    )
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output or os.path.join(
        args.model_dir, f"knockoff_return_{args.return_window}steps.png"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
