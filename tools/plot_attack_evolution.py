#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Plot attack pattern evolution over training checkpoints.

Usage example:
  python plot_attack_evolution.py \
    --model-dir ./models/stun4_r0_stun1_r1 \
    --robot-0-stun-steps 4 --robot-1-stun-steps 1
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
    p = argparse.ArgumentParser(description="Plot attack evolution from checkpoints")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--episodes-per-ckpt", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=500)
    # eval-epsilon removed: inference is always fully greedy (argmax)
    p.add_argument("--num-points", type=int, default=25)
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory (default: model-dir)")

    # Network architecture flags (must match how the model was trained)
    p.add_argument("--dueling", action="store_true", default=False)
    p.add_argument("--noisy", action="store_true", default=False)
    p.add_argument("--c51", action="store_true", default=False)

    # Start position
    p.add_argument("--no-random-starts", action="store_true", default=False,
                   help="Use fixed corner start positions (default: random, matching training default)")

    # Env config (defaults follow your stun experiments)
    p.add_argument("--env-n", type=int, default=5)
    p.add_argument("--e-move", type=float, default=0.0)
    p.add_argument("--e-charge", type=float, default=8.0)
    p.add_argument("--e-collision", type=float, default=30.0)
    p.add_argument("--e-boundary", type=float, default=0.0)
    p.add_argument("--e-decay", type=float, default=0.5)
    p.add_argument("--exclusive-charging", action="store_true", default=True)
    p.add_argument("--charger-range", type=int, default=0)
    p.add_argument("--charger-positions", type=str, default="2,2")
    p.add_argument("--robot-0-energy", type=int, default=100)
    p.add_argument("--robot-1-energy", type=int, default=100)
    p.add_argument("--robot-0-speed", type=int, default=2)
    p.add_argument("--robot-1-speed", type=int, default=1)
    p.add_argument("--robot-0-attack", type=float, default=30.0)
    p.add_argument("--robot-1-attack", type=float, default=2.0)
    p.add_argument("--robot-0-docking-steps", type=int, default=2)
    p.add_argument("--robot-1-docking-steps", type=int, default=0)
    p.add_argument("--robot-0-stun-steps", type=int, default=4)
    p.add_argument("--robot-1-stun-steps", type=int, default=1)

    # Opponent behavior for robot_1
    p.add_argument(
        "--r1-policy",
        choices=["seek_charger", "random", "model"],
        default="seek_charger",
        help="How robot_1 acts during evaluation",
    )
    return p.parse_args()


def parse_charger_positions(s: str):
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
        "exclusive_charging": args.exclusive_charging,
        "charger_range": args.charger_range,
        "charger_positions": parse_charger_positions(args.charger_positions),
        "dust_enabled": False,
        "initial_energy": args.robot_0_energy,
        "robot_energies": [args.robot_0_energy, args.robot_1_energy],
        "robot_speeds": [args.robot_0_speed, args.robot_1_speed],
        "robot_attack_powers": [args.robot_0_attack, args.robot_1_attack],
        "robot_docking_steps": [args.robot_0_docking_steps, args.robot_1_docking_steps],
        "robot_stun_steps": [args.robot_0_stun_steps, args.robot_1_stun_steps],
        "random_start_robots": set() if args.no_random_starts else {0, 1},
        "reward_mode": "delta-energy",
        "reward_alpha": 0.05,
        "agent_types_mode": "off",
    })


def discover_checkpoints(model_dir: str):
    eps = []
    for name in os.listdir(model_dir):
        if not name.startswith("episode_"):
            continue
        try:
            ep = int(name.split("_")[-1])
        except ValueError:
            continue
        path = os.path.join(model_dir, name, "robot_0.pt")
        if os.path.exists(path):
            eps.append(ep)
    eps.sort()
    return eps


def sample_evenly(values, n):
    if len(values) <= n:
        return values
    idx = np.linspace(0, len(values) - 1, n)
    idx = np.round(idx).astype(int)
    # Keep order + unique
    seen = set()
    out = []
    for i in idx:
        v = values[i]
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def seek_charger_action(env):
    obs = env.get_observation(1)
    charger_col = 10  # 3 + 4 + 3 in current obs layout
    cdx, cdy = obs[0, charger_col], obs[0, charger_col + 1]
    if abs(cdx) < 0.01 and abs(cdy) < 0.01:
        return 4
    if abs(cdx) >= abs(cdy):
        return 3 if cdx > 0.01 else (2 if cdx < -0.01 else 4)
    return 1 if cdy > 0.01 else (0 if cdy < -0.01 else 4)


def load_net(path, obs_dim, dueling=False, noisy=False, c51=False):
    net = build_network(num_actions=5, input_dim=obs_dim, dueling=dueling, noisy=noisy, c51=c51)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def eval_checkpoint(args, ckpt_ep, env):
    ckpt_dir = os.path.join(args.model_dir, f"episode_{ckpt_ep}")
    r0_net = load_net(os.path.join(ckpt_dir, "robot_0.pt"), env.obs_dim, args.dueling, args.noisy, args.c51)
    r1_net = None
    if args.r1_policy == "model":
        r1_path = os.path.join(ckpt_dir, "robot_1.pt")
        if os.path.exists(r1_path):
            r1_net = load_net(r1_path, env.obs_dim, args.dueling, args.noisy, args.c51)

    total_double = 0
    total_single = 0
    total_steps = 0
    total_attack_steps = 0
    total_offensive = 0   # hits where r0 not on charger after hit
    total_hits = 0
    r1_kills = 0
    n_episodes = 0
    ep_has_any_hit = 0          # episodes with >=1 hit (any kind)
    ep_has_hunt_hit = 0         # episodes with >=1 pure hunt hit

    charger_pos = parse_charger_positions(args.charger_positions)
    charger_yx = set((y, x) for (y, x) in charger_pos)

    for ep in range(args.episodes_per_ckpt):
        env.reset()
        seed = ep * 37 + 1000
        np.random.seed(seed)
        random.seed(seed)
        n_episodes += 1
        ep_any_hit = False
        ep_hunt_hit = False

        for _ in range(args.max_steps):
            if not env.alive[0].any():
                break
            total_steps += 1

            robot_order = [0, 1]
            random.shuffle(robot_order)

            for robot_id in robot_order:
                speed = [args.robot_0_speed, args.robot_1_speed][robot_id]
                hits_this_step = 0
                offensive_this_step = 0

                for turn_idx in range(speed):
                    is_last = (turn_idx == speed - 1)
                    if not env.alive[0, robot_id]:
                        continue

                    r1_e_before = float(env.energy[0, 1])
                    r1_alive_before = bool(env.alive[0, 1])

                    if robot_id == 0:
                        obs = env.get_observation(0)
                        with torch.no_grad():
                            q = r0_net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())
                    else:
                        if args.r1_policy == "seek_charger":
                            action = seek_charger_action(env)
                        elif args.r1_policy == "random":
                            action = random.randint(0, 4)
                        else:
                            if r1_net is None:
                                action = seek_charger_action(env)
                            else:
                                obs = env.get_observation(1)
                                with torch.no_grad():
                                    q = r1_net(torch.from_numpy(obs).float())
                                    action = int(q.argmax(dim=1).item())

                    env.step_single(robot_id, np.array([action], dtype=np.int32), is_last_turn=is_last)

                    if robot_id == 0 and r1_alive_before:
                        r1_e_after = float(env.energy[0, 1])
                        if r1_e_after < r1_e_before - 1.0:
                            hits_this_step += 1
                            ep_any_hit = True
                            r0y = int(env.pos[0, 0, 0])
                            r0x = int(env.pos[0, 0, 1])
                            r1y = int(env.pos[0, 1, 0])
                            r1x = int(env.pos[0, 1, 1])
                            r0_on_charger = (r0y, r0x) in charger_yx
                            r1_on_charger = (r1y, r1x) in charger_yx
                            # Offensive: r0 not on charger after hit
                            if not r0_on_charger:
                                offensive_this_step += 1
                            # Pure hunt: both r0 and r1 not on charger after hit
                            if not r0_on_charger and not r1_on_charger:
                                ep_hunt_hit = True

                if robot_id == 0:
                    if hits_this_step >= 1:
                        total_attack_steps += 1
                    if hits_this_step == 2:
                        total_double += 1
                    elif hits_this_step == 1:
                        total_single += 1
                    total_hits += hits_this_step
                    total_offensive += offensive_this_step

            done_mask, _ = env.advance_step()
            if done_mask[0]:
                break

        if not env.alive[0, 1]:
            r1_kills += 1
        if ep_any_hit:
            ep_has_any_hit += 1
        if ep_hunt_hit:
            ep_has_hunt_hit += 1

    hit_total = total_double + total_single
    return {
        "attack_rate":    ep_has_any_hit  / max(n_episodes, 1),  # episodes with >=1 hit / total ep
        "hunt_rate":      ep_has_hunt_hit / max(n_episodes, 1),  # episodes with pure hunt hit / total ep
        "double_rate":    total_double    / max(hit_total, 1),
        "single_rate":    total_single    / max(hit_total, 1),
        "offensive_rate": total_offensive / max(total_hits, 1),
        "kill_rate":      r1_kills        / max(n_episodes, 1),
    }


def plot_and_save(model_dir, episodes, attack_rates, hunt_rates, double_rates, single_rates, offensive_rates, output_dir=None):
    x = np.array(episodes) / 1e6
    title_name = os.path.basename(os.path.normpath(model_dir))
    out_dir = output_dir or model_dir

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(x, [r * 100 for r in attack_rates], "b-o", markersize=4, linewidth=1.5, label="Any hit episode rate")
    ax1.plot(x, [r * 100 for r in hunt_rates],   "r-s", markersize=4, linewidth=1.5, label="Pure hunt episode rate")
    ax1.set_ylabel("Episode Rate (%)\n(episodes with hit / total episodes)")
    ax1.set_title(f"{title_name}: Attack Pattern Evolution During Training", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2 = axes[1]
    ax2.plot(x, [r * 100 for r in offensive_rates], "g-^", markersize=4, linewidth=1.5, label="Offensive hit rate")
    ax2.set_ylabel("Offensive Hit Rate (%)\n(r0 not on charger after hit / total hits)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    ax3 = axes[2]
    ax3.fill_between(x, 0, [r * 100 for r in double_rates], alpha=0.6, color="red", label="Double-hit")
    ax3.fill_between(x, [r * 100 for r in double_rates], 100, alpha=0.6, color="steelblue", label="Hit-retreat")
    ax3.set_ylabel("Pattern Ratio (%)\n(among attack steps)")
    ax3.set_xlabel("Training Episodes (millions)")
    ax3.legend(loc="center right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "attack_evolution.png")
    out_csv = os.path.join(out_dir, "attack_evolution.csv")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("episode,any_hit_ep_rate,hunt_ep_rate,offensive_rate,double_hit_pct,hit_retreat_pct\n")
        for i in range(len(episodes)):
            f.write(
                f"{episodes[i]},{attack_rates[i]:.6f},{hunt_rates[i]:.6f},"
                f"{offensive_rates[i]:.6f},{double_rates[i]:.6f},{single_rates[i]:.6f}\n"
            )

    print(f"Plot saved: {out_png}")
    print(f"CSV saved: {out_csv}")


def main():
    args = parse_args()
    all_eps = discover_checkpoints(args.model_dir)
    if not all_eps:
        raise SystemExit(f"No checkpoints found under {args.model_dir}")

    sampled = sample_evenly(all_eps, args.num_points)
    print(f"Found {len(all_eps)} checkpoints. Evaluating {len(sampled)} points.")
    print(f"Range: {sampled[0]} -> {sampled[-1]}")

    env = make_env(args)

    episodes = []
    attack_rates = []
    hunt_rates = []
    double_rates = []
    single_rates = []
    offensive_rates = []

    for i, ep in enumerate(sampled, 1):
        s = eval_checkpoint(args, ep, env)
        episodes.append(ep)
        attack_rates.append(s["attack_rate"])
        hunt_rates.append(s["hunt_rate"])
        double_rates.append(s["double_rate"])
        single_rates.append(s["single_rate"])
        offensive_rates.append(s["offensive_rate"])
        print(
            f"[{i:02d}/{len(sampled)}] ep={ep:>8d} | any_hit={s['attack_rate']:.3f} | "
            f"hunt={s['hunt_rate']:.3f} | offensive={s['offensive_rate']:.1%} | "
            f"double={s['double_rate']:.1%} | retreat={s['single_rate']:.1%}"
        )

    plot_and_save(args.model_dir, episodes, attack_rates, hunt_rates, double_rates, single_rates, offensive_rates, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
