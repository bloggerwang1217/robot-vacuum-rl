#!/usr/bin/env python3
"""
Plot attack evolution for 2v1 alliance experiments.

Tracks r0→r2 and r1→r2 attack metrics across training checkpoints,
all in a single figure with 4 subplots.

Usage:
  python plot_attack_evolution_alliance.py \
    --model-dir ./models/alliance_3robot_v5 \
    --alliance-groups 0,1 \
    --robot-2-energy 200 \
    --robot-2-stun-steps 0 \
    --dueling --noisy --c51
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
    p.add_argument("--episodes-per-ckpt", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-points", type=int, default=25)
    p.add_argument("--output-dir", type=str, default=None)

    # Network architecture
    p.add_argument("--dueling", action="store_true", default=False)
    p.add_argument("--noisy", action="store_true", default=False)
    p.add_argument("--c51", action="store_true", default=False)

    # Env config
    p.add_argument("--env-n", type=int, default=5)
    p.add_argument("--e-move", type=float, default=0.0)
    p.add_argument("--e-charge", type=float, default=8.0)
    p.add_argument("--e-collision", type=float, default=30.0)
    p.add_argument("--e-boundary", type=float, default=0.0)
    p.add_argument("--e-decay", type=float, default=0.5)
    p.add_argument("--charger-positions", type=str, default="2,2")
    p.add_argument("--charger-range", type=int, default=0)
    p.add_argument("--exclusive-charging", action="store_true", default=True)

    # Per-robot config
    p.add_argument("--robot-0-energy", type=int, default=100)
    p.add_argument("--robot-1-energy", type=int, default=100)
    p.add_argument("--robot-2-energy", type=int, default=200)
    p.add_argument("--robot-0-speed", type=int, default=2)
    p.add_argument("--robot-1-speed", type=int, default=2)
    p.add_argument("--robot-2-speed", type=int, default=1)
    p.add_argument("--robot-0-attack", type=float, default=30.0)
    p.add_argument("--robot-1-attack", type=float, default=30.0)
    p.add_argument("--robot-2-attack", type=float, default=2.0)
    p.add_argument("--robot-0-stun-steps", type=int, default=5)
    p.add_argument("--robot-1-stun-steps", type=int, default=5)
    p.add_argument("--robot-2-stun-steps", type=int, default=0)
    p.add_argument("--robot-0-docking-steps", type=int, default=0)
    p.add_argument("--robot-1-docking-steps", type=int, default=0)
    p.add_argument("--robot-2-docking-steps", type=int, default=0)

    # Alliance config
    p.add_argument("--alliance-groups", type=str, default="0,1",
                   help="Comma-separated allied robot IDs, e.g. '0,1'")
    p.add_argument("--energy-sharing-mode", type=str, default="event_only")
    p.add_argument("--energy-sharing-events", type=str, default="charge")
    p.add_argument("--energy-sharing-self-weight", type=float, default=2.0/3.0)
    p.add_argument("--energy-sharing-ally-weight", type=float, default=1.0/3.0)

    p.add_argument("--no-random-starts", action="store_true", default=False)
    return p.parse_args()


def parse_charger_positions(s: str):
    positions = []
    for pair in s.split(";"):
        y, x = pair.strip().split(",")
        positions.append((int(y), int(x)))
    return positions


def parse_alliance_groups(s: str):
    if not s:
        return None
    groups = []
    for g in s.split(";"):
        ids = set(int(x) for x in g.split(",") if x.strip())
        if ids:
            groups.append(ids)
    return groups if groups else None


def make_env(args):
    alliance_groups = parse_alliance_groups(args.alliance_groups)
    sharing_events = [e.strip() for e in args.energy_sharing_events.split(",") if e.strip()]
    return BatchRobotVacuumEnv(num_envs=1, env_kwargs={
        "n": args.env_n,
        "num_robots": 3,
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
        "robot_energies": [args.robot_0_energy, args.robot_1_energy, args.robot_2_energy],
        "robot_speeds": [args.robot_0_speed, args.robot_1_speed, args.robot_2_speed],
        "robot_attack_powers": [args.robot_0_attack, args.robot_1_attack, args.robot_2_attack],
        "robot_docking_steps": [args.robot_0_docking_steps, args.robot_1_docking_steps, args.robot_2_docking_steps],
        "robot_stun_steps": [args.robot_0_stun_steps, args.robot_1_stun_steps, args.robot_2_stun_steps],
        "random_start_robots": set() if args.no_random_starts else {0, 1, 2},
        "reward_mode": "delta-energy",
        "reward_alpha": 0.05,
        "agent_types_mode": "off",
        "alliance_groups": alliance_groups,
        "energy_sharing_mode": args.energy_sharing_mode,
        "energy_sharing_events": sharing_events,
        "energy_sharing_self_weight": args.energy_sharing_self_weight,
        "energy_sharing_ally_weight": args.energy_sharing_ally_weight,
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
        # Need both r0 and r1 models
        if (os.path.exists(os.path.join(model_dir, name, "robot_0.pt")) and
                os.path.exists(os.path.join(model_dir, name, "robot_1.pt"))):
            eps.append(ep)
    eps.sort()
    return eps


def sample_evenly(values, n):
    if len(values) <= n:
        return values
    idx = np.linspace(0, len(values) - 1, n)
    idx = np.round(idx).astype(int)
    seen = set()
    out = []
    for i in idx:
        v = values[i]
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def load_net(path, obs_dim, dueling=False, noisy=False, c51=False):
    net = build_network(num_actions=5, input_dim=obs_dim, dueling=dueling, noisy=noisy, c51=c51)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def seek_charger_action(env, robot_id, num_robots, env_n):
    """Seek-charger heuristic. Charger col = 3 + (num_robots-1)*3."""
    obs = env.get_observation(robot_id)
    charger_col = 3 + (num_robots - 1) * 3
    cdx, cdy = float(obs[0, charger_col]), float(obs[0, charger_col + 1])
    if abs(cdx) < 0.01 and abs(cdy) < 0.01:
        return 4  # STAY
    if abs(cdx) >= abs(cdy):
        return 3 if cdx > 0.01 else 2
    return 1 if cdy > 0.01 else 0


def eval_checkpoint(args, ckpt_ep, env):
    ckpt_dir = os.path.join(args.model_dir, f"episode_{ckpt_ep}")
    r0_net = load_net(os.path.join(ckpt_dir, "robot_0.pt"), env.obs_dim, args.dueling, args.noisy, args.c51)
    r1_net = load_net(os.path.join(ckpt_dir, "robot_1.pt"), env.obs_dim, args.dueling, args.noisy, args.c51)

    charger_pos = parse_charger_positions(args.charger_positions)
    charger_yx = set((y, x) for (y, x) in charger_pos)

    n_episodes = 0
    # Per-attacker episode-level flags
    r0_ep_hit = 0       # episodes where r0 hit r2 at least once
    r1_ep_hit = 0       # episodes where r1 hit r2 at least once
    both_ep_hit = 0     # episodes where BOTH r0 and r1 hit r2
    r0_ep_hunt = 0      # episodes where r0 hunted r2 (neither on charger)
    r1_ep_hunt = 0
    both_ep_hunt = 0    # episodes where BOTH r0 AND r1 hunted r2
    coop_ep = 0         # episodes where hunt hit occurred AND ally was on charger at that step
    r2_kills = 0

    # Across all hits (for offensive rate)
    r0_total_hits = 0
    r0_offensive_hits = 0
    r1_total_hits = 0
    r1_offensive_hits = 0

    for ep in range(args.episodes_per_ckpt):
        env.reset()
        np.random.seed(ep * 37 + 1000)
        random.seed(ep * 37 + 1000)
        n_episodes += 1

        r0_hit_this_ep = False
        r1_hit_this_ep = False
        r0_hunt_this_ep = False
        r1_hunt_this_ep = False
        coop_this_ep = False

        for _ in range(args.max_steps):
            if not env.alive[0].any():
                break

            robot_order = [0, 1, 2]
            random.shuffle(robot_order)

            for robot_id in robot_order:
                speed = [args.robot_0_speed, args.robot_1_speed, args.robot_2_speed][robot_id]

                for turn_idx in range(speed):
                    is_last = (turn_idx == speed - 1)
                    if not env.alive[0, robot_id]:
                        continue

                    # Record r2 energy before this sub-turn
                    r2_e_before = float(env.energy[0, 2])
                    r2_alive_before = bool(env.alive[0, 2])

                    # Select action
                    if robot_id == 0:
                        obs = env.get_observation(0)
                        with torch.no_grad():
                            q = r0_net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())
                    elif robot_id == 1:
                        obs = env.get_observation(1)
                        with torch.no_grad():
                            q = r1_net(torch.from_numpy(obs).float())
                            action = int(q.argmax(dim=1).item())
                    else:
                        # r2: seek-charger scripted
                        action = seek_charger_action(env, 2, 3, args.env_n)

                    env.step_single(robot_id, np.array([action], dtype=np.int32), is_last_turn=is_last)

                    # Detect hit: r2 energy dropped by more than decay+small noise
                    if r2_alive_before and robot_id in (0, 1):
                        r2_e_after = float(env.energy[0, 2])
                        # A hit causes ≥ e_collision damage (30), far above decay (0.5)
                        if r2_e_after < r2_e_before - 1.0:
                            ry = int(env.pos[0, robot_id, 0])
                            rx = int(env.pos[0, robot_id, 1])
                            attacker_on_charger = (ry, rx) in charger_yx
                            is_hunt = not attacker_on_charger

                            if robot_id == 0:
                                r0_total_hits += 1
                                r0_hit_this_ep = True
                                if not attacker_on_charger:
                                    r0_offensive_hits += 1
                                if is_hunt:
                                    r0_hunt_this_ep = True
                                    # check if ally (r1) is on charger at this exact step
                                    ally_y = int(env.pos[0, 1, 0])
                                    ally_x = int(env.pos[0, 1, 1])
                                    if (ally_y, ally_x) in charger_yx:
                                        coop_this_ep = True
                            else:
                                r1_total_hits += 1
                                r1_hit_this_ep = True
                                if not attacker_on_charger:
                                    r1_offensive_hits += 1
                                if is_hunt:
                                    r1_hunt_this_ep = True
                                    # check if ally (r0) is on charger at this exact step
                                    ally_y = int(env.pos[0, 0, 0])
                                    ally_x = int(env.pos[0, 0, 1])
                                    if (ally_y, ally_x) in charger_yx:
                                        coop_this_ep = True

            done_mask, _ = env.advance_step()
            if done_mask[0]:
                break

        if not env.alive[0, 2]:
            r2_kills += 1
        if r0_hit_this_ep:
            r0_ep_hit += 1
        if r1_hit_this_ep:
            r1_ep_hit += 1
        if r0_hit_this_ep and r1_hit_this_ep:
            both_ep_hit += 1
        if r0_hunt_this_ep:
            r0_ep_hunt += 1
        if r1_hunt_this_ep:
            r1_ep_hunt += 1
        if r0_hunt_this_ep and r1_hunt_this_ep:
            both_ep_hunt += 1
        if coop_this_ep:
            coop_ep += 1

    N = max(n_episodes, 1)
    return {
        "r0_hit_rate":      r0_ep_hit   / N,
        "r1_hit_rate":      r1_ep_hit   / N,
        "both_hit_rate":    both_ep_hit / N,
        "r0_hunt_rate":     r0_ep_hunt   / N,
        "r1_hunt_rate":     r1_ep_hunt   / N,
        "both_hunt_rate":   both_ep_hunt / N,
        "coop_rate":        coop_ep     / N,
        "r0_offensive":     r0_offensive_hits / max(r0_total_hits, 1),
        "r1_offensive":     r1_offensive_hits / max(r1_total_hits, 1),
        "r2_kill_rate":     r2_kills    / N,
    }


def plot_and_save(model_dir, episodes, results, output_dir=None):
    x = np.array(episodes) / 1e6
    title_name = os.path.basename(os.path.normpath(model_dir))
    out_dir = output_dir or model_dir

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{title_name}: Alliance Hunt Evolution", fontsize=14, fontweight="bold")

    # ── Subplot 1: Hunt rate（攻擊者不在充電座）────────────────────────────
    ax = axes[0]
    ax.plot(x, [r["r0_hunt_rate"] * 100 for r in results], "b-o", markersize=4, linewidth=1.5, label="r0→r2 hunt")
    ax.plot(x, [r["r1_hunt_rate"] * 100 for r in results], "r-s", markersize=4, linewidth=1.5, label="r1→r2 hunt")
    ax.plot(x, [r["both_hunt_rate"] * 100 for r in results], "g-^", markersize=5, linewidth=2.0, label="r0 AND r1 both hunt (同集)")
    ax.plot(x, [r["coop_rate"] * 100 for r in results], "m-P", markersize=5, linewidth=2.0, label="guard+attack coop (hunt時盟友在充電座)")
    ax.set_ylabel("Episode Rate (%)")
    ax.set_title("Hunt Rate — attacker not on charger when hitting r2")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # ── Subplot 2: r2 kill rate ──────────────────────────────────────────────
    ax = axes[1]
    ax.plot(x, [r["r2_kill_rate"] * 100 for r in results], "k-D", markersize=4, linewidth=1.5, label="r2 kill rate")
    ax.set_ylabel("Rate (%)")
    ax.set_title("r2 Kill Rate")
    ax.set_xlabel("Training Episodes (millions)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "attack_evolution.png")
    out_csv = os.path.join(out_dir, "attack_evolution.csv")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("episode,r0_hit_rate,r1_hit_rate,both_hit_rate,r0_hunt_rate,r1_hunt_rate,both_hunt_rate,"
                "coop_rate,r0_offensive,r1_offensive,r2_kill_rate\n")
        for ep, r in zip(episodes, results):
            f.write(f"{ep},{r['r0_hit_rate']:.6f},{r['r1_hit_rate']:.6f},{r['both_hit_rate']:.6f},"
                    f"{r['r0_hunt_rate']:.6f},{r['r1_hunt_rate']:.6f},{r['both_hunt_rate']:.6f},"
                    f"{r['coop_rate']:.6f},{r['r0_offensive']:.6f},{r['r1_offensive']:.6f},{r['r2_kill_rate']:.6f}\n")

    print(f"Plot saved: {out_png}")
    print(f"CSV  saved: {out_csv}")


def main():
    args = parse_args()
    all_eps = discover_checkpoints(args.model_dir)
    if not all_eps:
        raise SystemExit(f"No checkpoints with both robot_0.pt and robot_1.pt found under {args.model_dir}")

    sampled = sample_evenly(all_eps, args.num_points)
    print(f"Found {len(all_eps)} checkpoints. Evaluating {len(sampled)} points.")
    print(f"Range: {sampled[0]} → {sampled[-1]}")

    env = make_env(args)
    print(f"obs_dim = {env.obs_dim}")

    results = []
    for i, ep in enumerate(sampled, 1):
        r = eval_checkpoint(args, ep, env)
        results.append(r)
        print(
            f"[{i:02d}/{len(sampled)}] ep={ep:>8d} | "
            f"r0_hit={r['r0_hit_rate']:.2f} r1_hit={r['r1_hit_rate']:.2f} both={r['both_hit_rate']:.2f} | "
            f"hunt r0={r['r0_hunt_rate']:.2f} r1={r['r1_hunt_rate']:.2f} | "
            f"coop={r['coop_rate']:.2f} | "
            f"off r0={r['r0_offensive']:.1%} r1={r['r1_offensive']:.1%} | "
            f"kill={r['r2_kill_rate']:.2f}"
        )

    plot_and_save(args.model_dir, sampled, results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
