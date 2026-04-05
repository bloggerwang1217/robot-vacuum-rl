"""
Evaluate multiple checkpoints and plot behavioral evolution over training.
Usage:
  python eval_evolution.py --model-base ./models/2robot_sym_evolution \
    --env-n 6 --num-robots 2 --charger-positions 3,3 --charger-range 0 \
    --e-decay 0.3 --e-move 0.1 --e-charge 3.0 --e-collision 50 --e-boundary 0 \
    --no-dust --no-noisy --shuffle-step-order \
    --robot-0-energy 100 --robot-1-energy 100 \
    --robot-0-attack-power 50 --robot-1-attack-power 50 \
    --eval-episodes 200 --max-steps 500
"""
import os
import sys
import argparse
import numpy as np
import torch
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from batch_env import BatchRobotVacuumEnv
from dqn import DQN, C51DQN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-base", required=True, help="Base model dir (contains episode_N subdirs)")
    p.add_argument("--env-n", type=int, default=6)
    p.add_argument("--num-robots", type=int, default=2)
    p.add_argument("--charger-positions", type=str, default="3,3")
    p.add_argument("--charger-range", type=int, default=0)
    p.add_argument("--e-move", type=float, default=0.1)
    p.add_argument("--e-charge", type=float, default=3.0)
    p.add_argument("--e-collision", type=float, default=50)
    p.add_argument("--e-boundary", type=float, default=0)
    p.add_argument("--e-decay", type=float, default=0.3)
    p.add_argument("--energy-cap", type=float, default=None)
    p.add_argument("--exclusive-charging", action="store_true")
    p.add_argument("--no-dust", action="store_true")
    p.add_argument("--shuffle-step-order", action="store_true")
    p.add_argument("--robot-0-energy", type=float, default=100)
    p.add_argument("--robot-1-energy", type=float, default=100)
    p.add_argument("--robot-2-energy", type=float, default=100)
    p.add_argument("--robot-3-energy", type=float, default=100)
    p.add_argument("--robot-0-attack-power", type=float, default=None)
    p.add_argument("--robot-1-attack-power", type=float, default=None)
    p.add_argument("--robot-2-attack-power", type=float, default=None)
    p.add_argument("--robot-3-attack-power", type=float, default=None)
    p.add_argument("--robot-0-speed", type=int, default=1)
    p.add_argument("--robot-1-speed", type=int, default=1)
    p.add_argument("--robot-2-speed", type=int, default=1)
    p.add_argument("--robot-3-speed", type=int, default=1)
    p.add_argument("--seek-charger-robots", type=str, default="")
    p.add_argument("--scripted-robots", type=str, default="")
    p.add_argument("--robot-start-positions", type=str, default=None,
                   help="Fixed start positions 'y0,x0;y1,x1;...'")
    p.add_argument("--docking-steps", type=int, default=0)
    p.add_argument("--robot-0-docking-steps", type=int, default=None)
    p.add_argument("--robot-1-docking-steps", type=int, default=None)
    p.add_argument("--stun-steps", type=int, default=0)
    p.add_argument("--robot-0-stun-steps", type=int, default=None)
    p.add_argument("--robot-1-stun-steps", type=int, default=None)
    p.add_argument("--no-noisy", action="store_true")
    p.add_argument("--no-c51", action="store_true")
    p.add_argument("--no-dueling", action="store_true")
    p.add_argument("--v-min", type=float, default=-120)
    p.add_argument("--v-max", type=float, default=20)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--output", type=str, default=None, help="Output image path")
    return p.parse_args()


def build_env(args):
    R = args.num_robots
    n = args.env_n
    energies = [args.robot_0_energy, args.robot_1_energy, args.robot_2_energy, args.robot_3_energy][:R]
    attacks = [args.robot_0_attack_power, args.robot_1_attack_power,
               args.robot_2_attack_power, args.robot_3_attack_power][:R]
    attacks = [a if a is not None else args.e_collision for a in attacks]
    speeds = [args.robot_0_speed, args.robot_1_speed, args.robot_2_speed, args.robot_3_speed][:R]

    cp = []
    for pair in args.charger_positions.split(";"):
        y, x = pair.strip().split(",")
        cp.append((int(y), int(x)))

    env_kwargs = {
        'n': n, 'num_robots': R, 'n_steps': args.max_steps,
        'charger_positions': cp, 'charger_range': args.charger_range,
        'e_move': args.e_move, 'e_charge': args.e_charge,
        'e_collision': args.e_collision, 'e_boundary': args.e_boundary,
        'e_decay': args.e_decay,
        'robot_energies': energies,
        'robot_attack_powers': attacks,
        'robot_speeds': speeds,
        'exclusive_charging': args.exclusive_charging,
        'dust_enabled': not args.no_dust,
        'energy_cap': args.energy_cap,
        'docking_steps': args.docking_steps,
        'stun_steps': args.stun_steps,
    }
    # Per-robot docking
    dock_list = [getattr(args, 'robot_0_docking_steps', None),
                 getattr(args, 'robot_1_docking_steps', None)][:R]
    if any(d is not None for d in dock_list):
        env_kwargs['robot_docking_steps'] = [d if d is not None else args.docking_steps for d in dock_list]
    # Per-robot stun
    stun_list = [getattr(args, 'robot_0_stun_steps', None),
                 getattr(args, 'robot_1_stun_steps', None)][:R]
    if any(s is not None for s in stun_list):
        env_kwargs['robot_stun_steps'] = [s if s is not None else args.stun_steps for s in stun_list]
    if args.robot_start_positions:
        rsp = {}
        for i, pair in enumerate(args.robot_start_positions.split(";")):
            y, x = pair.strip().split(",")
            rsp[i] = (int(y), int(x))
        env_kwargs['robot_start_positions'] = rsp
    return env_kwargs, energies, attacks, speeds


def load_models(checkpoint_dir, obs_dim, action_dim, device, noisy, v_min=-120, v_max=20, use_c51=True, use_dueling=True):
    models = []
    rid = 0
    while True:
        path = os.path.join(checkpoint_dir, f"robot_{rid}.pt")
        if not os.path.exists(path):
            break
        if use_c51:
            net = C51DQN(action_dim, obs_dim, num_atoms=51, v_min=v_min, v_max=v_max, dueling=use_dueling, noisy=noisy)
        else:
            net = DQN(action_dim, obs_dim)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        net.to(device)
        net.eval()
        models.append(net)
        rid += 1
    return models


def seek_charger_action(pos_y, pos_x, charger_y, charger_x):
    dy = charger_y - pos_y
    dx = charger_x - pos_x
    if dy == 0 and dx == 0:
        return 4  # STAY
    if abs(dy) >= abs(dx):
        return 1 if dy > 0 else 0  # DOWN or UP
    else:
        return 3 if dx > 0 else 2  # RIGHT or LEFT


def eval_checkpoint(checkpoint_dir, args, env_kwargs, device):
    R = args.num_robots
    n = args.env_n
    N = args.eval_episodes

    # Create env with N parallel envs
    ek = dict(env_kwargs)
    ek['n_steps'] = args.max_steps
    env = BatchRobotVacuumEnv(num_envs=N, env_kwargs=ek)

    # Obs dim
    obs_dim = env._build_obs(0).shape[1]
    action_dim = 5

    # Load models (only for learning robots)
    seek_set = set(int(x) for x in args.seek_charger_robots.split(',') if x.strip()) if args.seek_charger_robots else set()
    script_set = set(int(x) for x in args.scripted_robots.split(',') if x.strip()) if args.scripted_robots else set()

    v_min = getattr(args, 'v_min', -120)
    v_max = getattr(args, 'v_max', 20)
    models = load_models(checkpoint_dir, obs_dim, action_dim, device, noisy=not args.no_noisy, v_min=v_min, v_max=v_max, use_c51=not args.no_c51, use_dueling=not args.no_dueling)

    # Reset (uses robot_start_positions if set in env_kwargs)
    env.reset()
    # If no fixed start positions, randomize
    if 'robot_start_positions' not in env_kwargs:
        for rid in range(R):
            env.pos[:, rid, 0] = np.random.randint(0, n, size=N)
            env.pos[:, rid, 1] = np.random.randint(0, n, size=N)

    # Tracking
    alive_at_end = np.ones((N, R), dtype=bool)
    death_step = np.full((N, R), args.max_steps, dtype=np.int32)
    total_collisions = np.zeros((N, R), dtype=np.int32)  # hits given
    off_charger_collisions = np.zeros((N, R), dtype=np.int32)  # hits given while NOT on charger
    total_rewards = np.zeros((N, R), dtype=np.float32)
    charger_steps = np.zeros((N, R), dtype=np.int32)  # steps on charger
    dist_to_opponent_sum = np.zeros((N,), dtype=np.float32)  # r0-r1 distance
    dist_to_opponent_count = np.zeros((N,), dtype=np.int32)

    cp = env.charger_positions[0]

    for step in range(args.max_steps):
        robot_order = list(range(R))
        if args.shuffle_step_order:
            np.random.shuffle(robot_order)

        for rid in robot_order:
            speed = [args.robot_0_speed, args.robot_1_speed, args.robot_2_speed, args.robot_3_speed][rid]
            for turn in range(speed):
                is_last_turn = (turn == speed - 1)
                if rid in seek_set:
                    # Seek charger heuristic
                    actions = np.full(N, 4, dtype=np.int32)
                    for e in range(N):
                        if env.alive[e, rid]:
                            actions[e] = seek_charger_action(
                                env.pos[e, rid, 0], env.pos[e, rid, 1], cp[0], cp[1])
                elif rid in script_set:
                    actions = np.full(N, 4, dtype=np.int32)
                elif rid < len(models):
                    obs = env._build_obs(rid)
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs).to(device)
                        q = models[rid](obs_t)
                        if q.dim() == 3:  # C51
                            support = torch.linspace(-120, 20, 51).to(device)
                            q = (q * support.unsqueeze(0).unsqueeze(0)).sum(-1)
                        actions = q.argmax(dim=1).cpu().numpy().astype(np.int32)
                else:
                    actions = np.random.randint(0, 5, size=N).astype(np.int32)

                env.step_single(rid, actions, is_last_turn=is_last_turn)
                total_rewards[:, rid] += env.energy[:, rid] - env.prev_energy[:, rid]  # approx

            # Track collisions (hits given by rid)
            hits_this_turn = env.active_collisions_with[:, rid, :].sum(axis=1)
            total_collisions[:, rid] += hits_this_turn
            # Off-charger collisions: rid gave hits while NOT standing on any charger
            on_charger_rid = np.zeros(N, dtype=bool)
            for cp_ in env.charger_positions:
                on_charger_rid |= (env.pos[:, rid, 0] == cp_[0]) & (env.pos[:, rid, 1] == cp_[1])
            off_charger_collisions[:, rid] += hits_this_turn * (~on_charger_rid).astype(np.int32)

        # End of timestep tracking
        for rid in range(R):
            alive = env.alive[:, rid]
            just_died = alive_at_end[:, rid] & ~alive
            death_step[just_died, rid] = step
            alive_at_end[:, rid] = alive

            # Charger occupancy
            on_charger = alive & (env.pos[:, rid, 0] == cp[0]) & (env.pos[:, rid, 1] == cp[1])
            charger_steps[on_charger, rid] += 1

        # Distance between r0 and r1
        if R >= 2:
            both_alive = env.alive[:, 0] & env.alive[:, 1]
            if both_alive.any():
                dy = np.abs(env.pos[both_alive, 0, 0] - env.pos[both_alive, 1, 0])
                dx = np.abs(env.pos[both_alive, 0, 1] - env.pos[both_alive, 1, 1])
                dist_to_opponent_sum[both_alive] += (dy + dx).astype(np.float32)
                dist_to_opponent_count[both_alive] += 1

    # Compute metrics
    num_alive = alive_at_end.sum(axis=1)
    metrics = {}

    for rid in range(R):
        prefix = f"r{rid}"
        survived = alive_at_end[:, rid].sum()
        sole = (alive_at_end[:, rid] & (num_alive == 1)).sum()
        avg_death = death_step[~alive_at_end[:, rid], rid].mean() if (~alive_at_end[:, rid]).any() else args.max_steps
        avg_coll = total_collisions[:, rid].mean()
        avg_off_charger_coll = off_charger_collisions[:, rid].mean()
        avg_charger = charger_steps[:, rid].mean()

        metrics[f"{prefix}_survival_rate"] = survived / N
        metrics[f"{prefix}_sole_winner_rate"] = sole / N
        metrics[f"{prefix}_avg_death_step"] = float(avg_death)
        metrics[f"{prefix}_avg_collisions"] = float(avg_coll)
        metrics[f"{prefix}_avg_off_charger_collisions"] = float(avg_off_charger_coll)
        metrics[f"{prefix}_avg_charger_steps"] = float(avg_charger)

    metrics["both_alive_rate"] = (num_alive == R).sum() / N
    metrics["all_dead_rate"] = (num_alive == 0).sum() / N

    # Avg distance between r0 and r1 when both alive
    valid = dist_to_opponent_count > 0
    if valid.any():
        metrics["avg_dist_r0_r1"] = float((dist_to_opponent_sum[valid] / dist_to_opponent_count[valid]).mean())
    else:
        metrics["avg_dist_r0_r1"] = 0.0

    return metrics


def main():
    args = parse_args()

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    env_kwargs, _, _, _ = build_env(args)

    # Find all checkpoints
    base = args.model_base
    episodes = []
    for d in os.listdir(base):
        m = re.match(r"episode_(\d+)$", d)
        if m:
            ep = int(m.group(1))
            if ep > 0:
                episodes.append(ep)
    episodes.sort()

    print(f"Found {len(episodes)} checkpoints: {episodes[0]:,} - {episodes[-1]:,}")
    print(f"Evaluating each with {args.eval_episodes} greedy episodes...")

    R = args.num_robots
    all_metrics = {ep: {} for ep in episodes}

    for i, ep in enumerate(episodes):
        ckpt_dir = os.path.join(base, f"episode_{ep}")
        metrics = eval_checkpoint(ckpt_dir, args, env_kwargs, device)
        all_metrics[ep] = metrics
        # Progress
        r0_win = metrics["r0_sole_winner_rate"]
        r1_win = metrics["r1_sole_winner_rate"] if R >= 2 else 0
        r0_coll = metrics["r0_avg_collisions"]
        r1_coll = metrics["r1_avg_collisions"] if R >= 2 else 0
        print(f"  [{i+1}/{len(episodes)}] ep={ep:>8,} | r0_win={r0_win:.2f} r1_win={r1_win:.2f} | "
              f"r0_coll={r0_coll:.1f} r1_coll={r1_coll:.1f} | "
              f"both_alive={metrics['both_alive_rate']:.2f} all_dead={metrics['all_dead_rate']:.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    eps = np.array(episodes) / 1e6  # in millions

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(f"Behavioral Evolution: {os.path.basename(base)}", fontsize=14, fontweight='bold')
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # 1. Win rates
    ax1 = fig.add_subplot(gs[0, 0])
    for rid in range(R):
        rates = [all_metrics[ep][f"r{rid}_sole_winner_rate"] for ep in episodes]
        ax1.plot(eps, rates, color=colors[rid], label=f"r{rid} sole winner", linewidth=2)
    both_alive = [all_metrics[ep]["both_alive_rate"] for ep in episodes]
    all_dead = [all_metrics[ep]["all_dead_rate"] for ep in episodes]
    ax1.plot(eps, both_alive, color='gray', linestyle='--', label="both alive", linewidth=1.5)
    ax1.plot(eps, all_dead, color='black', linestyle=':', label="all dead", linewidth=1.5)
    ax1.set_xlabel("Training Episodes (M)")
    ax1.set_ylabel("Rate")
    ax1.set_title("Win Rates (Sole Survivor)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # 2. Avg collisions (hits given)
    ax2 = fig.add_subplot(gs[0, 1])
    for rid in range(R):
        colls = [all_metrics[ep][f"r{rid}_avg_collisions"] for ep in episodes]
        ax2.plot(eps, colls, color=colors[rid], label=f"r{rid} hits given", linewidth=2)
    ax2.set_xlabel("Training Episodes (M)")
    ax2.set_ylabel("Avg Collisions / Episode")
    ax2.set_title("Aggression (Hits Given)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Avg death step
    ax3 = fig.add_subplot(gs[1, 0])
    for rid in range(R):
        deaths = [all_metrics[ep][f"r{rid}_avg_death_step"] for ep in episodes]
        ax3.plot(eps, deaths, color=colors[rid], label=f"r{rid} avg death step", linewidth=2)
    ax3.set_xlabel("Training Episodes (M)")
    ax3.set_ylabel("Step")
    ax3.set_title("Average Death Step (higher = survives longer)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Charger occupancy
    ax4 = fig.add_subplot(gs[1, 1])
    for rid in range(R):
        charger = [all_metrics[ep][f"r{rid}_avg_charger_steps"] for ep in episodes]
        ax4.plot(eps, charger, color=colors[rid], label=f"r{rid} charger steps", linewidth=2)
    ax4.set_xlabel("Training Episodes (M)")
    ax4.set_ylabel("Avg Steps on Charger")
    ax4.set_title("Charger Occupancy")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Distance between r0 and r1
    if R >= 2:
        ax5 = fig.add_subplot(gs[2, 0])
        dists = [all_metrics[ep]["avg_dist_r0_r1"] for ep in episodes]
        ax5.plot(eps, dists, color='purple', linewidth=2)
        ax5.set_xlabel("Training Episodes (M)")
        ax5.set_ylabel("Manhattan Distance")
        ax5.set_title("Avg Distance r0↔r1 (while both alive)")
        ax5.grid(True, alpha=0.3)

    # 6. Survival rates
    ax6 = fig.add_subplot(gs[2, 1])
    for rid in range(R):
        surv = [all_metrics[ep][f"r{rid}_survival_rate"] for ep in episodes]
        ax6.plot(eps, surv, color=colors[rid], label=f"r{rid} survival", linewidth=2)
    ax6.set_xlabel("Training Episodes (M)")
    ax6.set_ylabel("Rate")
    ax6.set_title("Survival Rate")
    ax6.legend(fontsize=8)
    ax6.set_ylim(-0.05, 1.05)
    ax6.grid(True, alpha=0.3)

    # 7. Off-charger collisions (active pursuit indicator)
    ax7 = fig.add_subplot(gs[3, 0])
    for rid in range(R):
        off_colls = [all_metrics[ep][f"r{rid}_avg_off_charger_collisions"] for ep in episodes]
        ax7.plot(eps, off_colls, color=colors[rid], label=f"r{rid} off-charger hits", linewidth=2)
    ax7.set_xlabel("Training Episodes (M)")
    ax7.set_ylabel("Avg Off-Charger Hits / Episode")
    ax7.set_title("Off-Charger Aggression (Active Pursuit Indicator)")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. Off-charger ratio (off-charger hits / total hits)
    if R >= 2:
        ax8 = fig.add_subplot(gs[3, 1])
        for rid in range(R):
            total = [all_metrics[ep][f"r{rid}_avg_collisions"] for ep in episodes]
            off = [all_metrics[ep][f"r{rid}_avg_off_charger_collisions"] for ep in episodes]
            ratio = [o / t if t > 0.01 else 0.0 for o, t in zip(off, total)]
            ax8.plot(eps, ratio, color=colors[rid], label=f"r{rid} off-charger ratio", linewidth=2)
        ax8.set_xlabel("Training Episodes (M)")
        ax8.set_ylabel("Ratio")
        ax8.set_title("Off-Charger Hit Ratio (off / total)")
        ax8.legend(fontsize=8)
        ax8.set_ylim(-0.05, 1.05)
        ax8.grid(True, alpha=0.3)

    output = args.output or os.path.join(base, "evolution_plot.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output}")


if __name__ == "__main__":
    main()
