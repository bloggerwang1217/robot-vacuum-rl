"""
survival_analysis.py
--------------------
Run N evaluation episodes and analyse robot_0's survival:
  1. Death step distribution (histogram)
  2. Death position heatmap (where robot_1 kills robot_0)
  3. Average survival time by position (where robot_0 lives longer)

Usage:
  cd robot-vacuum-rl
  python analysis/survival_analysis.py \
    --model-dir ./models/p2_6x6/episode_183328_interrupted \
    --env-n 6 --num-robots 2 \
    --robot-0-energy 20 --robot-1-energy 100 \
    --robot-0-speed 1 --robot-1-speed 2 \
    --charger-positions 3,3 \
    --exclusive-charging --no-dust \
    --e-move 0 --e-charge 10 --e-collision 30 --e-boundary 0 \
    --max-steps 1000 --num-episodes 500 \
    --safe-random-robots 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dqn import DQN, init_weights
from gym import RobotVacuumGymEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_charger_positions(s: str):
    """Parse 'y1,x1;y2,x2' or 'y1,x1' into [(y,x), ...]"""
    if not s:
        return None
    result = []
    for token in s.replace(";", ",").split(","):
        pass  # we'll parse in pairs below
    parts = [int(t) for t in s.replace(";", ",").split(",")]
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]


def safe_random_action(obs, n: int) -> int:
    """Wall-avoiding random action for robot_0."""
    rx = round(float(obs[0]) * (n - 1))
    ry = round(float(obs[1]) * (n - 1))
    valid = [4]                          # STAY always valid
    if ry > 0:     valid.append(0)       # UP
    if ry < n - 1: valid.append(1)       # DOWN
    if rx > 0:     valid.append(2)       # LEFT
    if rx < n - 1: valid.append(3)       # RIGHT
    return int(np.random.choice(valid))


def greedy_action(q_net: torch.nn.Module, obs: np.ndarray, device: torch.device) -> int:
    """Greedy Q-network action."""
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        return int(q_net(obs_t).argmax(dim=1).item())


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(env: RobotVacuumGymEnv,
                q_nets: dict,          # {robot_id: DQN | None}  None = safe_random
                safe_random_ids: set,
                max_steps: int,
                device: torch.device,
                scripted_ids: set = None):
    """
    Returns:
        death_step  : int  (max_steps if robot_0 survived)
        death_pos   : (x, y) or None if survived
        r0_traj     : list of (x, y) visited by robot_0 while alive
    """
    env.reset()

    n = env.n
    robot_speeds = env.env.robot_speeds
    num_robots = env.n_robots
    terminated = [False] * num_robots

    death_step = None
    death_pos = None
    r0_traj = []

    for step in range(max_steps):
        # Record robot_0 position at start of this step (if alive)
        state = env.env.get_global_state()
        r0 = state['robots'][0]
        if r0['is_active']:
            r0_traj.append((r0['x'], r0['y']))

        # Each robot acts (sequential, multi-turn if speed > 1)
        for robot_id in range(num_robots):
            if terminated[robot_id]:
                continue
            for _ in range(robot_speeds[robot_id]):
                obs = env.get_observation(robot_id)
                if scripted_ids and robot_id in scripted_ids:
                    action = 4  # STAY
                elif robot_id in safe_random_ids:
                    action = safe_random_action(obs, n)
                else:
                    action = greedy_action(q_nets[robot_id], obs, device)

                _, _, term, _, _ = env.step_single(robot_id, action)

                if term:
                    terminated[robot_id] = True
                    if robot_id == 0 and death_step is None:
                        # Capture death position
                        s = env.env.get_global_state()
                        death_pos = (s['robots'][0]['x'], s['robots'][0]['y'])
                        death_step = step + 1
                    break  # robot is dead, no more turns

        done, _ = env.advance_step()

        if done or all(terminated):
            break

    if death_step is None:
        death_step = max_steps  # survived to the end

    return death_step, death_pos, r0_traj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CHARGER_COLOR = "#2979FF"   # blue rectangle for charger cells


def _mark_chargers(ax, charger_positions, offset=0.5):
    if not charger_positions:
        return
    for (cy, cx) in charger_positions:
        rect = plt.Rectangle(
            (cx - offset, cy - offset), 1, 1,
            fill=False, edgecolor=CHARGER_COLOR, linewidth=2.5, label="Charger"
        )
        ax.add_patch(rect)


def plot_death_hist(death_steps: np.ndarray, max_steps: int, out_path: Path):
    survived = int(np.sum(death_steps == max_steps))
    died_steps = death_steps[death_steps < max_steps]
    n_total = len(death_steps)

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(died_steps) > 0:
        bins = min(60, max(10, max_steps // 10))
        ax.hist(died_steps, bins=bins, color="#E53935", edgecolor="white",
                linewidth=0.4, alpha=0.85, label="Died")
        ax.axvline(died_steps.mean(), color="#7B1FA2", linestyle="--", linewidth=1.5,
                   label=f"Mean = {died_steps.mean():.1f}")
        ax.axvline(np.median(died_steps), color="#F57F17", linestyle=":", linewidth=1.5,
                   label=f"Median = {np.median(died_steps):.0f}")

    title = (f"Robot_0 Death Step Distribution  "
             f"(n={n_total},  survived={survived} [{100*survived/n_total:.1f}%])")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step when robot_0 died", fontsize=10)
    ax.set_ylabel("Episode count", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_death_heatmap(death_grid: np.ndarray, n: int,
                       charger_positions, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    vmax = max(death_grid.max(), 1)
    im = ax.imshow(death_grid, cmap="Reds", origin="upper", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Death count")
    _mark_chargers(ax, charger_positions)

    for y in range(n):
        for x in range(n):
            val = death_grid[y, x]
            if val > 0:
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(x, y, str(int(val)), ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    ax.set_title("Robot_0 Death Position Heatmap\n(where robot_1 kills robot_0)", fontsize=11)
    ax.set_xlabel("X (column)")
    ax.set_ylabel("Y (row)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if charger_positions:
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=CHARGER_COLOR,
                          linewidth=2, label="Charger")
        ], fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_spawn_heatmap(spawn_count: np.ndarray, n: int,
                       charger_positions, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    vmax = max(spawn_count.max(), 1)
    im = ax.imshow(spawn_count, cmap="Blues", origin="upper", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Spawn count")
    _mark_chargers(ax, charger_positions)

    for y in range(n):
        for x in range(n):
            val = spawn_count[y, x]
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(x, y, str(int(val)), ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Robot_0 Spawn Position Distribution\n(confirms random spawn is working)", fontsize=10)
    ax.set_xlabel("X (column)")
    ax.set_ylabel("Y (row)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_survival_heatmap(avg_survival: np.ndarray, n: int,
                          charger_positions, max_steps: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    masked = np.ma.masked_invalid(avg_survival)
    im = ax.imshow(masked, cmap="RdYlGn", origin="upper", vmin=0, vmax=max_steps)
    plt.colorbar(im, ax=ax, label="Avg survival step")
    _mark_chargers(ax, charger_positions)

    for y in range(n):
        for x in range(n):
            val = avg_survival[y, x]
            if not np.isnan(val):
                ax.text(x, y, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_title("Avg Survival Time by Robot_0 Position\n(green = survives longer, red = dies sooner)",
                 fontsize=10)
    ax.set_xlabel("X (column)")
    ax.set_ylabel("Y (row)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robot_0 survival analysis: death step distribution and spatial heatmaps"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Checkpoint dir (e.g. ./models/p2_6x6/episode_183328_interrupted)")
    parser.add_argument("--env-n", type=int, default=6)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--robot-0-energy", type=int, default=20)
    parser.add_argument("--robot-1-energy", type=int, default=100)
    parser.add_argument("--robot-0-speed", type=int, default=1)
    parser.add_argument("--robot-1-speed", type=int, default=2)
    parser.add_argument("--charger-positions", type=str, default=None,
                        help='e.g. "3,3" or "1,1;3,3"')
    parser.add_argument("--exclusive-charging", action="store_true", default=False)
    parser.add_argument("--no-dust", action="store_true", default=False)
    parser.add_argument("--e-move", type=int, default=0)
    parser.add_argument("--e-charge", type=float, default=10.0)
    parser.add_argument("--e-collision", type=int, default=30)
    parser.add_argument("--e-boundary", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-episodes", type=int, default=500,
                        help="Number of episodes to sample")
    parser.add_argument("--scripted-robots", type=str, default="",
                        help='Comma-separated robot IDs to always STAY (matches training scripted mode), e.g. "0"')
    parser.add_argument("--safe-random-robots", type=str, default="",
                        help='Comma-separated robot IDs to use wall-avoiding random walk, e.g. "0"')
    parser.add_argument("--random-start-robots", type=str, default="",
                        help='Comma-separated robot IDs to randomize start position each episode, e.g. "0"')
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for plots (default: analysis/output/<checkpoint_name>)")
    args = parser.parse_args()

    # --- Setup output dir ---
    ckpt_name = Path(args.model_dir).name
    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(__file__).parent / "output" / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # --- Parse args ---
    charger_positions = parse_charger_positions(args.charger_positions)
    scripted_ids = set(
        int(x) for x in args.scripted_robots.split(",") if x.strip()
    ) if args.scripted_robots else set()

    safe_random_ids = set(
        int(x) for x in args.safe_random_robots.split(",") if x.strip()
    ) if args.safe_random_robots else set()

    random_start_ids = set(
        int(x) for x in args.random_start_robots.split(",") if x.strip()
    ) if args.random_start_robots else set()

    robot_energies = [args.robot_0_energy, args.robot_1_energy]
    robot_speeds   = [args.robot_0_speed,  args.robot_1_speed]

    # --- Build env ---
    env = RobotVacuumGymEnv(
        n=args.env_n,
        num_robots=args.num_robots,
        initial_energy=max(robot_energies),
        robot_energies=robot_energies,
        e_move=args.e_move,
        e_charge=args.e_charge,
        e_collision=args.e_collision,
        e_boundary=args.e_boundary,
        n_steps=args.max_steps,
        charger_positions=charger_positions,
        dust_enabled=not args.no_dust,
        exclusive_charging=args.exclusive_charging,
        robot_speeds=robot_speeds,
        random_start_robots=random_start_ids,
    )
    obs_dim = env.observation_space.shape[0]
    print(f"Env: {args.env_n}x{args.env_n}  obs_dim={obs_dim}")

    # --- Load Q-networks ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    q_nets = {}
    for i in range(args.num_robots):
        if i in safe_random_ids:
            q_nets[i] = None
            print(f"  robot_{i}: SAFE_RANDOM (no model loaded)")
        else:
            net = DQN(5, obs_dim).to(device)
            net.apply(init_weights)
            model_path = os.path.join(args.model_dir, f"robot_{i}.pt")
            if os.path.exists(model_path):
                net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                net.eval()
                print(f"  robot_{i}: loaded from {model_path}")
            else:
                print(f"  robot_{i}: WARNING — {model_path} not found, using random weights")
            q_nets[i] = net

    # --- Run episodes ---
    print(f"\nRunning {args.num_episodes} episodes (max_steps={args.max_steps})...")
    death_steps   = []
    death_positions = []
    survival_sum  = np.zeros((args.env_n, args.env_n), dtype=np.float64)
    spawn_count   = np.zeros((args.env_n, args.env_n), dtype=np.int64)
    death_grid    = np.zeros((args.env_n, args.env_n), dtype=np.int64)

    for ep in range(args.num_episodes):
        ds, dp, traj = run_episode(env, q_nets, safe_random_ids,
                                   args.max_steps, device,
                                   scripted_ids=scripted_ids)
        death_steps.append(ds)

        # 只記 spawn 位置（traj[0]），每 episode 一次，避免 STAY robot 重複計 1000 次
        if traj:
            sx, sy = traj[0]
            spawn_count[sy, sx] += 1
            survival_sum[sy, sx] += ds

        if dp is not None:
            death_grid[dp[1], dp[0]] += 1
            death_positions.append(dp)

        if (ep + 1) % max(1, args.num_episodes // 10) == 0:
            print(f"  [{ep+1}/{args.num_episodes}] done")

    death_steps = np.array(death_steps)
    survived    = int(np.sum(death_steps == args.max_steps))
    died        = args.num_episodes - survived
    died_steps  = death_steps[death_steps < args.max_steps]

    with np.errstate(invalid="ignore"):
        avg_survival = np.where(spawn_count > 0,
                                survival_sum / spawn_count,
                                np.nan)

    # --- Print summary ---
    print(f"\n{'='*50}")
    print(f"Episodes:  {args.num_episodes}")
    print(f"Survived:  {survived}  ({100*survived/args.num_episodes:.1f}%)")
    print(f"Died:      {died}  ({100*died/args.num_episodes:.1f}%)")
    if len(died_steps) > 0:
        print(f"Death step stats (died episodes only):")
        print(f"  mean   = {died_steps.mean():.1f}")
        print(f"  std    = {died_steps.std():.1f}")
        print(f"  min    = {died_steps.min()}")
        print(f"  25th   = {np.percentile(died_steps, 25):.0f}")
        print(f"  median = {np.median(died_steps):.0f}")
        print(f"  75th   = {np.percentile(died_steps, 75):.0f}")
        print(f"  max    = {died_steps.max()}")
    print(f"{'='*50}")

    # --- Generate plots ---
    print("\nGenerating plots...")
    plot_death_hist(death_steps, args.max_steps,
                    out_dir / "death_step_hist.png")
    plot_death_heatmap(death_grid, args.env_n, charger_positions,
                       out_dir / "death_pos_heatmap.png")
    plot_spawn_heatmap(spawn_count, args.env_n, charger_positions,
                       out_dir / "spawn_distribution.png")
    plot_survival_heatmap(avg_survival, args.env_n, charger_positions,
                          args.max_steps, out_dir / "survival_by_pos.png")

    print(f"\nDone. All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
