"""
Behavioral Analysis of Multi-Agent Training via W&B Data

Fetches training history from W&B and generates visualizations for:
1. Survival rates per robot
2. Attack patterns (who attacks whom, and how it evolves)
3. Pursuit vs Flee dynamics (robot_0 vs robot_1)
4. Weak robot resistance over time
5. Kill timeline and monopoly speed

Usage:
    python analysis/analyze_wandb_behavior.py --run-id xldcqgew --run-name p4_4x4_shared
    python analysis/analyze_wandb_behavior.py --run-id 981uroel --run-name p4_4x4_excl
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']  # r0=red, r1=blue, r2=green, r3=orange


def fetch_wandb_data(entity, project, run_id, n_samples=500):
    """Fetch history from W&B cloud."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    rows = run.history(samples=n_samples, pandas=False)
    rows = [r for r in rows if r.get("episode")]
    rows.sort(key=lambda r: r["episode"])
    print(f"Fetched {len(rows)} data points (max episode: {rows[-1]['episode']:.0f})")
    return rows


def bucket(rows, n_buckets=50):
    """Split rows into n_buckets by episode number, return list of (ep_mid, bucket_rows)."""
    max_ep = rows[-1]["episode"]
    size = max_ep / n_buckets
    result = []
    for b in range(n_buckets):
        lo, hi = b * size, (b + 1) * size
        bkt = [r for r in rows if lo <= r["episode"] < hi]
        if bkt:
            ep_mid = np.mean([r["episode"] for r in bkt])
            result.append((ep_mid, bkt))
    return result


def avg(bkt_rows, key):
    vals = [r[key] for r in bkt_rows if r.get(key) is not None]
    return np.mean(vals) if vals else np.nan


def make_plots(rows, run_name, out_dir, n_robots=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = bucket(rows, n_buckets=50)
    eps = [ep for ep, _ in buckets]

    # ── 1. Survival Rate ──────────────────────────────────────────────────
    fig, ax = plt.subplots()
    for i in range(n_robots):
        surv = [avg(bkt, f"robot_{i}/survived") for _, bkt in buckets]
        ax.plot(eps, surv, color=COLORS[i], label=f"robot_{i}", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Survival Rate")
    ax.set_title(f"{run_name} — Survival Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_survival.png")
    plt.close(fig)
    print(f"  Saved {run_name}_survival.png")

    # ── 2. Episode Reward ─────────────────────────────────────────────────
    fig, ax = plt.subplots()
    for i in range(n_robots):
        rwd = [avg(bkt, f"robot_{i}/episode_reward") for _, bkt in buckets]
        ax.plot(eps, rwd, color=COLORS[i], label=f"robot_{i}", linewidth=2)
    ax.axhline(y=-100, color='gray', linestyle='--', alpha=0.5, label='death penalty')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"{run_name} — Episode Rewards")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_rewards.png")
    plt.close(fig)
    print(f"  Saved {run_name}_rewards.png")

    # ── 3. robot_0 Attack Pattern (stacked area) ─────────────────────────
    fig, ax = plt.subplots()
    targets = {}
    for j in range(1, n_robots):
        targets[j] = [avg(bkt, f"robot_0/attacks_robot_{j}") for _, bkt in buckets]
    bottom = np.zeros(len(eps))
    for j in range(1, n_robots):
        vals = np.array(targets[j])
        vals = np.nan_to_num(vals, 0)
        ax.fill_between(eps, bottom, bottom + vals, alpha=0.6,
                         color=COLORS[j], label=f"r0 → r{j}")
        bottom += vals
    ax.plot(eps, bottom, color='black', linewidth=1.5, alpha=0.7, label='total')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Attacks per Episode")
    ax.set_title(f"{run_name} — robot_0 Attack Targets (stacked)")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_r0_attacks.png")
    plt.close(fig)
    print(f"  Saved {run_name}_r0_attacks.png")

    # ── 4. Pursuit vs Flee: robot_0 ↔ robot_1 ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    a01 = [avg(bkt, "robot_0/attacks_robot_1") for _, bkt in buckets]
    a10 = [avg(bkt, "robot_1/attacks_robot_0") for _, bkt in buckets]
    ax1.plot(eps, a01, color=COLORS[0], linewidth=2, label="r0 → r1 (attack)")
    ax1.plot(eps, a10, color=COLORS[1], linewidth=2, label="r1 → r0 (counter)")
    ax1.fill_between(eps, a01, a10, alpha=0.15, color='red',
                      where=[a > b for a, b in zip(a01, a10)], label='r0 dominates')
    ax1.fill_between(eps, a01, a10, alpha=0.15, color='blue',
                      where=[a <= b for a, b in zip(a01, a10)], label='r1 dominates')
    ax1.set_ylabel("Attacks per Episode")
    ax1.set_title(f"{run_name} — r0 vs r1: Attack Symmetry")
    ax1.legend(loc='upper right')

    r1_surv = [avg(bkt, "robot_1/survived") for _, bkt in buckets]
    r1_death = [avg(bkt, "robot_1/death_step") for _, bkt in buckets]
    ax2.plot(eps, r1_surv, color=COLORS[1], linewidth=2, label="r1 survival rate")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(eps, r1_death, color=COLORS[1], linewidth=2, linestyle='--',
                   alpha=0.6, label="r1 death step (when dies)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Survival Rate")
    ax2_twin.set_ylabel("Death Step")
    ax2.set_title(f"{run_name} — r1 Survival & Death Timing")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_r0_vs_r1.png")
    plt.close(fig)
    print(f"  Saved {run_name}_r0_vs_r1.png")

    # ── 5. Weak Robots Resistance ─────────────────────────────────────────
    if n_robots > 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for i in range(2, n_robots):
            coll = [avg(bkt, f"robot_{i}/active_collisions") for _, bkt in buckets]
            ax1.plot(eps, coll, color=COLORS[i], linewidth=2, label=f"r{i} active collisions")
        ax1.set_ylabel("Active Collisions per Episode")
        ax1.set_title(f"{run_name} — Weak Robots' Resistance (active collisions)")
        ax1.legend()

        for i in range(2, n_robots):
            surv = [avg(bkt, f"robot_{i}/survived") for _, bkt in buckets]
            ax2.plot(eps, surv, color=COLORS[i], linewidth=2, label=f"r{i} survival")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Survival Rate")
        ax2.set_title(f"{run_name} — Weak Robots' Survival")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
        fig.tight_layout()
        fig.savefig(out_dir / f"{run_name}_weak_resistance.png")
        plt.close(fig)
        print(f"  Saved {run_name}_weak_resistance.png")

    # ── 6. Monopoly Speed & Kill Order ────────────────────────────────────
    fig, ax = plt.subplots()
    mono = [avg(bkt, "robot_0/monopoly_step") for _, bkt in buckets]
    ax.plot(eps, mono, color=COLORS[0], linewidth=2.5, label="monopoly_step")
    for j in range(1, n_robots):
        ds = [avg(bkt, f"robot_{j}/death_step") for _, bkt in buckets]
        ax.plot(eps, ds, color=COLORS[j], linewidth=1.5, linestyle='--',
                 alpha=0.7, label=f"r{j} death_step")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Step")
    ax.set_title(f"{run_name} — Kill Timeline (when each robot dies)")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_kill_timeline.png")
    plt.close(fig)
    print(f"  Saved {run_name}_kill_timeline.png")

    # ── 7. All-Pair Attack Heatmap (late training) ────────────────────────
    late_buckets = buckets[-5:]  # last 10% of training
    late_rows = []
    for _, bkt in late_buckets:
        late_rows.extend(bkt)

    attack_matrix = np.zeros((n_robots, n_robots))
    for i in range(n_robots):
        for j in range(n_robots):
            if i != j:
                attack_matrix[i, j] = avg(late_rows, f"robot_{i}/attacks_robot_{j}")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attack_matrix, cmap='YlOrRd', aspect='auto')
    for i in range(n_robots):
        for j in range(n_robots):
            color = 'white' if attack_matrix[i, j] > attack_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{attack_matrix[i, j]:.1f}', ha='center', va='center',
                     fontsize=14, fontweight='bold', color=color)
    ax.set_xticks(range(n_robots))
    ax.set_yticks(range(n_robots))
    ax.set_xticklabels([f'r{i} (victim)' for i in range(n_robots)])
    ax.set_yticklabels([f'r{i} (attacker)' for i in range(n_robots)])
    ax.set_title(f"{run_name} — Attack Heatmap (late training, last 10%)")
    fig.colorbar(im, label='Avg attacks per episode')
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_attack_heatmap.png")
    plt.close(fig)
    print(f"  Saved {run_name}_attack_heatmap.png")

    print(f"\nAll plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Behavioral analysis from W&B training data")
    parser.add_argument("--run-id", type=str, required=True, help="W&B run ID")
    parser.add_argument("--run-name", type=str, required=True, help="Human-readable run name (for plot titles)")
    parser.add_argument("--entity", type=str, default="lazyhao-national-taiwan-university")
    parser.add_argument("--project", type=str, default="multi-robot-idqn")
    parser.add_argument("--n-robots", type=int, default=4)
    parser.add_argument("--samples", type=int, default=500, help="Number of data points to fetch from W&B")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: analysis/output/<run_name>)")
    args = parser.parse_args()

    out_dir = args.out_dir or f"analysis/output/{args.run_name}"

    print(f"Fetching data for {args.run_name} (run_id: {args.run_id})...")
    rows = fetch_wandb_data(args.entity, args.project, args.run_id, args.samples)

    print(f"Generating plots...")
    make_plots(rows, args.run_name, out_dir, n_robots=args.n_robots)


if __name__ == "__main__":
    main()
