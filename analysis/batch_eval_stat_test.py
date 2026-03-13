"""
Batch Evaluation with Statistical Testing
Compare early vs late checkpoints on kill rates at epsilon=0
to determine if late-training aggression is learned or noise.

Usage:
  python analysis/batch_eval_stat_test.py \
    --checkpoint-early ./models/p4_4x4_shared/episode_500000 \
    --checkpoint-late  ./models/p4_4x4_shared/episode_5000000 \
    --num-episodes 500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from collections import defaultdict
from gym import RobotVacuumGymEnv
from dqn import DQN, build_network


class LightweightAgent:
    """Minimal agent that just does greedy inference."""
    def __init__(self, model_path, obs_dim, n_actions=5, device='cpu',
                 dueling=False, noisy=False, c51=False, num_atoms=51,
                 v_min=-100.0, v_max=100.0):
        self.device = device
        self.c51 = c51
        self.q_net = build_network(
            n_actions, obs_dim,
            dueling=dueling, noisy=noisy,
            c51=c51, num_atoms=num_atoms,
            v_min=v_min, v_max=v_max,
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.q_net.load_state_dict(checkpoint)
        self.q_net.eval()

    @torch.no_grad()
    def act(self, obs):
        """Greedy action selection (epsilon=0)."""
        t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        q = self.q_net(t)
        return q.argmax(1).item()


def make_env(args):
    """Create environment matching p4_4x4_shared config."""
    charger_positions = []
    for pos_str in args.charger_positions.split(';'):
        y, x = map(int, pos_str.split(','))
        charger_positions.append((y, x))

    robot_energies = [args.robot_0_energy, args.robot_1_energy,
                      args.robot_2_energy, args.robot_3_energy][:args.num_robots]
    robot_speeds = [args.robot_0_speed, args.robot_1_speed,
                    args.robot_2_speed, args.robot_3_speed][:args.num_robots]

    # All robots get random start positions for fair evaluation
    random_start_robots = set(range(args.num_robots))

    env = RobotVacuumGymEnv(
        n=args.env_n,
        num_robots=args.num_robots,
        initial_energy=100,
        robot_energies=robot_energies,
        e_move=args.e_move,
        e_charge=args.e_charge,
        e_collision=args.e_collision,
        e_boundary=args.e_boundary,
        n_steps=args.max_steps,
        charger_positions=charger_positions,
        dust_enabled=False,
        exclusive_charging=False,
        charger_range=1,
        robot_speeds=robot_speeds,
        random_start_robots=random_start_robots,
        legacy_obs=args.legacy_obs,
        agent_types_mode='off',
    )
    return env


def load_agents(model_dir, obs_dim, num_robots, device,
                dueling=False, noisy=False, c51=False,
                num_atoms=51, v_min=-100.0, v_max=100.0):
    """Load all robot agents from a checkpoint directory."""
    agents = []
    for i in range(num_robots):
        path = os.path.join(model_dir, f"robot_{i}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        agents.append(LightweightAgent(
            path, obs_dim, device=device,
            dueling=dueling, noisy=noisy, c51=c51,
            num_atoms=num_atoms, v_min=v_min, v_max=v_max,
        ))
    return agents


def run_episode(env, agents, max_steps):
    """
    Run one episode with sequential actions (matching training).
    Returns dict of per-episode stats.
    """
    obs, _ = env.reset()
    num_robots = len(agents)
    robot_speeds = env.env.robot_speeds

    # Track stats
    kills = defaultdict(int)          # kills[attacker] = count
    kill_victims = defaultdict(list)  # kill_victims[attacker] = [victim_ids]
    collisions = np.zeros((num_robots, num_robots), dtype=int)  # collisions[attacker][victim]
    deaths = set()
    total_rewards = np.zeros(num_robots)
    death_steps = {}

    # Track cumulative collision counts to compute deltas
    prev_active_col = np.zeros((num_robots, num_robots), dtype=int)

    for step in range(max_steps):
        for rid in range(num_robots):
            n_turns = robot_speeds[rid]
            for turn in range(n_turns):
                # Check if this robot is alive
                state = env.env.get_global_state()
                if not state['robots'][rid]['is_active']:
                    continue

                obs_r = env.get_observation(rid)
                action = agents[rid].act(obs_r)

                # Record alive robots before action
                alive_before = {i: state['robots'][i]['is_active'] for i in range(num_robots)}

                next_obs, reward, terminated, truncated, info = env.step_single(rid, action)
                total_rewards[rid] += reward

                # Track collision deltas from cumulative info
                for j in range(num_robots):
                    if j == rid:
                        continue
                    cur = info.get(f'active_collisions_with_{j}', 0)
                    delta = cur - prev_active_col[rid][j]
                    if delta > 0:
                        collisions[rid][j] += delta
                    prev_active_col[rid][j] = cur

                # Check who died from this action
                state_after = env.env.get_global_state()
                for j in range(num_robots):
                    if j == rid:
                        continue
                    if alive_before[j] and not state_after['robots'][j]['is_active']:
                        kills[rid] += 1
                        kill_victims[rid].append(j)
                        deaths.add(j)
                        death_steps[j] = step

                # Check if attacker also died
                if alive_before[rid] and not state_after['robots'][rid]['is_active']:
                    deaths.add(rid)
                    death_steps[rid] = step

        # Advance step
        max_reached, _ = env.advance_step()

        # Check termination
        alive_count = sum(1 for i in range(num_robots)
                         if env.env.get_global_state()['robots'][i]['is_active'])
        if alive_count <= 1 or max_reached:
            break

    return {
        'kills': dict(kills),
        'kill_victims': dict(kill_victims),
        'collisions': collisions,
        'deaths': deaths,
        'death_steps': death_steps,
        'total_rewards': total_rewards,
        'steps': step + 1,
        'survivors': [i for i in range(num_robots)
                      if env.env.get_global_state()['robots'][i]['is_active']],
    }


def run_batch(env, agents, num_episodes, max_steps, label=""):
    """Run many episodes, collect statistics."""
    all_results = []
    for ep in range(num_episodes):
        result = run_episode(env, agents, max_steps)
        all_results.append(result)
        if (ep + 1) % 100 == 0:
            print(f"  [{label}] {ep+1}/{num_episodes} episodes done")
    return all_results


def analyze_results(results, label, num_robots):
    """Compute aggregate statistics from batch results."""
    n = len(results)

    # Kill counts per attacker
    total_kills = defaultdict(int)
    total_kill_victims = defaultdict(lambda: defaultdict(int))  # [attacker][victim]
    total_collisions = np.zeros((num_robots, num_robots))
    episodes_with_any_kill = 0
    r0_kill_episodes = 0
    any_death_episodes = 0

    survivor_counts = defaultdict(int)

    for r in results:
        had_kill = False
        for attacker, count in r['kills'].items():
            total_kills[attacker] += count
            if count > 0:
                had_kill = True
                if attacker == 0:
                    r0_kill_episodes += 1
            for victim in r['kill_victims'].get(attacker, []):
                total_kill_victims[attacker][victim] += 1
        if had_kill:
            episodes_with_any_kill += 1
        if r['deaths']:
            any_death_episodes += 1
        total_collisions += r['collisions']

        surv_key = tuple(sorted(r['survivors']))
        survivor_counts[surv_key] += 1

    print(f"\n{'='*60}")
    print(f"  {label}  ({n} episodes)")
    print(f"{'='*60}")

    print(f"\n  Episodes with any kill: {episodes_with_any_kill}/{n} ({episodes_with_any_kill/n*100:.1f}%)")
    print(f"  Episodes with any death: {any_death_episodes}/{n} ({any_death_episodes/n*100:.1f}%)")
    print(f"  Episodes where r0 killed someone: {r0_kill_episodes}/{n} ({r0_kill_episodes/n*100:.1f}%)")

    print(f"\n  Total kills per attacker:")
    for i in range(num_robots):
        k = total_kills[i]
        print(f"    robot_{i}: {k} kills ({k/n:.2f} per episode)")

    print(f"\n  Kill matrix (attacker → victim):")
    for i in range(num_robots):
        victims_str = "  ".join(f"→r{j}:{total_kill_victims[i][j]}" for j in range(num_robots) if j != i)
        print(f"    r{i}: {victims_str}")

    print(f"\n  Collision matrix (attacker → victim, total across all episodes):")
    for i in range(num_robots):
        row = "  ".join(f"→r{j}:{int(total_collisions[i][j])}" for j in range(num_robots) if j != i)
        print(f"    r{i}: {row}")

    print(f"\n  Survivor distribution (top 5):")
    for surv, count in sorted(survivor_counts.items(), key=lambda x: -x[1])[:5]:
        robots = [f"r{i}" for i in surv] if surv else ["none"]
        print(f"    {','.join(robots)}: {count}/{n} ({count/n*100:.1f}%)")

    avg_rewards = np.mean([r['total_rewards'] for r in results], axis=0)
    print(f"\n  Average rewards: " + "  ".join(f"r{i}:{avg_rewards[i]:.1f}" for i in range(num_robots)))

    return {
        'n': n,
        'r0_kill_episodes': r0_kill_episodes,
        'episodes_with_any_kill': episodes_with_any_kill,
        'any_death_episodes': any_death_episodes,
        'total_kills': dict(total_kills),
        'total_collisions': total_collisions,
        'total_kill_victims': {k: dict(v) for k, v in total_kill_victims.items()},
    }


def statistical_tests(stats_early, stats_late, num_robots):
    """Run proportion z-tests and Fisher's exact test."""
    from scipy import stats as scipy_stats

    print(f"\n{'='*60}")
    print(f"  STATISTICAL TESTS")
    print(f"{'='*60}")

    n1, n2 = stats_early['n'], stats_late['n']

    tests = [
        ("r0 kill rate", stats_early['r0_kill_episodes'], stats_late['r0_kill_episodes']),
        ("Any kill rate", stats_early['episodes_with_any_kill'], stats_late['episodes_with_any_kill']),
        ("Any death rate", stats_early['any_death_episodes'], stats_late['any_death_episodes']),
    ]

    for name, x1, x2 in tests:
        p1, p2 = x1/n1, x2/n2

        # Two-proportion z-test (two-sided)
        p_pool = (x1 + x2) / (n1 + n2)
        if p_pool == 0 or p_pool == 1:
            z_stat, z_pval = 0, 1.0
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_stat = (p2 - p1) / se
            z_pval = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

        # Fisher's exact test
        table = [[x1, n1 - x1], [x2, n2 - x2]]
        _, fisher_pval = scipy_stats.fisher_exact(table)

        sig = "***" if z_pval < 0.001 else "**" if z_pval < 0.01 else "*" if z_pval < 0.05 else "n.s."

        print(f"\n  {name}:")
        print(f"    Early: {x1}/{n1} = {p1*100:.1f}%")
        print(f"    Late:  {x2}/{n2} = {p2*100:.1f}%")
        print(f"    Δ = {(p2-p1)*100:+.1f}%")
        print(f"    Z-test: z={z_stat:.3f}, p={z_pval:.4f} {sig}")
        print(f"    Fisher exact: p={fisher_pval:.4f} {sig}")

    # Per-attacker collision rate (Welch's t-test on per-episode collision counts)
    print(f"\n  --- Per-attacker collision count (Welch t-test) ---")
    # We'd need per-episode data for this, so skip if not available

    print(f"\n  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")


def main():
    parser = argparse.ArgumentParser(description="Batch eval with statistical testing")
    parser.add_argument("--checkpoint-early", type=str, required=True)
    parser.add_argument("--checkpoint-late", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)

    # Env config (defaults match p4_4x4_shared)
    parser.add_argument("--env-n", type=int, default=4)
    parser.add_argument("--num-robots", type=int, default=4)
    parser.add_argument("--robot-0-energy", type=int, default=100)
    parser.add_argument("--robot-1-energy", type=int, default=20)
    parser.add_argument("--robot-2-energy", type=int, default=20)
    parser.add_argument("--robot-3-energy", type=int, default=20)
    parser.add_argument("--robot-0-speed", type=int, default=2)
    parser.add_argument("--robot-1-speed", type=int, default=2)
    parser.add_argument("--robot-2-speed", type=int, default=1)
    parser.add_argument("--robot-3-speed", type=int, default=1)
    parser.add_argument("--charger-positions", type=str, default="2,2")
    parser.add_argument("--e-move", type=int, default=0)
    parser.add_argument("--e-charge", type=float, default=10)
    parser.add_argument("--e-collision", type=int, default=30)
    parser.add_argument("--e-boundary", type=int, default=0)
    parser.add_argument("--legacy-obs", action="store_true", default=True)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Episodes per checkpoint: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")

    # Create env to get obs_dim
    env = make_env(args)
    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim}")

    # Load agents for early checkpoint
    print(f"\nLoading EARLY checkpoint: {args.checkpoint_early}")
    agents_early = load_agents(args.checkpoint_early, obs_dim, args.num_robots, device)

    print(f"Loading LATE checkpoint: {args.checkpoint_late}")
    agents_late = load_agents(args.checkpoint_late, obs_dim, args.num_robots, device)

    # Run batch evaluations
    print(f"\n--- Running EARLY checkpoint ---")
    results_early = run_batch(env, agents_early, args.num_episodes, args.max_steps, "EARLY")

    print(f"\n--- Running LATE checkpoint ---")
    results_late = run_batch(env, agents_late, args.num_episodes, args.max_steps, "LATE")

    # Analyze
    stats_early = analyze_results(results_early, f"EARLY ({os.path.basename(args.checkpoint_early)})", args.num_robots)
    stats_late = analyze_results(results_late, f"LATE ({os.path.basename(args.checkpoint_late)})", args.num_robots)

    # Statistical tests
    statistical_tests(stats_early, stats_late, args.num_robots)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
