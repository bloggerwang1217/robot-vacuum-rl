#!/usr/bin/env python3
"""Quick batch evaluation: run N episodes in parallel using batch_env, report kill stats."""
import argparse
import numpy as np
import torch
from batch_env import BatchRobotVacuumEnv
from dqn import DQN, build_network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--env-n", type=int, default=5)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--charger-positions", type=str, default="2,2")
    parser.add_argument("--charger-range", type=int, default=1)
    parser.add_argument("--e-move", type=float, default=0)
    parser.add_argument("--e-charge", type=float, default=8)
    parser.add_argument("--e-collision", type=float, default=100)
    parser.add_argument("--e-boundary", type=float, default=0)
    parser.add_argument("--e-decay", type=float, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--random-robots", type=str, default="")
    parser.add_argument("--seek-charger-robots", type=str, default="")
    parser.add_argument("--eval-epsilon", type=float, default=0.01)
    parser.add_argument("--no-dust", action="store_true", default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--robot-0-energy", type=int, default=100)
    parser.add_argument("--robot-1-energy", type=int, default=100)
    parser.add_argument("--robot-2-energy", type=int, default=100)
    parser.add_argument("--robot-3-energy", type=int, default=100)
    parser.add_argument("--robot-0-attack-power", type=float, default=None)
    parser.add_argument("--robot-1-attack-power", type=float, default=None)
    parser.add_argument("--robot-2-attack-power", type=float, default=None)
    parser.add_argument("--robot-3-attack-power", type=float, default=None)
    parser.add_argument("--robot-start-positions", type=str, default=None,
                        help='Fixed start positions "y0,x0;y1,x1"')
    parser.add_argument("--scripted-robots", type=str, default="",
                        help='Comma-separated robot IDs that always STAY')
    parser.add_argument("--exclusive-charging", action="store_true", default=False)
    parser.add_argument("--shuffle-step-order", action="store_true", default=False)
    parser.add_argument("--dueling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--c51", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Parse charger positions
    charger_positions = []
    for pos_str in args.charger_positions.split(';'):
        y, x = map(int, pos_str.split(','))
        charger_positions.append((y, x))

    random_robots = set(int(x) for x in args.random_robots.split(',') if x.strip()) if args.random_robots else set()
    seek_charger_robots = set(int(x) for x in args.seek_charger_robots.split(',') if x.strip()) if args.seek_charger_robots else set()
    scripted_robots = set(int(x) for x in args.scripted_robots.split(',') if x.strip()) if args.scripted_robots else set()

    # Parse attack powers
    all_attack = [args.robot_0_attack_power, args.robot_1_attack_power,
                  args.robot_2_attack_power, args.robot_3_attack_power][:args.num_robots]
    robot_attack_powers = all_attack if any(p is not None for p in all_attack) else None

    # Parse start positions
    robot_start_positions = {}
    if args.robot_start_positions:
        for i, pos_str in enumerate(args.robot_start_positions.split(';')):
            y, x = map(int, pos_str.split(','))
            robot_start_positions[i] = (y, x)

    N = args.num_episodes  # run all episodes in parallel
    env_kwargs = {
        'n': args.env_n,
        'num_robots': args.num_robots,
        'n_steps': args.max_steps,
        'charger_positions': charger_positions,
        'charger_range': args.charger_range,
        'e_move': args.e_move,
        'e_charge': args.e_charge,
        'e_collision': args.e_collision,
        'e_boundary': args.e_boundary,
        'e_decay': args.e_decay,
        'dust_enabled': False,
        'robot_energies': [args.robot_0_energy, args.robot_1_energy, args.robot_2_energy, args.robot_3_energy][:args.num_robots],
        'robot_speeds': [1] * args.num_robots,
        'random_start_robots': set(range(args.num_robots)) if not robot_start_positions else set(),
        'robot_start_positions': robot_start_positions,
        'robot_attack_powers': robot_attack_powers,
        'exclusive_charging': args.exclusive_charging,
    }
    env = BatchRobotVacuumEnv(num_envs=N, env_kwargs=env_kwargs)

    # Load models
    obs_dim = env.get_observation(0).shape[1]
    models = {}
    for rid in range(args.num_robots):
        if rid in random_robots or rid in scripted_robots or rid in seek_charger_robots:
            continue
        net = build_network(num_actions=5, input_dim=obs_dim,
                            dueling=args.dueling, noisy=args.noisy, c51=args.c51).to(device)
        path = f"{args.model_dir}/robot_{rid}.pt"
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        net.eval()
        models[rid] = net

    # Reset env to randomize start positions
    env.reset()

    # Run episodes
    R = args.num_robots
    total_collisions = np.zeros((N, R), dtype=np.int32)  # per-robot total hits given
    collision_matrix = np.zeros((N, R, R), dtype=np.int32)  # [env, attacker, victim]
    alive_at_end = np.zeros((N, R), dtype=bool)
    env_ended = np.zeros(N, dtype=bool)
    death_step = np.full((N, R), args.max_steps, dtype=np.int32)  # step when each robot died

    for step in range(args.max_steps):
        robot_order = list(range(args.num_robots))
        if args.shuffle_step_order:
            np.random.shuffle(robot_order)
        for rid in robot_order:
            if rid in scripted_robots:
                actions = np.full(N, 4, dtype=np.int32)  # STAY
            elif rid in random_robots:
                actions = np.random.randint(0, 5, size=N).astype(np.int32)
            elif rid in seek_charger_robots:
                obs = env.get_observation(rid)
                ccol = 3 + (args.num_robots - 1) * 3
                cdx, cdy = obs[:, ccol], obs[:, ccol+1]
                at = (np.abs(cdx) < 0.01) & (np.abs(cdy) < 0.01)
                h_act = np.where(cdx > 0.01, 3, np.where(cdx < -0.01, 2, 4))
                v_act = np.where(cdy > 0.01, 1, np.where(cdy < -0.01, 0, 4))
                actions = np.where(np.abs(cdx) >= np.abs(cdy), h_act, v_act)
                actions[at] = 4
                actions = actions.astype(np.int32)
            else:
                obs = env.get_observation(rid)
                obs_t = torch.from_numpy(obs).to(device)
                with torch.no_grad():
                    q = models[rid](obs_t)
                    actions = q.argmax(dim=1).cpu().numpy().astype(np.int32)
                # epsilon-greedy
                rand_mask = np.random.random(N) < args.eval_epsilon
                actions[rand_mask] = np.random.randint(0, 5, size=rand_mask.sum())

            env.step_single(rid, actions)

        # Accumulate collisions BEFORE advance_step resets
        for rid in range(R):
            for j in range(R):
                if j == rid:
                    continue
                hits = env.active_collisions_with[:, rid, j]
                total_collisions[:, rid] += hits
                collision_matrix[:, rid, j] += hits

        # Track death steps
        alive = env.alive  # (N, R)
        for rid in range(R):
            just_died = ~alive[:, rid] & (death_step[:, rid] == args.max_steps) & ~env_ended
            death_step[just_died, rid] = step

        # Track alive at episode end
        all_dead = ~alive.any(axis=1)
        is_last_step = (step == args.max_steps - 1)
        ending_now = (all_dead | is_last_step) & ~env_ended
        alive_at_end[ending_now] = alive[ending_now]
        env_ended |= ending_now

        done_mask, _ = env.advance_step()

    # === Report Results ===
    print(f"\n=== Eval: {args.model_dir} ({N} episodes) ===")

    # Per-robot survival & sole survivor
    num_alive = alive_at_end.sum(axis=1)  # (N,)
    print(f"\n--- Survival ---")
    for rid in range(R):
        survived = alive_at_end[:, rid].sum()
        sole = (alive_at_end[:, rid] & (num_alive == 1)).sum()
        avg_death = death_step[~alive_at_end[:, rid], rid].mean() if (~alive_at_end[:, rid]).any() else float('nan')
        print(f"  r{rid}: survived {survived:4d} ({100*survived/N:.1f}%) | sole winner {sole:4d} ({100*sole/N:.1f}%) | avg death step {avg_death:.0f}")

    all_dead_count = (num_alive == 0).sum()
    multi_alive = (num_alive > 1).sum()
    sole_winner_total = (num_alive == 1).sum()
    print(f"\n  Sole winner:  {sole_winner_total:4d} ({100*sole_winner_total/N:.1f}%)")
    print(f"  All dead:     {all_dead_count:4d} ({100*all_dead_count/N:.1f}%)")
    print(f"  Multi alive:  {multi_alive:4d} ({100*multi_alive/N:.1f}%)")

    # Per-robot collision stats
    print(f"\n--- Collisions (hits given) ---")
    for rid in range(R):
        total = total_collisions[:, rid].sum()
        avg = total_collisions[:, rid].mean()
        active_eps = (total_collisions[:, rid] > 0).sum()
        print(f"  r{rid}: {total:5d} total (avg {avg:.2f}/ep, in {active_eps}/{N} eps)")

    # Domination: sole survivor who actively hit all opponents
    print(f"\n--- Domination (sole survivor + hit all opponents) ---")
    for rid in range(R):
        sole_mask = alive_at_end[:, rid] & (num_alive == 1)
        sole_count = sole_mask.sum()
        if sole_count > 0:
            # Count how many distinct opponents this robot hit
            opp_hit = np.zeros(sole_count, dtype=np.int32)
            idx = 0
            for j in range(R):
                if j == rid:
                    continue
                opp_hit += (collision_matrix[sole_mask, rid, j] > 0).astype(np.int32)
            dominated = (opp_hit == R - 1).sum()
            print(f"  r{rid}: sole winner {sole_count:4d} | dominated (hit all) {dominated:4d} ({100*dominated/sole_count:.1f}% of wins)")
        else:
            print(f"  r{rid}: sole winner    0")

if __name__ == "__main__":
    main()
