"""
通用 replay JSON 產生器。
JSON 格式與 gen_stun4_replay.py 完全相同，包含 stun 欄位。

用法範例:
  python gen_replay.py \
    --model-dir ./models/stun5_r0_vs_pest_v2/episode_7140001 \
    --output-dir ./analyze/stun5_r0_vs_pest_v2 \
    --output-prefix pest_v2_final \
    --seeds 42,123,7,99,256 \
    --r1-policy model \
    --robot-0-stun-steps 5 --robot-1-stun-steps 1 \
    --e-move 1
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from batch_env import BatchRobotVacuumEnv
from dqn import build_network

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate replay JSONs from DQN checkpoints")

    # Checkpoint / output
    p.add_argument("--model-dir", required=True,
                   help="Checkpoint directory containing robot_0.pt (and optionally robot_1.pt)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write JSON files (default: same as --model-dir)")
    p.add_argument("--output-prefix", default="replay",
                   help="Filename prefix, e.g. 'pest_v2_final' → pest_v2_final_seed42.json")
    p.add_argument("--seeds", default="42,123,7,99,256",
                   help="Comma-separated random seeds (default: 42,123,7,99,256)")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--epsilon", type=float, default=0.01, help="Eval epsilon (default 0.01)")

    # Network architecture
    p.add_argument("--dueling", action="store_true", default=False)
    p.add_argument("--noisy", action="store_true", default=False)
    p.add_argument("--c51", action="store_true", default=False)

    # r0 / r1 policy
    p.add_argument("--r0-policy", choices=["model", "random"],
                   default="model",
                   help="r0 behaviour: model (DQN from checkpoint) / random")
    p.add_argument("--r1-policy", choices=["seek_charger", "random", "model"],
                   default="seek_charger",
                   help="r1 behaviour: seek_charger heuristic / random / DQN model from checkpoint")

    # Environment
    p.add_argument("--env-n", type=int, default=5)
    p.add_argument("--charger-positions", type=str, default="2,2",
                   help="Semicolon-separated y,x pairs, e.g. '2,2' or '1,1;3,3'")
    p.add_argument("--charger-range", type=int, default=0)
    p.add_argument("--exclusive-charging", action="store_true", default=True)
    p.add_argument("--no-dust", action="store_true", default=True)

    p.add_argument("--robot-0-energy", type=int, default=100)
    p.add_argument("--robot-1-energy", type=int, default=100)
    p.add_argument("--robot-0-speed", type=int, default=2)
    p.add_argument("--robot-1-speed", type=int, default=1)
    p.add_argument("--robot-0-attack", type=float, default=30.0)
    p.add_argument("--robot-1-attack", type=float, default=2.0)
    p.add_argument("--robot-0-docking-steps", type=int, default=2)
    p.add_argument("--robot-1-docking-steps", type=int, default=0)
    p.add_argument("--robot-0-stun-steps", type=int, default=0)
    p.add_argument("--robot-1-stun-steps", type=int, default=0)

    p.add_argument("--e-move", type=float, default=0.0)
    p.add_argument("--e-charge", type=float, default=8.0)
    p.add_argument("--e-decay", type=float, default=0.5)
    p.add_argument("--e-boundary", type=float, default=0.0)
    p.add_argument("--e-collision", type=float, default=30.0)

    p.add_argument("--reward-alpha", type=float, default=0.05)

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_charger_positions(s):
    positions = []
    for pair in s.split(";"):
        y, x = pair.strip().split(",")
        positions.append((int(y), int(x)))
    return positions


def make_env(args):
    charger_pos = parse_charger_positions(args.charger_positions)
    return BatchRobotVacuumEnv(num_envs=1, env_kwargs={
        'n': args.env_n,
        'num_robots': 2,
        'n_steps': args.max_steps,
        'e_move': args.e_move,
        'e_charge': args.e_charge,
        'e_collision': args.e_collision,
        'e_boundary': args.e_boundary,
        'e_decay': args.e_decay,
        'exclusive_charging': args.exclusive_charging,
        'charger_range': args.charger_range,
        'charger_positions': charger_pos,
        'dust_enabled': not args.no_dust,
        'initial_energy': 100,
        'robot_energies': [args.robot_0_energy, args.robot_1_energy],
        'robot_speeds': [args.robot_0_speed, args.robot_1_speed],
        'robot_attack_powers': [args.robot_0_attack, args.robot_1_attack],
        'robot_docking_steps': [args.robot_0_docking_steps, args.robot_1_docking_steps],
        'robot_stun_steps': [args.robot_0_stun_steps, args.robot_1_stun_steps],
        'random_start_robots': {0, 1},
        'reward_mode': 'delta-energy',
        'reward_alpha': args.reward_alpha,
        'agent_types_mode': 'off',
    })


def load_net(path, obs_dim, dueling=False, noisy=False, c51=False):
    net = build_network(num_actions=5, input_dim=obs_dim, dueling=dueling, noisy=noisy, c51=c51)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def robot_state(env, rid):
    """JSON 格式與 gen_stun4_replay.py 相同，增加 stun 欄位。"""
    y = int(env.pos[0, rid, 0])
    x = int(env.pos[0, rid, 1])
    e = float(env.energy[0, rid])
    dead = not bool(env.alive[0, rid])
    stun = int(env.stun_counter[0, rid]) if env.stun_enabled else 0
    return {"position": [x, y], "energy": round(e, 4), "is_dead": dead, "stun": stun}


def all_robots_state(env):
    return {f"robot_{i}": robot_state(env, i) for i in range(2)}


def seek_charger_action(env, charger_col=10):
    obs = env.get_observation(1)
    cdx, cdy = obs[0, charger_col], obs[0, charger_col + 1]
    if abs(cdx) < 0.01 and abs(cdy) < 0.01:
        return 4
    if abs(cdx) >= abs(cdy):
        return 3 if cdx > 0.01 else (2 if cdx < -0.01 else 4)
    return 1 if cdy > 0.01 else (0 if cdy < -0.01 else 4)


def select_action(net, obs, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 4)
    with torch.no_grad():
        return int(net(torch.from_numpy(obs).float()).argmax(dim=1).item())


def get_q_values(net, obs):
    with torch.no_grad():
        q = net(torch.from_numpy(obs).float())[0].cpu().numpy()
    return {ACTION_NAMES[i]: float(q[i]) for i in range(5)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = args.output_dir or args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    charger_pos = parse_charger_positions(args.charger_positions)
    env = make_env(args)
    obs_dim = env.obs_dim

    # Load r0 model
    r0_net = None
    if args.r0_policy == "model":
        r0_path = os.path.join(args.model_dir, "robot_0.pt")
        r0_net = load_net(r0_path, obs_dim, args.dueling, args.noisy, args.c51)
        print(f"Loaded r0: {r0_path}")
    else:
        print("r0: random policy")

    # Load r1 model if needed
    r1_net = None
    if args.r1_policy == "model":
        r1_path = os.path.join(args.model_dir, "robot_1.pt")
        r1_net = load_net(r1_path, obs_dim, args.dueling, args.noisy, args.c51)
        print(f"Loaded r1: {r1_path}")

    # Build config block (same structure as gen_stun4_replay.py)
    config = {
        "grid_size": args.env_n,
        "num_robots": 2,
        "charger_positions": [[x, y] for (y, x) in charger_pos],
        "robot_initial_energies": {
            "robot_0": args.robot_0_energy,
            "robot_1": args.robot_1_energy,
        },
        "agent_types_mode": "off",
        "agent_types": {},
        "heterotype_charge_mode": "off",
        "heterotype_charge_factor": 1.0,
        "parameters": {
            "e_move": args.e_move,
            "e_collision": args.e_collision,
            "e_boundary": args.e_boundary,
            "e_charge": args.e_charge,
            "e_decay": args.e_decay,
            "charger_range": args.charger_range,
            "exclusive_charging": args.exclusive_charging,
            "robot_speeds": [args.robot_0_speed, args.robot_1_speed],
            "robot_attack_powers": [args.robot_0_attack, args.robot_1_attack],
            "robot_docking_steps": [args.robot_0_docking_steps, args.robot_1_docking_steps],
            "robot_stun_steps": [args.robot_0_stun_steps, args.robot_1_stun_steps],
        },
    }

    speeds = [args.robot_0_speed, args.robot_1_speed]

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        env.reset()
        steps_log = []

        for step_idx in range(args.max_steps):
            if not env.alive[0].any():
                break

            step_record = {
                "step": step_idx,
                "actions": {},
                "rewards": {},
                "q_values": {},
                "robots": all_robots_state(env),
                "sub_steps": [],
                "events": [],
            }

            robot_order = list(range(2))
            random.shuffle(robot_order)

            for robot_id in robot_order:
                speed = speeds[robot_id]
                agent_id = f"robot_{robot_id}"
                action = 4; reward = 0.0; terminated = False; qv = {}

                for turn_idx in range(speed):
                    is_last = (turn_idx == speed - 1)
                    if not env.alive[0, robot_id]:
                        continue

                    robots_before = all_robots_state(env)
                    obs = env.get_observation(robot_id)

                    if robot_id == 0:
                        if args.r0_policy == "random":
                            action = random.randint(0, 4)
                            qv = {}
                        else:
                            action = select_action(r0_net, obs, args.epsilon)
                            qv = get_q_values(r0_net, obs)
                    else:
                        if args.r1_policy == "seek_charger":
                            action = seek_charger_action(env)
                            qv = {}
                        elif args.r1_policy == "random":
                            action = random.randint(0, 4)
                            qv = {}
                        else:  # model
                            action = select_action(r1_net, obs, args.epsilon)
                            qv = get_q_values(r1_net, obs)

                    acts = np.array([action], dtype=np.int32)
                    _, rewards, terms, _, _ = env.step_single(
                        robot_id, acts, is_last_turn=is_last
                    )
                    robots_after = all_robots_state(env)
                    reward = float(rewards[0])
                    terminated = bool(terms[0])

                    step_record["sub_steps"].append({
                        "robot_id": robot_id,
                        "agent_id": agent_id,
                        "turn": turn_idx,
                        "action": ACTION_NAMES[action],
                        "q_values": qv,
                        "robots_before": robots_before,
                        "robots_after": robots_after,
                        "reward": round(reward, 4),
                        "terminated": terminated,
                    })

                step_record["actions"][agent_id] = (
                    ACTION_NAMES[action] if env.alive[0, robot_id] or terminated else "DEAD"
                )
                step_record["rewards"][agent_id] = round(reward, 4)
                step_record["q_values"][agent_id] = qv

            steps_log.append(step_record)
            done_mask, _ = env.advance_step()
            if done_mask[0]:
                break

        # Summary
        n_steps = len(steps_log)
        r1_died = False
        death_step = None
        for s in steps_log:
            for sub in s["sub_steps"]:
                if (sub["robots_after"]["robot_1"]["is_dead"]
                        and not sub["robots_before"]["robot_1"]["is_dead"]):
                    r1_died = True
                    death_step = s["step"]
                    break
            if r1_died:
                break

        charger_y, charger_x = charger_pos[0]
        charger_steps = sum(
            1 for s in steps_log
            if s["robots"]["robot_1"]["position"] == [charger_x, charger_y]
            and not s["robots"]["robot_1"]["is_dead"]
        )
        charger_pct = charger_steps / max(n_steps, 1) * 100

        fname = f"{args.output_prefix}_seed{seed}.json"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w") as f:
            json.dump({"config": config, "steps": steps_log}, f, indent=2)

        status = f"r1 died @ step {death_step}" if r1_died else "r1 survived"
        print(f"  seed={seed}: {n_steps} steps, charger={charger_pct:.1f}%, {status}  →  {fname}")


if __name__ == "__main__":
    main()
