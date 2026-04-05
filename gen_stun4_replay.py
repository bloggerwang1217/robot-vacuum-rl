"""
Generate replay JSONs from stun4_r0_vs_seeker checkpoint.
r0: greedy DQN policy (epsilon=0.01), r1: seek-charger heuristic.
Produces 3 replays with different random seeds.
"""

import json
import os
import numpy as np
import random
import torch
from dqn import build_network
from batch_env import BatchRobotVacuumEnv

# ── Config ────────────────────────────────────────────────────────────────
CHECKPOINT = "./models/stun4_higheps/episode_2200000/robot_0.pt"
OUTPUT_DIR = "./models/stun4_higheps"
SEEDS = [42, 123, 7, 99, 256]
MAX_STEPS = 500
EPSILON = 0.01

GRID_N = 5
NUM_ROBOTS = 2
CHARGER_POS = [(2, 2)]  # (y, x)
CHARGER_RANGE = 0

R0_ENERGY = 100
R1_ENERGY = 100
R0_SPEED = 2
R1_SPEED = 1
R0_ATK = 30
R1_ATK = 2
R0_DOCKING = 2
R1_DOCKING = 0
R0_STUN = 4
R1_STUN = 0

E_MOVE = 0
E_CHARGE = 8
E_DECAY = 0.5
E_BOUNDARY = 0
E_COLLISION = 30  # default, but attack_powers override per-robot
EXCLUSIVE_CHARGING = True
NO_DUST = True

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


def make_env():
    env_kwargs = {
        'n': GRID_N,
        'num_robots': NUM_ROBOTS,
        'n_steps': MAX_STEPS,
        'e_move': E_MOVE,
        'e_charge': E_CHARGE,
        'e_collision': E_COLLISION,
        'e_boundary': E_BOUNDARY,
        'e_decay': E_DECAY,
        'exclusive_charging': EXCLUSIVE_CHARGING,
        'charger_range': CHARGER_RANGE,
        'charger_positions': CHARGER_POS,
        'dust_enabled': not NO_DUST,
        'initial_energy': 100,
        'robot_energies': [R0_ENERGY, R1_ENERGY],
        'robot_speeds': [R0_SPEED, R1_SPEED],
        'robot_attack_powers': [R0_ATK, R1_ATK],
        'robot_docking_steps': [R0_DOCKING, R1_DOCKING],
        'robot_stun_steps': [R0_STUN, R1_STUN],
        'random_start_robots': {0, 1},
        'reward_mode': 'delta-energy',
        'reward_alpha': 0.05,
        'agent_types_mode': 'off',
    }
    return BatchRobotVacuumEnv(num_envs=1, env_kwargs=env_kwargs)


def load_model(env):
    device = torch.device("cpu")
    obs_dim = env.obs_dim  # should be 12
    net = build_network(num_actions=5, input_dim=obs_dim)
    state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net, device


def seek_charger_action(env):
    """Seek-charger heuristic for robot 1 (single env)."""
    obs = env.get_observation(1)  # (1, obs_dim)
    # charger_col = 3 + 4 + (R-1)*3 = 10 for 2 robots, agent_types_mode=off
    charger_col = 3 + 4 + 1 * 3  # = 10
    cdx = obs[0, charger_col]      # positive = charger is RIGHT
    cdy = obs[0, charger_col + 1]  # positive = charger is DOWN

    if abs(cdx) < 0.01 and abs(cdy) < 0.01:
        return 4  # STAY

    if abs(cdx) >= abs(cdy):
        if cdx > 0.01:
            return 3  # RIGHT
        elif cdx < -0.01:
            return 2  # LEFT
        else:
            return 4
    else:
        if cdy > 0.01:
            return 1  # DOWN
        elif cdy < -0.01:
            return 0  # UP
        else:
            return 4


def select_r0_action(net, device, obs, epsilon):
    """Epsilon-greedy action for r0."""
    if random.random() < epsilon:
        return random.randint(0, 4)
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().to(device)  # (1, obs_dim)
        q_vals = net(obs_t)  # (1, 5)
        return int(q_vals.argmax(dim=1).item())


def get_q_values(net, device, obs):
    """Get Q-values dict for r0."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().to(device)
        q_vals = net(obs_t)[0].cpu().numpy()
    return {ACTION_NAMES[i]: float(q_vals[i]) for i in range(5)}


def robot_state(env, rid):
    """Extract robot state as dict with [x, y] position format."""
    y = int(env.pos[0, rid, 0])
    x = int(env.pos[0, rid, 1])
    e = float(env.energy[0, rid])
    dead = not bool(env.alive[0, rid])
    stun = int(env.stun_counter[0, rid]) if env.stun_enabled else 0
    return {"position": [x, y], "energy": round(e, 4), "is_dead": dead, "stun": stun}


def all_robots_state(env):
    return {f"robot_{i}": robot_state(env, i) for i in range(NUM_ROBOTS)}


def run_episode(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    net, device = load_model(env)
    env.reset()

    config = {
        "grid_size": GRID_N,
        "num_robots": NUM_ROBOTS,
        "charger_positions": [[cx, cy] for (cy, cx) in CHARGER_POS],  # [x,y] in JSON
        "robot_initial_energies": {"robot_0": R0_ENERGY, "robot_1": R1_ENERGY},
        "agent_types_mode": "off",
        "agent_types": {},
        "heterotype_charge_mode": "off",
        "heterotype_charge_factor": 1.0,
        "parameters": {
            "e_move": E_MOVE,
            "e_collision": E_COLLISION,
            "e_boundary": E_BOUNDARY,
            "e_charge": E_CHARGE,
            "e_decay": E_DECAY,
            "charger_range": CHARGER_RANGE,
            "exclusive_charging": EXCLUSIVE_CHARGING,
            "robot_speeds": [R0_SPEED, R1_SPEED],
            "robot_attack_powers": [R0_ATK, R1_ATK],
            "robot_docking_steps": [R0_DOCKING, R1_DOCKING],
            "robot_stun_steps": [R0_STUN, R1_STUN],
        },
    }

    steps_log = []

    for step_idx in range(MAX_STEPS):
        # Check if all dead
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

        # Randomize turn order (shuffle_step_order=True)
        robot_order = list(range(NUM_ROBOTS))
        random.shuffle(robot_order)

        for robot_id in robot_order:
            speed = [R0_SPEED, R1_SPEED][robot_id]
            agent_id = f"robot_{robot_id}"

            for turn_idx in range(speed):
                is_last_turn = (turn_idx == speed - 1)

                if not env.alive[0, robot_id]:
                    continue

                robots_before = all_robots_state(env)

                # Select action
                obs = env.get_observation(robot_id)  # (1, obs_dim)
                if robot_id == 0:
                    action = select_r0_action(net, device, obs, EPSILON)
                    qv = get_q_values(net, device, obs)
                else:
                    action = seek_charger_action(env)
                    qv = {}

                actions_arr = np.array([action], dtype=np.int32)
                _, rewards, terminated, _, _ = env.step_single(
                    robot_id, actions_arr, is_last_turn=is_last_turn
                )

                robots_after = all_robots_state(env)
                reward = float(rewards[0])

                sub_step = {
                    "robot_id": robot_id,
                    "agent_id": agent_id,
                    "turn": turn_idx,
                    "action": ACTION_NAMES[action],
                    "q_values": qv,
                    "robots_before": robots_before,
                    "robots_after": robots_after,
                    "reward": round(reward, 4),
                    "terminated": bool(terminated[0]),
                }
                step_record["sub_steps"].append(sub_step)

            # Record action/reward/q_values at step level (last turn values)
            step_record["actions"][agent_id] = ACTION_NAMES[action] if env.alive[0, robot_id] or terminated[0] else "DEAD"
            step_record["rewards"][agent_id] = round(float(rewards[0]), 4)
            if robot_id == 0:
                step_record["q_values"][agent_id] = qv

        steps_log.append(step_record)

        # Advance step (check termination)
        done_mask, _ = env.advance_step()
        if done_mask[0]:
            break

    replay = {"config": config, "steps": steps_log}
    return replay


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, seed in enumerate(SEEDS):
        print(f"Generating replay {i+1}/3 (seed={seed})...")
        replay = run_episode(seed)
        n_steps = len(replay["steps"])

        # Check if r1 died
        r1_died = False
        death_step = None
        for s in replay["steps"]:
            for sub in s["sub_steps"]:
                if sub["robots_after"]["robot_1"]["is_dead"] and not sub["robots_before"]["robot_1"]["is_dead"]:
                    r1_died = True
                    death_step = s["step"]
                    break
            if r1_died:
                break

        fname = f"stun4_replay_seed{seed}.json"
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w") as f:
            json.dump(replay, f, indent=2)

        status = f"r1 died at step {death_step}" if r1_died else "r1 survived"
        print(f"  Saved: {fpath} ({n_steps} steps, {status})")


if __name__ == "__main__":
    main()
