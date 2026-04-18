"""
QA tests for the new friendly collision energy-averaging mechanic.

Covers both training path (BatchRobotVacuumEnv) and eval path (RobotVacuumGymEnv).

Mechanic under test:
  When ally rid moves into ally j's cell:
    - Position resolves normally (knockback or stationary_blocked)
    - No damage, no stun
    - NEW: avg_e = (energy[rid] + energy[j]) / 2
           both get avg_e; j is capped immediately to its own cap.

Charge sharing has been REMOVED. Old CLI flags accepted but ignored.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from batch_env import BatchRobotVacuumEnv
from gym import RobotVacuumGymEnv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_batch_env(N=1, num_robots=2, robot_energies=None,
                   robot_attack_powers=None, alliance_groups=None,
                   robot_start_positions=None, exclusive_charging=False,
                   stun_steps=5, charger_positions=None):
    if robot_energies is None:
        robot_energies = [100] * num_robots
    if robot_attack_powers is None:
        robot_attack_powers = [30] * num_robots
    if alliance_groups is None:
        alliance_groups = []
    if robot_start_positions is None:
        robot_start_positions = {0: (0, 0), 1: (0, 1)}
    if charger_positions is None:
        charger_positions = [(3, 3)]
    cfg = {
        'n': 6, 'num_robots': num_robots, 'n_steps': 200,
        'e_move': 0, 'e_charge': 10, 'e_collision': 30, 'e_boundary': 0,
        'exclusive_charging': exclusive_charging,
        'charger_positions': charger_positions,
        'dust_enabled': False,
        'robot_speeds': [1] * num_robots,
        'robot_energies': robot_energies,
        'robot_attack_powers': robot_attack_powers,
        'stun_steps': stun_steps,
        'alliance_groups': alliance_groups,
        'robot_start_positions': robot_start_positions,
    }
    env = BatchRobotVacuumEnv(N, cfg)
    env.reset()
    return env


def make_eval_env(num_robots=2, robot_energies=None,
                  robot_attack_powers=None, alliance_groups=None,
                  robot_start_positions=None, exclusive_charging=False,
                  stun_steps=5, charger_positions=None,
                  energy_cap=None):
    if robot_energies is None:
        robot_energies = [100] * num_robots
    if robot_attack_powers is None:
        robot_attack_powers = [30] * num_robots
    if alliance_groups is None:
        alliance_groups = []
    if robot_start_positions is None:
        robot_start_positions = {0: (0, 0), 1: (0, 1)}
    if charger_positions is None:
        charger_positions = [(3, 3)]
    env = RobotVacuumGymEnv(
        n=6, num_robots=num_robots,
        charger_positions=charger_positions,
        e_move=0, e_charge=10, e_collision=30, e_boundary=0,
        exclusive_charging=exclusive_charging, dust_enabled=False,
        stun_steps=stun_steps,
        alliance_groups=alliance_groups,
        n_steps=200,
        robot_start_positions=robot_start_positions,
        robot_energies=robot_energies,
        robot_attack_powers=robot_attack_powers,
        energy_cap=energy_cap,
    )
    env.reset()
    return env


# Action constants (per CLAUDE description)
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


# Result tracker
results = []
def report(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    results.append((name, ok, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Basic energy averaging (batch_env)
# r0 at (0,0), r1 at (0,1), r0 moves RIGHT into r1.
# r0=100, r1=60 -> both become 80. Knockback pushes r1 to (0,2).
# ─────────────────────────────────────────────────────────────────────────────
def test_1_batch_basic_avg():
    env = make_batch_env(
        N=1, num_robots=2,
        robot_energies=[100, 100],  # caps remain 100
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    # Override energies after reset
    env.energy[0, 0] = 100.0
    env.energy[0, 1] = 60.0
    env.prev_energy[0, 0] = 100.0
    env.prev_energy[0, 1] = 60.0

    actions = np.array([RIGHT], dtype=np.int32)
    env.step_single(0, actions, is_last_turn=False)

    e0 = float(env.energy[0, 0])
    e1 = float(env.energy[0, 1])
    pos0 = tuple(env.pos[0, 0].tolist())
    pos1 = tuple(env.pos[0, 1].tolist())

    ok = (abs(e0 - 80.0) < 1e-5 and abs(e1 - 80.0) < 1e-5
          and pos0 == (0, 1) and pos1 == (0, 2))
    report(
        "1. batch_env basic averaging (100,60 -> 80,80)",
        ok,
        f"r0 e={e0} pos={pos0}; r1 e={e1} pos={pos1}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Basic energy averaging (eval path)
# ─────────────────────────────────────────────────────────────────────────────
def test_2_eval_basic_avg():
    env = make_eval_env(
        num_robots=2,
        robot_energies=[100, 100],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    env.env.robots[0]['energy'] = 100.0
    env.env.robots[1]['energy'] = 60.0
    env.prev_robots[0]['energy'] = 100.0
    env.prev_robots[1]['energy'] = 60.0

    env.step_single(0, RIGHT, is_last_turn=False)

    e0 = env.env.robots[0]['energy']
    e1 = env.env.robots[1]['energy']
    pos0 = (env.env.robots[0]['y'], env.env.robots[0]['x'])
    pos1 = (env.env.robots[1]['y'], env.env.robots[1]['x'])

    ok = (abs(e0 - 80.0) < 1e-5 and abs(e1 - 80.0) < 1e-5
          and pos0 == (0, 1) and pos1 == (0, 2))
    report(
        "2. eval path basic averaging (100,60 -> 80,80)",
        ok,
        f"r0 e={e0} pos={pos0}; r1 e={e1} pos={pos1}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: No double-cap.
# After r0 hits r1: r1 should stay at the average even when r1 takes its own
# step_single (whose end-of-step cap should NOT downgrade r1's already-set value).
# Use init_energies=[100,100] so cap is 100; avg=80 should not be re-clipped.
# Specifically, ensure r1's energy after r1.step_single(STAY) is still 80.
# ─────────────────────────────────────────────────────────────────────────────
def test_3_no_double_cap_batch():
    env = make_batch_env(
        N=1, num_robots=2,
        robot_energies=[100, 100],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    env.energy[0, 0] = 100.0
    env.energy[0, 1] = 60.0
    env.prev_energy[0, 0] = 100.0
    env.prev_energy[0, 1] = 60.0

    env.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)
    e1_after_r0 = float(env.energy[0, 1])
    # Now run r1's step (STAY) — r1's end-of-step cap should not change anything
    env.step_single(1, np.array([STAY], dtype=np.int32), is_last_turn=True)
    e1_after_r1 = float(env.energy[0, 1])

    ok = (abs(e1_after_r0 - 80.0) < 1e-5 and abs(e1_after_r1 - 80.0) < 1e-5)
    report(
        "3. no double-cap: r1 stays at 80 after own step (batch_env)",
        ok,
        f"after r0={e1_after_r0}, after r1={e1_after_r1}",
    )


def test_3_no_double_cap_eval():
    env = make_eval_env(
        num_robots=2,
        robot_energies=[100, 100],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    env.env.robots[0]['energy'] = 100.0
    env.env.robots[1]['energy'] = 60.0
    env.prev_robots[0]['energy'] = 100.0
    env.prev_robots[1]['energy'] = 60.0

    env.step_single(0, RIGHT, is_last_turn=False)
    e1_after_r0 = env.env.robots[1]['energy']
    env.step_single(1, STAY, is_last_turn=True)
    e1_after_r1 = env.env.robots[1]['energy']

    ok = (abs(e1_after_r0 - 80.0) < 1e-5 and abs(e1_after_r1 - 80.0) < 1e-5)
    report(
        "3b. no double-cap: r1 stays at 80 after own step (eval)",
        ok,
        f"after r0={e1_after_r0}, after r1={e1_after_r1}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Asymmetric caps.
# r0 cap=100, r1 cap=40. r0=100, r1=30 -> avg=65. r0 gets 65; r1 capped to 40.
# Verified in both batch_env and eval.
# ─────────────────────────────────────────────────────────────────────────────
def test_4_asymmetric_caps_batch():
    env = make_batch_env(
        N=1, num_robots=2,
        robot_energies=[100, 40],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    env.energy[0, 0] = 100.0
    env.energy[0, 1] = 30.0
    env.prev_energy[0, 0] = 100.0
    env.prev_energy[0, 1] = 30.0

    env.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)

    e0 = float(env.energy[0, 0])
    e1 = float(env.energy[0, 1])
    ok = (abs(e0 - 65.0) < 1e-5 and abs(e1 - 40.0) < 1e-5)
    report(
        "4. asymmetric caps (cap0=100, cap1=40): 100,30 -> 65,40 (batch)",
        ok,
        f"r0={e0}, r1={e1}",
    )


def test_4_asymmetric_caps_eval():
    env = make_eval_env(
        num_robots=2,
        robot_energies=[100, 40],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    env.env.robots[0]['energy'] = 100.0
    env.env.robots[1]['energy'] = 30.0
    env.prev_robots[0]['energy'] = 100.0
    env.prev_robots[1]['energy'] = 30.0

    env.step_single(0, RIGHT, is_last_turn=False)

    e0 = env.env.robots[0]['energy']
    e1 = env.env.robots[1]['energy']
    ok = (abs(e0 - 65.0) < 1e-5 and abs(e1 - 40.0) < 1e-5)
    report(
        "4b. asymmetric caps (eval): 100,30 -> 65,40",
        ok,
        f"r0={e0}, r1={e1}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: No charge sharing.
# r0 sits ON the charger (3,3). r1 sits far away (0,0). After several charging
# turns, r1's energy must NOT increase (was 50 -> stays 50).
# ─────────────────────────────────────────────────────────────────────────────
def test_5_no_charge_sharing_batch():
    env = make_batch_env(
        N=1, num_robots=2,
        robot_energies=[100, 100],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (3, 3), 1: (0, 0)},
        charger_positions=[(3, 3)],
    )
    env.energy[0, 0] = 50.0
    env.energy[0, 1] = 50.0
    env.prev_energy[0, 0] = 50.0
    env.prev_energy[0, 1] = 50.0

    initial_r1 = float(env.energy[0, 1])

    # Run several charging cycles. Each "step" in the env model
    # is rid=0 (STAY, is_last_turn=True for charging), then rid=1 (STAY),
    # then advance_step.
    for _ in range(5):
        env.step_single(0, np.array([STAY], dtype=np.int32), is_last_turn=True)
        env.step_single(1, np.array([STAY], dtype=np.int32), is_last_turn=True)
        env.advance_step()

    e0 = float(env.energy[0, 0])
    e1 = float(env.energy[0, 1])

    # r0 energy should be capped at 100 (was 50, charging would have raised it)
    # r1 energy must remain at 50 (no charge sharing)
    ok = (abs(e1 - initial_r1) < 1e-5)
    report(
        "5. no charge sharing (batch): r1 energy unchanged after r0 charges",
        ok,
        f"initial r1={initial_r1}, final r1={e1}, r0={e0}",
    )


def test_5_no_charge_sharing_eval():
    env = make_eval_env(
        num_robots=2,
        robot_energies=[100, 100],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (3, 3), 1: (0, 0)},
        charger_positions=[(3, 3)],
    )
    env.env.robots[0]['energy'] = 50.0
    env.env.robots[1]['energy'] = 50.0
    env.prev_robots[0]['energy'] = 50.0
    env.prev_robots[1]['energy'] = 50.0

    initial_r1 = env.env.robots[1]['energy']

    for _ in range(5):
        env.step_single(0, STAY, is_last_turn=True)
        env.step_single(1, STAY, is_last_turn=True)
        env.advance_step()

    e0 = env.env.robots[0]['energy']
    e1 = env.env.robots[1]['energy']
    ok = (abs(e1 - initial_r1) < 1e-5)
    report(
        "5b. no charge sharing (eval): r1 energy unchanged after r0 charges",
        ok,
        f"initial r1={initial_r1}, final r1={e1}, r0={e0}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Enemy still takes damage.
# 3 robots: alliance [{0,1}]. Robot 2 is enemy. r0 hits r2 -> r2 takes
# attack_powers[0]=30 damage and stun_counter[2]=stun_steps.
# r0=(0,0), r2=(0,1), r0 moves RIGHT.
# Place r1 somewhere far away so it doesn't interfere.
# ─────────────────────────────────────────────────────────────────────────────
def test_6_enemy_takes_damage_batch():
    env = make_batch_env(
        N=1, num_robots=3,
        robot_energies=[100, 100, 100],
        robot_attack_powers=[30, 30, 30],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 2: (0, 1), 1: (5, 5)},
        stun_steps=5,
    )
    env.energy[0, 0] = 100.0
    env.energy[0, 1] = 100.0
    env.energy[0, 2] = 100.0
    env.prev_energy[0, :] = 100.0

    env.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)

    e2 = float(env.energy[0, 2])
    stun2 = int(env.stun_counter[0, 2])
    # r2 should be at (0,2) due to knockback; r0 at (0,1)
    pos0 = tuple(env.pos[0, 0].tolist())
    pos2 = tuple(env.pos[0, 2].tolist())

    ok = (abs(e2 - 70.0) < 1e-5 and stun2 == 5
          and pos0 == (0, 1) and pos2 == (0, 2))
    report(
        "6. enemy r2 takes damage and stun from allied r0 (batch)",
        ok,
        f"r2 e={e2}, stun={stun2}, pos0={pos0}, pos2={pos2}",
    )


def test_6_enemy_takes_damage_eval():
    env = make_eval_env(
        num_robots=3,
        robot_energies=[100, 100, 100],
        robot_attack_powers=[30, 30, 30],
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 2: (0, 1), 1: (5, 5)},
        stun_steps=5,
    )
    for i in range(3):
        env.env.robots[i]['energy'] = 100.0
        env.prev_robots[i]['energy'] = 100.0

    env.step_single(0, RIGHT, is_last_turn=False)

    e2 = env.env.robots[2]['energy']
    stun2 = env.env.robots[2]['stun_counter']
    pos0 = (env.env.robots[0]['y'], env.env.robots[0]['x'])
    pos2 = (env.env.robots[2]['y'], env.env.robots[2]['x'])

    ok = (abs(e2 - 70.0) < 1e-5 and stun2 == 5
          and pos0 == (0, 1) and pos2 == (0, 2))
    report(
        "6b. enemy r2 takes damage and stun from allied r0 (eval)",
        ok,
        f"r2 e={e2}, stun={stun2}, pos0={pos0}, pos2={pos2}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Ally NOT stunned.
# ─────────────────────────────────────────────────────────────────────────────
def test_7_ally_not_stunned_batch():
    env = make_batch_env(
        N=1, num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
        stun_steps=5,
    )
    env.energy[0, 0] = 100.0
    env.energy[0, 1] = 60.0
    env.prev_energy[0, 0] = 100.0
    env.prev_energy[0, 1] = 60.0

    env.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)

    stun1 = int(env.stun_counter[0, 1])
    just_set1 = bool(env.stun_just_set[0, 1])
    ok = (stun1 == 0 and not just_set1)
    report(
        "7. ally r1 NOT stunned by r0 friendly collision (batch)",
        ok,
        f"stun_counter={stun1}, just_set={just_set1}",
    )


def test_7_ally_not_stunned_eval():
    env = make_eval_env(
        num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
        stun_steps=5,
    )
    env.env.robots[0]['energy'] = 100.0
    env.env.robots[1]['energy'] = 60.0
    env.prev_robots[0]['energy'] = 100.0
    env.prev_robots[1]['energy'] = 60.0

    env.step_single(0, RIGHT, is_last_turn=False)

    stun1 = env.env.robots[1]['stun_counter']
    just_set1 = env.env.robots[1].get('stun_just_set', False)
    ok = (stun1 == 0 and not just_set1)
    report(
        "7b. ally r1 NOT stunned by r0 friendly collision (eval)",
        ok,
        f"stun_counter={stun1}, just_set={just_set1}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Reward signal reflects energy delta.
# r0=100, r1=60 -> after collision r0=80, r1=80.
# r0's reward = (80-100)*0.05 = -1.0
# r1's reward (next step, STAY): pre=60 (saved before r0's step), post=80
# Wait: in eval, prev_robots[1] was set to 60 before r0's step. After r0's
# friendly collision, env.env.robots[1].energy=80. When r1.step_single(STAY)
# runs, it computes reward from prev_robots[1] (=60) vs robots[1] (=80) -> +1.0.
#
# But in batch_env: prev_energy[:,1] is updated only at the END of r1's own
# step. So when r1.step_single(STAY) runs:
#   energy_delta = energy[1] - prev_energy[1] = 80 - 60 = +20 -> reward +1.0.
# ─────────────────────────────────────────────────────────────────────────────
def test_8_reward_batch():
    env = make_batch_env(
        N=1, num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    env.energy[0, 0] = 100.0
    env.energy[0, 1] = 60.0
    env.prev_energy[0, 0] = 100.0
    env.prev_energy[0, 1] = 60.0

    _, r0_rewards, _, _, _ = env.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)
    r0_reward = float(r0_rewards[0])

    _, r1_rewards, _, _, _ = env.step_single(1, np.array([STAY], dtype=np.int32), is_last_turn=False)
    r1_reward = float(r1_rewards[0])

    expected_r0 = (80.0 - 100.0) * 0.05  # -1.0
    expected_r1 = (80.0 - 60.0) * 0.05   # +1.0
    ok = (abs(r0_reward - expected_r0) < 1e-5 and abs(r1_reward - expected_r1) < 1e-5)
    report(
        "8. reward signal matches energy averaging (batch)",
        ok,
        f"r0 reward={r0_reward} (expected {expected_r0}); r1 reward={r1_reward} (expected {expected_r1})",
    )


def test_8_reward_eval():
    env = make_eval_env(
        num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    env.env.robots[0]['energy'] = 100.0
    env.env.robots[1]['energy'] = 60.0
    env.prev_robots[0]['energy'] = 100.0
    env.prev_robots[1]['energy'] = 60.0

    _, r0_reward, _, _, _ = env.step_single(0, RIGHT, is_last_turn=False)
    _, r1_reward, _, _, _ = env.step_single(1, STAY, is_last_turn=False)

    expected_r0 = (80.0 - 100.0) * 0.05
    expected_r1 = (80.0 - 60.0) * 0.05
    ok = (abs(r0_reward - expected_r0) < 1e-5 and abs(r1_reward - expected_r1) < 1e-5)
    report(
        "8b. reward signal matches energy averaging (eval)",
        ok,
        f"r0 reward={r0_reward} (expected {expected_r0}); r1 reward={r1_reward} (expected {expected_r1})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: stationary_blocked case (batch_env).
# r1 at corner against a wall, r0 tries to push but knockback would be off-grid.
# r0=(0,0), r1=(0,1) is fine for knockback. We need r1 against wall such that
# knockback target is off-grid. Use r0=(0,4), r1=(0,5) on n=6 grid;
# r0 RIGHT -> knockback to (0,6) which is off-grid -> stationary_blocked.
# Energy averaging should still apply, r0 stays at (0,4).
# ─────────────────────────────────────────────────────────────────────────────
def test_9_stationary_blocked_batch():
    env = make_batch_env(
        N=1, num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 4), 1: (0, 5)},
    )
    env.energy[0, 0] = 100.0
    env.energy[0, 1] = 60.0
    env.prev_energy[0, 0] = 100.0
    env.prev_energy[0, 1] = 60.0

    env.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)

    e0 = float(env.energy[0, 0])
    e1 = float(env.energy[0, 1])
    pos0 = tuple(env.pos[0, 0].tolist())
    pos1 = tuple(env.pos[0, 1].tolist())

    # r0 should NOT have moved (stationary_blocked)
    ok = (abs(e0 - 80.0) < 1e-5 and abs(e1 - 80.0) < 1e-5
          and pos0 == (0, 4) and pos1 == (0, 5))
    report(
        "9. stationary_blocked: r0 stays put, energy averaging applies (batch)",
        ok,
        f"r0 e={e0} pos={pos0}; r1 e={e1} pos={pos1}",
    )


def test_9_stationary_blocked_eval():
    env = make_eval_env(
        num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 4), 1: (0, 5)},
    )
    env.env.robots[0]['energy'] = 100.0
    env.env.robots[1]['energy'] = 60.0
    env.prev_robots[0]['energy'] = 100.0
    env.prev_robots[1]['energy'] = 60.0

    env.step_single(0, RIGHT, is_last_turn=False)

    e0 = env.env.robots[0]['energy']
    e1 = env.env.robots[1]['energy']
    pos0 = (env.env.robots[0]['y'], env.env.robots[0]['x'])
    pos1 = (env.env.robots[1]['y'], env.env.robots[1]['x'])

    ok = (abs(e0 - 80.0) < 1e-5 and abs(e1 - 80.0) < 1e-5
          and pos0 == (0, 4) and pos1 == (0, 5))
    report(
        "9b. stationary_blocked: r0 stays put, energy averaging applies (eval)",
        ok,
        f"r0 e={e0} pos={pos0}; r1 e={e1} pos={pos1}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Both paths consistent.
# Run an identical sequence through batch_env and through RobotVacuumGymEnv,
# verify final positions and energies match.
# Sequence:
#   r0 RIGHT (friendly hit r1 at (0,1)): r0=80,r1=80; r0->(0,1), r1->(0,2)
#   r1 STAY: positions unchanged; energies unchanged
#   advance_step
#   r0 RIGHT (hit r1 again): avg(80,80)=80; r0->(0,2), r1->(0,3)
# ─────────────────────────────────────────────────────────────────────────────
def test_10_both_paths_consistent():
    # batch
    benv = make_batch_env(
        N=1, num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    benv.energy[0, 0] = 100.0
    benv.energy[0, 1] = 60.0
    benv.prev_energy[0, 0] = 100.0
    benv.prev_energy[0, 1] = 60.0

    benv.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)
    benv.step_single(1, np.array([STAY], dtype=np.int32), is_last_turn=False)
    benv.advance_step()
    benv.step_single(0, np.array([RIGHT], dtype=np.int32), is_last_turn=False)
    benv.step_single(1, np.array([STAY], dtype=np.int32), is_last_turn=False)

    b_e0 = float(benv.energy[0, 0])
    b_e1 = float(benv.energy[0, 1])
    b_pos0 = tuple(benv.pos[0, 0].tolist())
    b_pos1 = tuple(benv.pos[0, 1].tolist())

    # eval
    eenv = make_eval_env(
        num_robots=2,
        alliance_groups=[{0, 1}],
        robot_start_positions={0: (0, 0), 1: (0, 1)},
    )
    eenv.env.robots[0]['energy'] = 100.0
    eenv.env.robots[1]['energy'] = 60.0
    eenv.prev_robots[0]['energy'] = 100.0
    eenv.prev_robots[1]['energy'] = 60.0

    eenv.step_single(0, RIGHT, is_last_turn=False)
    eenv.step_single(1, STAY, is_last_turn=False)
    eenv.advance_step()
    eenv.step_single(0, RIGHT, is_last_turn=False)
    eenv.step_single(1, STAY, is_last_turn=False)

    e_e0 = eenv.env.robots[0]['energy']
    e_e1 = eenv.env.robots[1]['energy']
    e_pos0 = (eenv.env.robots[0]['y'], eenv.env.robots[0]['x'])
    e_pos1 = (eenv.env.robots[1]['y'], eenv.env.robots[1]['x'])

    ok_pos = (b_pos0 == e_pos0 and b_pos1 == e_pos1)
    ok_e = (abs(b_e0 - e_e0) < 1e-5 and abs(b_e1 - e_e1) < 1e-5)
    report(
        "10. batch vs eval path consistency",
        ok_pos and ok_e,
        f"batch: pos0={b_pos0},pos1={b_pos1},e0={b_e0},e1={b_e1} | "
        f"eval: pos0={e_pos0},pos1={e_pos1},e0={e_e0},e1={e_e1}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_1_batch_basic_avg()
    test_2_eval_basic_avg()
    test_3_no_double_cap_batch()
    test_3_no_double_cap_eval()
    test_4_asymmetric_caps_batch()
    test_4_asymmetric_caps_eval()
    test_5_no_charge_sharing_batch()
    test_5_no_charge_sharing_eval()
    test_6_enemy_takes_damage_batch()
    test_6_enemy_takes_damage_eval()
    test_7_ally_not_stunned_batch()
    test_7_ally_not_stunned_eval()
    test_8_reward_batch()
    test_8_reward_eval()
    test_9_stationary_blocked_batch()
    test_9_stationary_blocked_eval()
    test_10_both_paths_consistent()

    print("\n=== Summary ===")
    npass = sum(1 for _, ok, _ in results if ok)
    nfail = sum(1 for _, ok, _ in results if not ok)
    print(f"Passed: {npass} / {len(results)}")
    print(f"Failed: {nfail}")
    if nfail > 0:
        print("\nFailures:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    sys.exit(0)
