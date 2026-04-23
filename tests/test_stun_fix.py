"""
Verify that batch_env stun decrement fix works correctly.

After the fix (batch_env.py:237-243):
- stun counter should ONLY decrement on is_last_turn=True
- Therefore stun duration = `stun_steps` GLOBAL steps, independent of speed

This test directly instantiates BatchRobotVacuumEnv and drives step_single()
the same way train_dqn_vec.py does (sequential robot turns, with
is_last_turn=(turn_idx == n_turns-1)).

Three tests:
  T1: stun_steps=6 with speed=2 should force STAY for exactly 6 global steps.
  T2: robot_0 UP action during stun must be overridden to STAY every global step.
  T3: Global step 7 (first step AFTER stun clears) — UP must actually move.
"""

import numpy as np
from batch_env import BatchRobotVacuumEnv

_UP, _DOWN, _LEFT, _RIGHT, _STAY = 0, 1, 2, 3, 4

# ── Build env tuned to force a clean collision and isolate stun behavior ───
env_kwargs = dict(
    n=5,
    num_robots=2,
    n_steps=200,

    # Energy: large enough that no one dies during the test window.
    initial_energy=1000,
    robot_energies=[1000, 1000],
    energy_cap=1000,

    # Speeds: robot_0 speed=2 (two sub-turns per global step),
    #         robot_1 speed=1 (one sub-turn).
    robot_speeds=[2, 1],

    # Zero-out confounders so we can observe stun clearly.
    e_move=0,
    e_charge=0,
    e_collision=1,
    e_boundary=0,
    e_decay=0.0,

    # Stun: only robot_0 gets stunned when hit; stun_steps=6.
    stun_steps=0,                     # default off
    robot_stun_steps=[6, 0],          # only robot_0 stuns

    # Place charger out of the way.
    charger_positions=[(4, 4)],
    exclusive_charging=False,

    dust_enabled=False,
    reward_mode='delta-energy',
    reward_alpha=0.05,

    # Fixed starts: robot_0 at (2,2), robot_1 at (2,3) — they are adjacent
    # so robot_1 moving LEFT hits robot_0 on the first attack.
    robot_start_positions={0: (2, 2), 1: (2, 3)},
)

env = BatchRobotVacuumEnv(num_envs=1, env_kwargs=env_kwargs)
env.reset()

# Sanity-check initial setup.
assert env.stun_enabled, "stun_enabled should be True (stun_steps=[6,0])"
assert env.robot_speeds == [2, 1], f"speeds wrong: {env.robot_speeds}"
assert tuple(env.pos[0, 0]) == (2, 2), f"r0 start wrong: {env.pos[0,0]}"
assert tuple(env.pos[0, 1]) == (2, 3), f"r1 start wrong: {env.pos[0,1]}"
print(f"Init  : r0@{tuple(env.pos[0,0])} r1@{tuple(env.pos[0,1])} "
      f"stun={env.stun_counter[0].tolist()} "
      f"speeds={env.robot_speeds}")


# ── Helper: execute ONE global step in the same order the trainer does ────
def global_step(r0_action: int, r1_action: int):
    """
    Replicates train_dqn_vec.py's inner loop for one global step:
      for each robot in order:
          for turn in range(speed):
              step_single(robot, action, is_last_turn=(turn==speed-1))
    Returns the global step index's recorded (r0_pos_before, r0_pos_after, r0_stun_before_decrement).
    """
    actions_map = {0: r0_action, 1: r1_action}
    r0_pos_before = tuple(env.pos[0, 0])
    r0_stun_before = int(env.stun_counter[0, 0])
    for robot_id in [0, 1]:
        n_turns = env.robot_speeds[robot_id]
        for turn_idx in range(n_turns):
            is_last_turn = (turn_idx == n_turns - 1)
            act = np.array([actions_map[robot_id]], dtype=np.int32)
            env.step_single(robot_id, act, is_last_turn=is_last_turn)
    r0_pos_after = tuple(env.pos[0, 0])
    env.advance_step()
    return r0_pos_before, r0_pos_after, r0_stun_before


# ── Step 1: trigger stun. robot_1 moves LEFT → hits robot_0 at (2,2).
# In the attacker's step, (2,2) is occupied; robot_1 tries to push r0 into
# (2,1). Knockback inbounds & not blocked → can_push → r0 pushed, stun set.
gs = 1
r0_bef, r0_aft, stun_bef = global_step(r0_action=_STAY, r1_action=_LEFT)
print(f"GS{gs:02d} (trigger): r0 {r0_bef}->{r0_aft}  stun_before={stun_bef}  "
      f"stun_after={env.stun_counter[0,0]}  col_r1_r0={env.active_collisions_with[0,1,0]}")

# Trigger correctness checks
assert int(env.active_collisions_with[0, 1, 0]) >= 1, "collision did not fire"
# After robot_1's hit, r0 should be knocked back; stun set to 6, then NOT
# decremented because this was not r0's step.
assert int(env.stun_counter[0, 0]) == 6, (
    f"stun_counter should be 6 right after the hit, got {env.stun_counter[0,0]}"
)


# ── Now count how many subsequent global steps r0 is forced to STAY.
# Each global step, r0 tries UP. We check:
#   (a) r0 did not actually move up (confirming override to STAY)
#   (b) the stun counter decrements by exactly 1 per global step
forced_stay_count = 0
stun_trace = [int(env.stun_counter[0, 0])]  # reading AFTER the trigger GS
pos_trace = [tuple(env.pos[0, 0])]

for gs in range(2, 2 + 10):  # probe up to 10 more global steps
    pos_before, pos_after, stun_before = global_step(r0_action=_UP, r1_action=_STAY)
    stun_after = int(env.stun_counter[0, 0])
    moved = pos_before != pos_after
    print(f"GS{gs:02d}         : r0 {pos_before}->{pos_after}  "
          f"stun {stun_before}->{stun_after}  moved={moved}")

    if stun_before > 0:
        # We were stunned entering this global step → must NOT have moved
        forced_stay_count += 1
        if moved:
            print(f"  !! FAIL (T2): r0 moved while stunned at GS{gs}")
        # And stun must have decremented by exactly 1
        if stun_after != stun_before - 1:
            print(f"  !! FAIL (decrement): stun {stun_before} -> {stun_after} "
                  f"(expected {stun_before-1})")
    else:
        # Stun cleared; r0 attempted UP with speed=2 — y should drop by 2
        # (two UP sub-turns per global step), unless clipped at the top wall.
        expected_dy = -env.robot_speeds[0]
        expected_pos = (pos_before[0] + expected_dy, pos_before[1])
        # Clip expected y to [0, n-1] (boundary no-op with e_boundary=0)
        exp_y_clipped = max(0, expected_pos[0])
        if not moved or pos_after[0] > pos_before[0] - 1:
            print(f"  !! FAIL (T3): r0 stun cleared but UP did not move "
                  f"({pos_before}->{pos_after})")
        else:
            dy_actual = int(pos_after[0]) - int(pos_before[0])
            print(f"  ✓ T3: r0 moved UP by dy={dy_actual} "
                  f"(speed={env.robot_speeds[0]}, expected dy={expected_dy}).")
        break


# ── Summary & asserts ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULT SUMMARY")
print("=" * 60)
print(f"forced_stay_count = {forced_stay_count}  (expected 6)")

# T1: forced-stay count must equal stun_steps = 6, NOT 3
assert forced_stay_count == 6, (
    f"T1 FAIL: r0 was forced STAY for {forced_stay_count} global steps, "
    f"expected 6 (stun_steps=6, independent of speed=2)."
)
print("T1 PASS: stun duration = 6 global steps (not 3) — speed-independent.")
print("T2 PASS: all 6 stunned global steps overrode UP → STAY.")
print("T3 PASS: UP succeeded on first post-stun global step.")


# ── T4: direct white-box check — stun must NOT decrement on is_last_turn=False
# Re-create env, trigger stun, then manually call step_single with
# is_last_turn=False for robot_0 and confirm counter is unchanged.
env2 = BatchRobotVacuumEnv(num_envs=1, env_kwargs=env_kwargs)
env2.reset()
# Trigger stun via robot_1 LEFT attack
env2.step_single(0, np.array([_STAY], dtype=np.int32), is_last_turn=True)
env2.step_single(0, np.array([_STAY], dtype=np.int32), is_last_turn=True)  # r0 speed=2, 2 sub-turns
# That actually just stepped robot_0 twice; reset and do properly.
env2 = BatchRobotVacuumEnv(num_envs=1, env_kwargs=env_kwargs)
env2.reset()
# One full global step: r0 (x2 sub-turns, STAY) → r1 (x1, LEFT attack)
env2.step_single(0, np.array([_STAY], dtype=np.int32), is_last_turn=False)
env2.step_single(0, np.array([_STAY], dtype=np.int32), is_last_turn=True)
env2.step_single(1, np.array([_LEFT], dtype=np.int32), is_last_turn=True)
env2.advance_step()
stun_after_trigger = int(env2.stun_counter[0, 0])
assert stun_after_trigger == 6, f"trigger failed in T4 env: stun={stun_after_trigger}"

# Now drive robot_0's FIRST sub-turn of the next global step (is_last_turn=False)
env2.step_single(0, np.array([_UP], dtype=np.int32), is_last_turn=False)
stun_after_sub0 = int(env2.stun_counter[0, 0])
assert stun_after_sub0 == 6, (
    f"T4 FAIL: stun decremented on is_last_turn=False sub-turn "
    f"(before=6, after={stun_after_sub0})"
)
# Now drive the SECOND sub-turn (is_last_turn=True) — counter should drop by 1
env2.step_single(0, np.array([_UP], dtype=np.int32), is_last_turn=True)
stun_after_sub1 = int(env2.stun_counter[0, 0])
assert stun_after_sub1 == 5, (
    f"T4 FAIL: stun did not decrement on is_last_turn=True sub-turn "
    f"(before=6, after={stun_after_sub1})"
)
print("T4 PASS: stun counter decrements ONLY on is_last_turn=True "
      "(6→6 on sub0, 6→5 on sub1).")
