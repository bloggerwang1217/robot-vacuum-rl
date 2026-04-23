"""Sanity check for --alliance-zone feature.

Tests:
  1. Spawn test (BatchRobotVacuumEnv): allied robots start within Chebyshev <= 1.
  2. Mask test: get_valid_action_mask correctly flags illegal moves.
  3. Step enforcement test: step_single forces STAY on illegal actions.
  4. No-zone baseline: mask is all-True when alliance_zone disabled.
  5. Spawn test (RobotVacuumGymEnv via RobotVacuumEnv config path).

All tests use unit-level assertions; no training is performed.
"""
import os
import sys
import random
import traceback
import numpy as np

# Make repo root importable when this file is run directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from batch_env import BatchRobotVacuumEnv
from gym import RobotVacuumGymEnv  # noqa: F401 (used in test 5)
from robot_vacuum_env import RobotVacuumEnv


# ─────────────────────────────────────────────────────────────────────
# Shared env_kwargs
# ─────────────────────────────────────────────────────────────────────

BASE_KW = dict(
    n=5,
    num_robots=3,
    n_steps=500,                 # required by BatchRobotVacuumEnv
    robot_energies=[40, 40, 150],
    robot_speeds=[2, 2, 1],
    charger_positions=[(2, 2)],
    exclusive_charging=True,
    e_move=1, e_charge=8, e_collision=30, e_boundary=0, e_decay=1.5,
    alliance_groups=[{0, 1}],
    alliance_zone=True,
    random_start_robots={0, 1, 2},
    dust_enabled=False,          # simplify
)

# Action constants (same as batch_env._UP/_DOWN/_LEFT/_RIGHT/_STAY)
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
DY = np.array([-1, 1, 0, 0, 0], dtype=np.int32)
DX = np.array([0, 0, -1, 1, 0], dtype=np.int32)


def chebyshev(p0, p1):
    return max(abs(int(p0[0]) - int(p1[0])), abs(int(p0[1]) - int(p1[1])))


# ─────────────────────────────────────────────────────────────────────
# Test 1: spawn test (BatchRobotVacuumEnv)
# ─────────────────────────────────────────────────────────────────────

def test_1_spawn_batch_env(N=100):
    print("\n=== Test 1: Batch spawn within Chebyshev <= 1 ===")
    random.seed(0); np.random.seed(0)
    env = BatchRobotVacuumEnv(num_envs=N, env_kwargs=BASE_KW)
    env.reset()  # triggers _reset_single per env

    pos = env.pos  # (N, R, 2)
    violations = []
    max_d = 0
    for i in range(N):
        d = chebyshev(pos[i, 0], pos[i, 1])
        max_d = max(max_d, d)
        if d > 1:
            violations.append((i, tuple(pos[i, 0]), tuple(pos[i, 1]), d))

    print(f"  envs checked: {N}")
    print(f"  max Chebyshev dist observed between r0 and r1: {max_d}")
    print(f"  violations (d>1): {len(violations)}")
    if violations:
        for v in violations[:5]:
            print(f"    env_idx={v[0]}  r0={v[1]}  r1={v[2]}  d={v[3]}")
    ok = len(violations) == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Test 2: action-mask correctness
# ─────────────────────────────────────────────────────────────────────

def test_2_action_mask():
    print("\n=== Test 2: get_valid_action_mask correctness ===")
    random.seed(1); np.random.seed(1)
    N = 1
    env = BatchRobotVacuumEnv(num_envs=N, env_kwargs=BASE_KW)
    env.reset()

    # Force a known configuration for env 0:
    #   r0 at (2,2), r1 at (2,3), r2 somewhere irrelevant.
    env.pos[0, 0] = (2, 2)
    env.pos[0, 1] = (2, 3)
    env.pos[0, 2] = (0, 0)
    env.alive[0, :] = True

    # From (2,2), Chebyshev distances to r1 at (2,3) after each move:
    #   UP   -> (1,2): d = max(|1-2|,|2-3|) = 1  -> legal
    #   DOWN -> (3,2): d = max(|3-2|,|2-3|) = 1  -> legal
    #   LEFT -> (2,1): d = max(|2-2|,|1-3|) = 2  -> ILLEGAL
    #   RIGHT-> (2,3): d = max(|2-2|,|3-3|) = 0  -> legal (overlap allowed by mask; collision handled later)
    #   STAY -> (2,2): d = 1                      -> legal
    expected = {UP: True, DOWN: True, LEFT: False, RIGHT: True, STAY: True}

    mask = env.get_valid_action_mask(0)  # (N, 5)
    print(f"  mask for robot 0 at (2,2), ally r1 at (2,3): {mask[0]}")
    mismatches = []
    for a, exp in expected.items():
        got = bool(mask[0, a])
        name = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][a]
        print(f"    {name:5s}: expected={exp}  got={got}  {'OK' if got == exp else 'MISMATCH'}")
        if got != exp:
            mismatches.append((name, exp, got))

    # Also check STAY is always True regardless of geometry.
    assert bool(mask[0, STAY]) is True, "STAY must always be legal"

    # Check robot with no allies (r2) has all-True mask.
    mask2 = env.get_valid_action_mask(2)
    all_true = bool(mask2[0].all())
    print(f"  robot 2 (no ally) mask all-True: {all_true}")

    ok = (len(mismatches) == 0) and all_true
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Test 3: step_single enforcement (backup)
# ─────────────────────────────────────────────────────────────────────

def test_3_step_enforcement():
    print("\n=== Test 3: step_single backup enforcement (illegal -> STAY) ===")
    random.seed(2); np.random.seed(2)
    N = 1
    env = BatchRobotVacuumEnv(num_envs=N, env_kwargs=BASE_KW)
    env.reset()
    env.pos[0, 0] = (2, 2)
    env.pos[0, 1] = (2, 3)
    env.pos[0, 2] = (0, 0)
    env.alive[0, :] = True
    env.energy[0, :] = [40, 40, 150]
    env.prev_energy[0, :] = env.energy[0, :].copy()

    before = env.pos[0, 0].copy()
    print(f"  before step: r0={tuple(before)}  r1={tuple(env.pos[0, 1])}")

    # Attempt the illegal LEFT action (would move r0 to (2,1), dist to r1=(2,3) is 2).
    actions = np.array([LEFT], dtype=np.int32)
    env.step_single(0, actions, is_last_turn=True)

    after = env.pos[0, 0].copy()
    print(f"  after  step: r0={tuple(after)}  r1={tuple(env.pos[0, 1])}")

    moved = not np.array_equal(before, after)
    print(f"  robot moved? {moved}  (expected: False)")

    # Sanity: attempt a legal UP action from the same state.
    env.pos[0, 0] = (2, 2)
    env.pos[0, 1] = (2, 3)
    env.alive[0, :] = True
    env.energy[0, :] = [40, 40, 150]
    env.prev_energy[0, :] = env.energy[0, :].copy()
    actions = np.array([UP], dtype=np.int32)
    env.step_single(0, actions, is_last_turn=True)
    moved_legal = not np.array_equal(np.array([2, 2]), env.pos[0, 0])
    print(f"  legal UP applied after reset: r0={tuple(env.pos[0, 0])}  moved? {moved_legal} (expected True)")

    ok = (not moved) and moved_legal
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Test 4: no-zone baseline
# ─────────────────────────────────────────────────────────────────────

def test_4_no_zone_baseline():
    print("\n=== Test 4: alliance_zone=False baseline ===")
    random.seed(3); np.random.seed(3)
    kw = dict(BASE_KW)
    kw["alliance_zone"] = False
    kw["alliance_groups"] = None  # no alliance at all

    N = 50
    env = BatchRobotVacuumEnv(num_envs=N, env_kwargs=kw)
    env.reset()

    # Mask is all-True
    mask = env.get_valid_action_mask(0)
    all_true = bool(mask.all())
    print(f"  mask.all() == True: {all_true}  (shape={mask.shape})")

    # Spawns not constrained to Chebyshev <= 1 — expect at least some envs with d > 1
    dists = [chebyshev(env.pos[i, 0], env.pos[i, 1]) for i in range(N)]
    max_d = max(dists)
    n_far = sum(1 for d in dists if d > 1)
    print(f"  max Cheb(r0,r1) across {N} envs: {max_d}")
    print(f"  envs with Cheb(r0,r1) > 1:        {n_far}/{N}")

    # In a 5x5 board with 3 random robots, max possible Cheb = 4. With 50 trials
    # and no constraint, it is overwhelmingly likely to see some d > 1.
    ok = all_true and (n_far > 0) and (max_d > 1)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Test 5: spawn test for RobotVacuumEnv / RobotVacuumGymEnv
# ─────────────────────────────────────────────────────────────────────

def test_5_robot_vacuum_env_spawn(n_resets=50):
    print("\n=== Test 5: RobotVacuumEnv spawn Chebyshev <= 1 ===")
    random.seed(4); np.random.seed(4)

    # RobotVacuumGymEnv's kwargs path does NOT forward alliance_zone. Use a
    # config dict (matching how the real pipeline builds its config) so the
    # underlying RobotVacuumEnv actually sees the flag.
    config = dict(BASE_KW)
    # Map gym-side keys to the RobotVacuumEnv-side key names where they
    # differ; for the fields used here they already match.
    config.setdefault("initial_energy", max(config["robot_energies"]))
    config["n_steps"] = 500
    config["epsilon"] = 0.2

    gym_env = RobotVacuumGymEnv(config=config)

    # Assert the inner env actually received the flag (plumbing sanity check).
    assert gym_env.env._alliance_zone is True, \
        "inner RobotVacuumEnv did not receive alliance_zone=True via config dict"

    violations = []
    max_d = 0
    for k in range(n_resets):
        gym_env.reset()
        r0 = gym_env.env.robots[0]
        r1 = gym_env.env.robots[1]
        d = chebyshev((r0["y"], r0["x"]), (r1["y"], r1["x"]))
        max_d = max(max_d, d)
        if d > 1:
            violations.append((k, (r0["y"], r0["x"]), (r1["y"], r1["x"]), d))

    print(f"  resets: {n_resets}")
    print(f"  max Chebyshev dist observed: {max_d}")
    print(f"  violations (d>1): {len(violations)}")
    for v in violations[:5]:
        print(f"    reset={v[0]}  r0={v[1]}  r1={v[2]}  d={v[3]}")
    ok = len(violations) == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Bonus: verify gym.py kwargs path silently drops alliance_zone
# ─────────────────────────────────────────────────────────────────────

def test_6_gym_kwargs_plumbing_regression():
    print("\n=== Test 6: RobotVacuumGymEnv(**kwargs) alliance_zone plumbing ===")
    # Instantiate via kwargs; observe whether the inner env sees alliance_zone.
    # NOTE: this is intentionally a diagnostic test — PASS means the kwarg
    # actually propagates. If it does NOT, we report FAIL so the plumbing gap
    # is visible rather than silent.
    env = RobotVacuumGymEnv(
        n=5, num_robots=3,
        robot_energies=[40, 40, 150],
        charger_positions=[(2, 2)],
        alliance_groups=[{0, 1}],
        alliance_zone=True,
        random_start_robots={0, 1, 2},
    )
    inner_has_flag = getattr(env.env, "_alliance_zone", False)
    print(f"  inner env._alliance_zone after kwargs init: {inner_has_flag}")
    ok = bool(inner_has_flag)
    print(f"  RESULT: {'PASS' if ok else 'FAIL  (alliance_zone silently dropped by gym.py kwargs path)'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────

def main():
    tests = [
        ("1. Batch spawn within Cheb<=1", test_1_spawn_batch_env),
        ("2. Action-mask correctness",    test_2_action_mask),
        ("3. step_single enforcement",    test_3_step_enforcement),
        ("4. No-zone baseline",           test_4_no_zone_baseline),
        ("5. RobotVacuumEnv spawn",       test_5_robot_vacuum_env_spawn),
        ("6. gym.py kwargs plumbing",     test_6_gym_kwargs_plumbing_regression),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
