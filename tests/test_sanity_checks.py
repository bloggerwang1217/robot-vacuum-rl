"""
Sanity checks for 2v1 alliance experiment (batch_env).
Covers: stun duration, energy averaging, reward integrity,
        exclusive charging, pest mechanic, alliance config.

Usage:
    python tests/test_sanity_checks.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from batch_env import BatchRobotVacuumEnv

# ── Action aliases ──────────────────────────────────────────────────────────
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def _assert(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  {msg}")
    return cond


def make_env(num_envs=1, n=6, R=3, stun_steps=None, init_energies=None,
             alliance_groups=None, exclusive_charging=True,
             e_move=0, e_charge=8, e_decay=0, e_collision=30,
             robot_speeds=None, charger_positions=None, n_steps=500):
    """Helper: build BatchRobotVacuumEnv with fixed corners, no dust."""
    kw = dict(
        n=n, num_robots=R, n_steps=n_steps,
        e_move=e_move, e_charge=e_charge, e_decay=e_decay,
        e_collision=e_collision, e_boundary=0,
        exclusive_charging=exclusive_charging,
        charger_range=1,
        charger_positions=charger_positions or [(n//2, n//2)],
        initial_energy=100,
        robot_energies=init_energies or [100]*R,
        robot_speeds=robot_speeds or [1]*R,
        robot_stun_steps=stun_steps or [0]*R,
        alliance_groups=alliance_groups or [],
        robot_start_positions={},   # will be overridden manually
        random_start_robots=set(),
        dust_enabled=False,
        reward_mode='delta-energy',
        reward_alpha=0.05,
        energy_cap=None,
        docking_steps=0,
    )
    env = BatchRobotVacuumEnv(num_envs=num_envs, env_kwargs=kw)
    env.reset()
    return env


def set_state(env, env_idx=0, positions=None, energies=None, alive=None):
    """Manually set state for env_idx."""
    if positions:
        for rid, (y, x) in enumerate(positions):
            env.pos[env_idx, rid] = [y, x]
    if energies:
        for rid, e in enumerate(energies):
            env.energy[env_idx, rid] = e
            env.prev_energy[env_idx, rid] = e  # keep prev_energy in sync
    if alive is not None:
        for rid, a in enumerate(alive):
            env.alive[env_idx, rid] = a
    env.stun_counter[env_idx] = 0
    env.stun_just_set[env_idx] = False


def act(env, rid, action):
    """Single-env helper: step robot rid with given action."""
    actions = np.full(env.N, action, dtype=np.int32)
    return env.step_single(rid, actions, is_last_turn=True)


def advance(env):
    env.advance_step()


# ═══════════════════════════════════════════════════════════════════════════
# Check 1 — Stun duration asymmetry (r0/r1 stun=5, r2 stun=1)
# ═══════════════════════════════════════════════════════════════════════════
def check1_stun_duration():
    print("\n[Check 1] Stun duration asymmetry")
    ok = True

    # --- Part A: r2 hits r0, r0 stunned for exactly 5 steps ---
    env = make_env(R=3, stun_steps=[5, 5, 1],
                   init_energies=[100, 100, 100],
                   charger_positions=[(5, 5)])  # far away
    set_state(env, positions=[(3, 3), (0, 0), (3, 2)],
              energies=[100, 100, 100])

    # r2 moves RIGHT → hits r0 at (3,3)
    act(env, 2, RIGHT)
    advance(env)
    # After r2's step: stun_just_set[r0]=True, stun_counter[r0]=5, advance clears just_set
    ok &= _assert(env.stun_counter[0, 0] == 5, f"after r2 hits: r0.stun_counter==5 (got {env.stun_counter[0,0]})")
    ok &= _assert(not env.stun_just_set[0, 0], "stun_just_set cleared after advance_step")

    # r0 tries to move UP for 5 steps: should be forced STAY each time
    r0_pos_before = env.pos[0, 0].copy()
    for step in range(1, 6):
        act(env, 0, UP)
        advance(env)
        stayed = np.array_equal(env.pos[0, 0], r0_pos_before)
        remaining = env.stun_counter[0, 0]
        ok &= _assert(stayed, f"  step {step}: r0 stayed (stun={remaining})")

    # Step 6: stun should be 0, r0 can move
    act(env, 0, UP)
    advance(env)
    moved = not np.array_equal(env.pos[0, 0], r0_pos_before)
    ok &= _assert(moved, f"step 6: r0 can move (stun=0)")

    # --- Part B: r0 hits r2, r2 stunned for exactly 1 step ---
    env2 = make_env(R=3, stun_steps=[5, 5, 1],
                    init_energies=[100, 100, 100],
                    charger_positions=[(5, 5)])
    set_state(env2, positions=[(3, 2), (0, 0), (3, 3)],
              energies=[100, 100, 100])

    # r0 moves RIGHT → hits r2 at (3,3)
    act(env2, 0, RIGHT)
    advance(env2)
    ok &= _assert(env2.stun_counter[0, 2] == 1,
                  f"after r0 hits r2: r2.stun_counter==1 (got {env2.stun_counter[0,2]})")

    r2_pos_before = env2.pos[0, 2].copy()
    act(env2, 2, UP)  # r2 tries to move: should be stunned
    advance(env2)
    stayed = np.array_equal(env2.pos[0, 2], r2_pos_before)
    ok &= _assert(stayed, "step 1: r2 stayed due to stun=1")
    ok &= _assert(env2.stun_counter[0, 2] == 0,
                  f"after 1 stun step: r2.stun_counter==0 (got {env2.stun_counter[0,2]})")

    act(env2, 2, UP)  # step 2: r2 should be free to move
    advance(env2)
    moved = not np.array_equal(env2.pos[0, 2], r2_pos_before)
    ok &= _assert(moved, "step 2: r2 can move (stun expired)")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Check 3 — Alliance energy averaging + cap correctness
# ═══════════════════════════════════════════════════════════════════════════
def check3_energy_averaging():
    print("\n[Check 3] Alliance energy averaging + cap")
    ok = True

    env = make_env(R=3, alliance_groups=[[0, 1]],
                   init_energies=[100, 20, 100],  # r1 cap = 20
                   charger_positions=[(5, 5)], e_move=0)
    set_state(env, positions=[(2, 2), (2, 3), (0, 0)],
              energies=[40.0, 10.0, 100.0])

    # r0 moves RIGHT → hits r1 (ally)
    _, rewards, _, _, _ = act(env, 0, RIGHT)

    r0_e = env.energy[0, 0]
    r1_e = env.energy[0, 1]
    avg   = (40.0 + 10.0) / 2.0   # 25.0
    cap1  = 20.0                    # r1's init_energy

    ok &= _assert(abs(r0_e - avg) < 0.01,
                  f"r0 energy == avg=25.0 (got {r0_e:.2f})")
    ok &= _assert(abs(r1_e - min(avg, cap1)) < 0.01,
                  f"r1 energy == min(25, cap=20)=20.0 (got {r1_e:.2f})")
    ok &= _assert(env.stun_counter[0, 1] == 0,
                  f"r1 not stunned after allied collision (got stun={env.stun_counter[0,1]})")

    # Check no energy created / destroyed (accounting for cap)
    total_before = 40.0 + 10.0
    total_after  = r0_e + r1_e
    # Energy may be lost if r1 is capped (r0 gets 25, r1 gets 20 → 45 ≤ 50 is OK)
    ok &= _assert(total_after <= total_before + 0.01,
                  f"no energy created: before={total_before:.1f}, after={total_after:.1f}")

    # Variant: both under cap → average exactly preserved
    env2 = make_env(R=3, alliance_groups=[[0, 1]],
                    init_energies=[100, 100, 100], charger_positions=[(5, 5)], e_move=0)
    set_state(env2, positions=[(2, 2), (2, 3), (0, 0)],
              energies=[6.0, 4.0, 100.0])
    act(env2, 0, RIGHT)
    expected = 5.0
    ok &= _assert(abs(env2.energy[0, 0] - expected) < 0.01 and
                  abs(env2.energy[0, 1] - expected) < 0.01,
                  f"r0=r1=5.0 when both under cap (got r0={env2.energy[0,0]:.2f}, r1={env2.energy[0,1]:.2f})")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Check 4 — Alliance reward integrity after energy averaging
# ═══════════════════════════════════════════════════════════════════════════
def check4_reward_integrity():
    print("\n[Check 4] Reward integrity after energy averaging")
    ok = True

    env = make_env(R=3, alliance_groups=[[0, 1]],
                   init_energies=[100, 20, 100], charger_positions=[(5, 5)], e_move=0)
    set_state(env, positions=[(2, 2), (2, 3), (0, 0)],
              energies=[40.0, 10.0, 100.0])
    # prev_energy for r0 is 40 (set via set_state)

    _, rewards, _, _, _ = act(env, 0, RIGHT)
    r0_reward = rewards[0]
    expected  = (25.0 - 40.0) * 0.05  # -0.75

    ok &= _assert(abs(r0_reward - expected) < 0.001,
                  f"r0 reward == (25-40)*0.05 = -0.75 (got {r0_reward:.4f})")

    # r1's reward comes on its own step_single
    # prev_energy[r1] is still 10 (saved at init / last step), now energy=20
    _, r1_rewards, _, _, _ = act(env, 1, STAY)
    r1_reward = r1_rewards[0]
    # prev_energy[r1] was 10 (from set_state), energy is now 20 → delta=+10 → +0.5
    expected_r1 = (20.0 - 10.0) * 0.05   # +0.50
    ok &= _assert(abs(r1_reward - expected_r1) < 0.001,
                  f"r1 reward == (20-10)*0.05 = +0.50 (got {r1_reward:.4f})")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Check 5 — Exclusive charging enforcement
# ═══════════════════════════════════════════════════════════════════════════
def check5_exclusive_charging():
    print("\n[Check 5] Exclusive charging enforcement")
    ok = True
    cy, cx = 3, 3

    # Both r0 and r1 in charger range → neither charges
    env = make_env(R=3, exclusive_charging=True, e_charge=8,
                   init_energies=[100, 100, 100],
                   charger_positions=[(cy, cx)], e_move=0)
    set_state(env, positions=[(cy, cx), (cy, cx+1), (0, 0)],
              energies=[10.0, 10.0, 10.0])
    act(env, 0, STAY)
    act(env, 1, STAY)
    act(env, 2, STAY)
    ok &= _assert(abs(env.energy[0, 0] - 10.0) < 0.01,
                  f"r0 not charged when r1 also in range (got {env.energy[0,0]:.2f})")
    ok &= _assert(abs(env.energy[0, 1] - 10.0) < 0.01,
                  f"r1 not charged when r0 also in range (got {env.energy[0,1]:.2f})")

    # Only r0 in range → r0 charges
    env2 = make_env(R=3, exclusive_charging=True, e_charge=8,
                    init_energies=[100, 100, 100],
                    charger_positions=[(cy, cx)], e_move=0)
    set_state(env2, positions=[(cy, cx), (0, 0), (0, 5)],
              energies=[10.0, 10.0, 10.0])
    act(env2, 0, STAY)
    ok &= _assert(abs(env2.energy[0, 0] - 18.0) < 0.01,
                  f"r0 charges when sole occupant (got {env2.energy[0,0]:.2f})")

    # All three in range → nobody charges
    env3 = make_env(R=3, exclusive_charging=True, e_charge=8,
                    init_energies=[100, 100, 100],
                    charger_positions=[(cy, cx)], e_move=0)
    set_state(env3, positions=[(cy, cx), (cy+1, cx), (cy, cx+1)],
              energies=[10.0, 10.0, 10.0])
    act(env3, 0, STAY); act(env3, 1, STAY); act(env3, 2, STAY)
    ok &= _assert(all(abs(env3.energy[0, r] - 10.0) < 0.01 for r in range(3)),
                  f"nobody charges when all 3 in range (got {env3.energy[0].tolist()})")

    # r1 dead → r0 alone in range → r0 charges
    env4 = make_env(R=3, exclusive_charging=True, e_charge=8,
                    init_energies=[100, 100, 100],
                    charger_positions=[(cy, cx)], e_move=0)
    set_state(env4, positions=[(cy, cx), (cy, cx+1), (0, 0)],
              energies=[10.0, 10.0, 10.0], alive=[True, False, True])
    act(env4, 0, STAY)
    ok &= _assert(abs(env4.energy[0, 0] - 18.0) < 0.01,
                  f"r0 charges when r1 is dead (got {env4.energy[0,0]:.2f})")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Check 7 — Pest mechanic: scripted seek_charger r2 damages r0
# ═══════════════════════════════════════════════════════════════════════════
def check7_pest_mechanic():
    print("\n[Check 7] Pest mechanic (seek_charger r2 actually hits r0)")
    ok = True

    n = 6
    cy, cx = 3, 3

    # Simple seek_charger: move toward charger using actual env.pos
    def seek_charger_action(env, rid, charger_yx):
        cy, cx = charger_yx
        pos = env.pos[0, rid]  # [y, x]
        dy = cy - pos[0]
        dx = cx - pos[1]
        if dy == 0 and dx == 0:
            return STAY
        if abs(dy) >= abs(dx):
            return DOWN if dy > 0 else UP
        else:
            return RIGHT if dx > 0 else LEFT

    env = make_env(R=3, stun_steps=[5, 5, 1],
                   init_energies=[100, 100, 100],
                   alliance_groups=[[0, 1]],
                   exclusive_charging=True, e_charge=8, e_move=0,
                   charger_positions=[(cy, cx)], n_steps=200)
    # r0 on charger, r1 far, r2 starts at (3, 0) — must walk right to reach charger
    set_state(env, positions=[(cy, cx), (0, 0), (cy, 0)],
              energies=[100.0, 100.0, 100.0])

    total_hits = 0
    for _ in range(100):
        # r0: STAY on charger
        act(env, 0, STAY)
        # r1: STAY far
        act(env, 1, STAY)
        # r2: seek charger
        a2 = seek_charger_action(env, 2, (cy, cx))
        act(env, 2, a2)
        advance(env)
        hits = int(env.active_collisions_with[0, 2, 0])  # r2→r0
        total_hits += hits

    ok &= _assert(total_hits > 0,
                  f"r2 hit r0 at least once over 100 steps (got total_hits={total_hits})")

    # Verify hit causes damage (e_collision=30 default)
    env2 = make_env(R=3, stun_steps=[5, 5, 1],
                    init_energies=[100, 100, 100],
                    e_collision=30, e_move=0,
                    charger_positions=[(cy, cx)])
    set_state(env2, positions=[(cy, cx), (0, 0), (cy, cx-1)],
              energies=[100.0, 100.0, 100.0])
    act(env2, 2, RIGHT)   # r2 moves RIGHT → hits r0 at (cy, cx)
    ok &= _assert(env2.energy[0, 0] < 100.0,
                  f"r0 loses energy after r2 hits (got {env2.energy[0,0]:.1f})")
    ok &= _assert(abs(env2.energy[0, 0] - 70.0) < 0.01,
                  f"r0 energy = 100-30 = 70 after hit (got {env2.energy[0,0]:.2f})")
    ok &= _assert(env2.stun_counter[0, 0] == 5,
                  f"r0.stun_counter==5 after r2 hit (got {env2.stun_counter[0,0]})")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Check 8 — Alliance group config round-trip
# ═══════════════════════════════════════════════════════════════════════════
def check8_alliance_config():
    print("\n[Check 8] Alliance group configuration round-trip")
    ok = True

    env = make_env(R=3, alliance_groups=[[0, 1]], stun_steps=[5, 5, 1])

    # Should be exactly {(0,1)} — r2 not included
    ok &= _assert(env._allied_pairs == {(0, 1)},
                  f"_allied_pairs == {{(0,1)}} (got {env._allied_pairs})")

    # r0 hitting r2 (enemy) should deal damage and stun
    set_state(env, positions=[(2, 2), (0, 0), (2, 3)],
              energies=[100.0, 100.0, 100.0])
    act(env, 0, RIGHT)   # r0 → hits r2
    ok &= _assert(env.energy[0, 2] < 100.0,
                  f"r2 takes damage from r0 (got {env.energy[0,2]:.1f})")
    ok &= _assert(env.stun_counter[0, 2] > 0,
                  f"r2 is stunned after r0 hit (got stun={env.stun_counter[0,2]})")

    # r0 hitting r1 (ally) should NOT deal damage, no stun
    env2 = make_env(R=3, alliance_groups=[[0, 1]], e_move=0,
                    init_energies=[100, 100, 100])
    set_state(env2, positions=[(2, 2), (2, 3), (0, 0)],
              energies=[60.0, 60.0, 100.0])
    act(env2, 0, RIGHT)
    ok &= _assert(env2.stun_counter[0, 1] == 0,
                  f"r1 NOT stunned after allied collision (got stun={env2.stun_counter[0,1]})")
    # Energy should be averaged, not damaged
    ok &= _assert(abs(env2.energy[0, 0] - 60.0) < 0.01 and
                  abs(env2.energy[0, 1] - 60.0) < 0.01,
                  f"equal energy stays equal after ally collision (got {env2.energy[0,0]:.1f}, {env2.energy[0,1]:.1f})")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    results = {}
    results['Check 1: Stun duration']         = check1_stun_duration()
    results['Check 3: Energy averaging']       = check3_energy_averaging()
    results['Check 4: Reward integrity']       = check4_reward_integrity()
    results['Check 5: Exclusive charging']     = check5_exclusive_charging()
    results['Check 7: Pest mechanic']          = check7_pest_mechanic()
    results['Check 8: Alliance config']        = check8_alliance_config()

    print("\n" + "="*55)
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        all_pass = all_pass and passed
    print("="*55)
    sys.exit(0 if all_pass else 1)
