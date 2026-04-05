#!/usr/bin/env python3
"""
精確模擬兩個策略的 reward 比較。
所有機制完全對齊 batch_env.py 的 step_single()。

環境參數 (pest_curriculum_v4):
  Grid: 5x5, charger (2,2), charger_range=0
  r0: E=100, speed=2, atk=30, docking=2, stun=2
  r1: E=100, speed=1, atk=2, docking=0, stun=0
  e_move=0, e_charge=8, e_decay=0.5, exclusive_charging=True
  max_steps=500, energy_cap=init_energy (100)
  reward = ΔE × 0.05 - 100 × died

碰撞規則 (sequential):
  移動者 = 攻擊者，被撞者 = 受害者
  受害者承受攻擊者的 atk damage
  受害者往攻擊者移動方向被擊退 1 格
  受害者 docking_counter 歸零
  受害者 stun_counter 設為 stun_steps[victim]
  攻擊者移動到受害者原本的格子

Stun 機制:
  在 step_single 開頭：若 stun_counter > 0，強制 STAY，counter -= 1
  （speed=2 時每個 sub-step 都會 decrement，所以 stun=2 只持續 1 個 game step）

Docking 機制:
  每個 is_last_turn sub-step：
    若在充電座上 且 not stunned → docking_counter += 1
    否則 → docking_counter = 0
  充電條件：docking_counter >= docking_steps[rid]
"""

import sys

# ── Parameters ─────────────────────────────────────────────────────────
GRID = 5
CHARGER = (2, 2)
E_INIT = [100.0, 100.0]
SPEED = [2, 1]
ATK = [30.0, 2.0]
DOCK = [2, 0]
STUN = [4, 0]
E_MOVE = 0.0
E_CHARGE = 8.0
E_DECAY = 0.5
E_CAP = [100.0, 100.0]
MAX_STEPS = 500
REWARD_ALPHA = 0.05

# Direction deltas: UP, DOWN, LEFT, RIGHT, STAY
DY = [-1, 1, 0, 0, 0]
DX = [0, 0, -1, 1, 0]
ACT_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


class Robot:
    def __init__(self, rid, y, x):
        self.rid = rid
        self.y, self.x = y, x
        self.energy = E_INIT[rid]
        self.alive = True
        self.stun_counter = 0
        self.docking_counter = 0
        self.prev_energy = E_INIT[rid]
        self.total_reward = 0.0


def move_toward(fy, fx, ty, tx):
    """Return action to move from (fy,fx) toward (ty,tx). Manhattan greedy."""
    dy, dx = ty - fy, tx - fx
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1 if dy > 0 else 4  # UP or DOWN
    else:
        return 2 if dx < 0 else 3 if dx > 0 else 4  # LEFT or RIGHT


def on_charger(y, x):
    return y == CHARGER[0] and x == CHARGER[1]


def in_bounds(y, x):
    return 0 <= y < GRID and 0 <= x < GRID


def do_substep(mover, other, action, is_last_turn, step_log=None):
    """Execute one sub-step for mover. Returns reward for this sub-step."""
    rid = mover.rid
    if not mover.alive:
        return 0.0

    # ── Stun check ──
    original_action = action
    if mover.stun_counter > 0:
        action = 4  # STAY
        mover.stun_counter -= 1

    # ── Decay (only last turn) ──
    if E_DECAY > 0 and is_last_turn:
        mover.energy -= E_DECAY
        mover.energy = max(mover.energy, 0.0)
        if mover.energy <= 0:
            mover.alive = False

    if not mover.alive:
        reward = (mover.energy - mover.prev_energy) * REWARD_ALPHA - 100.0
        mover.prev_energy = mover.energy
        mover.total_reward += reward
        return reward

    is_move = action != 4
    dy, dx = DY[action], DX[action]
    py, px = mover.y + dy, mover.x + dx

    # ── Move cost ──
    if is_move:
        mover.energy -= E_MOVE

    # ── Boundary ──
    is_boundary = is_move and not in_bounds(py, px)
    if is_boundary:
        is_move = False  # can't move

    # ── Collision ──
    collision = False
    if is_move and other.alive and py == other.y and px == other.x:
        collision = True
        # Knockback
        kby, kbx = other.y + dy, other.x + dx
        if in_bounds(kby, kbx):
            # Push: mover takes victim's spot, victim pushed
            mover.y, mover.x = py, px
            other.y, other.x = kby, kbx
        # else: stationary hit, mover stays
        # Victim takes damage
        other.energy -= ATK[rid]
        other.docking_counter = 0
        other.stun_counter = STUN[other.rid]

        # Check victim death
        other.energy = max(other.energy, 0.0)
        if other.energy <= 0:
            other.alive = False

        if step_log is not None:
            step_log.append(
                f"    r{rid} hits r{other.rid}: {ACT_NAMES[action]}, "
                f"r{other.rid} takes {ATK[rid]} dmg → E={other.energy:.1f}"
                f"{' DEAD!' if not other.alive else ''}, "
                f"knockback→({other.y},{other.x})"
            )
    elif is_move and not collision:
        mover.y, mover.x = py, px

    # ── Docking (last turn only) ──
    if is_last_turn:
        if on_charger(mover.y, mover.x) and mover.alive and mover.stun_counter == 0:
            mover.docking_counter += 1
        else:
            mover.docking_counter = 0

    # ── Charging (last turn only) ──
    charged = 0.0
    if is_last_turn and on_charger(mover.y, mover.x) and mover.alive:
        if mover.docking_counter >= DOCK[rid]:
            # Exclusive: check no other alive robot on charger
            other_on = other.alive and on_charger(other.y, other.x)
            if not other_on:
                mover.energy += E_CHARGE
                charged = E_CHARGE

    # ── Reward (before cap) ──
    reward = (mover.energy - mover.prev_energy) * REWARD_ALPHA
    if not mover.alive:
        reward -= 100.0

    # ── Cap ──
    mover.energy = min(mover.energy, E_CAP[rid])

    # ── Save prev ──
    mover.prev_energy = mover.energy
    mover.total_reward += reward

    if step_log is not None and (charged > 0 or original_action != action):
        extras = []
        if original_action != action:
            extras.append(f"STUNNED(wanted {ACT_NAMES[original_action]})")
        if charged > 0:
            extras.append(f"charged +{charged:.0f}")
        step_log.append(f"    r{rid}: {', '.join(extras)}")

    return reward


def simulate(strategy_name, r0_start, r1_start,
             r0_policy, r1_policy, r0_first_order=None, verbose=True):
    """
    Simulate one episode.
    r0_policy(r0, r1) → action
    r1_policy(r0, r1) → action
    r0_first_order: list of bool per step, True = r0 first. None = alternating.
    """
    r0 = Robot(0, *r0_start)
    r1 = Robot(1, *r1_start)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {strategy_name}")
        print(f"  r0 start: ({r0.y},{r0.x})  r1 start: ({r1.y},{r1.x})")
        print(f"{'='*70}")

    step_rewards_r0 = []
    step_rewards_r1 = []

    for step in range(MAX_STEPS):
        if not r0.alive and not r1.alive:
            break

        step_log = [] if verbose else None

        # Determine order
        if r0_first_order is not None:
            r0_first = r0_first_order[step % len(r0_first_order)]
        else:
            r0_first = (step % 2 == 0)

        first = (r0, r1) if r0_first else (r1, r0)

        r0_reward_this_step = 0.0
        r1_reward_this_step = 0.0

        for mover, other in [first, (first[1], first[0])]:
            rid = mover.rid
            speed = SPEED[rid]
            for sub in range(speed):
                is_last = (sub == speed - 1)
                if rid == 0:
                    action = r0_policy(r0, r1)
                else:
                    action = r1_policy(r1, r0)
                r = do_substep(mover, other, action, is_last, step_log)
                if rid == 0:
                    r0_reward_this_step += r
                else:
                    r1_reward_this_step += r

        step_rewards_r0.append(r0_reward_this_step)
        step_rewards_r1.append(r1_reward_this_step)

        if verbose and step < 50:  # 只印前 50 步
            order_str = "r0→r1" if r0_first else "r1→r0"
            alive_str = f"r0:{'A' if r0.alive else 'D'} r1:{'A' if r1.alive else 'D'}"
            print(
                f"Step {step:3d} [{order_str}] "
                f"r0({r0.y},{r0.x}) E={r0.energy:6.1f} dock={r0.docking_counter} stun={r0.stun_counter} | "
                f"r1({r1.y},{r1.x}) E={r1.energy:6.1f} dock={r1.docking_counter} stun={r1.stun_counter} | "
                f"R: r0={r0_reward_this_step:+.3f} r1={r1_reward_this_step:+.3f} | {alive_str}"
            )
            for line in (step_log or []):
                print(line)

        if verbose and step == 50:
            print("  ... (truncated, showing summary) ...")

    if verbose:
        print(f"\n  Final: step={step}, r0 E={r0.energy:.1f} alive={r0.alive}, "
              f"r1 E={r1.energy:.1f} alive={r1.alive}")
        print(f"  Total reward: r0={r0.total_reward:+.2f}  r1={r1.total_reward:+.2f}")

    return r0.total_reward, r1.total_reward, step, r0.energy, r1.energy, r0.alive, r1.alive


# ── Policies ───────────────────────────────────────────────────────────

def seek_charger_policy(me, other):
    """Always move toward charger, STAY when on it."""
    if me.y == CHARGER[0] and me.x == CHARGER[1]:
        return 4  # STAY
    return move_toward(me.y, me.x, CHARGER[0], CHARGER[1])


def camp_charger_policy(me, other):
    """Move to charger if not on it, STAY when on it. (Same as seek_charger)"""
    if me.y == CHARGER[0] and me.x == CHARGER[1]:
        return 4  # STAY
    return move_toward(me.y, me.x, CHARGER[0], CHARGER[1])


def active_pursuit_policy(me, other):
    """Always move toward opponent to kill. After kill, go to charger."""
    if not other.alive:
        return seek_charger_policy(me, other)
    return move_toward(me.y, me.x, other.y, other.x)


def main():
    # ── Starting positions ──
    # 典型場景：r0 已在充電座，r1 從角落出發
    r0_start = (2, 2)  # on charger
    r1_start = (0, 0)  # corner, ~4 steps away

    print("=" * 70)
    print("環境參數:")
    print(f"  Grid: {GRID}×{GRID}, Charger: {CHARGER}, range=0")
    print(f"  r0: E={E_INIT[0]}, spd={SPEED[0]}, atk={ATK[0]}, dock={DOCK[0]}, stun={STUN[0]}")
    print(f"  r1: E={E_INIT[1]}, spd={SPEED[1]}, atk={ATK[1]}, dock={DOCK[1]}, stun={STUN[1]}")
    print(f"  e_move={E_MOVE}, e_charge={E_CHARGE}, e_decay={E_DECAY}")
    print(f"  exclusive_charging=True, energy_cap={E_CAP}")
    print(f"  reward = ΔE × {REWARD_ALPHA} - 100 × died")
    print(f"  max_steps={MAX_STEPS}")
    print("=" * 70)

    # ── Strategy A: r0 camp charger ──
    print("\n" + "▓" * 70)
    print("  策略 A：r0 蹲充電座（被撞就回來）")
    print("▓" * 70)
    a_r0, a_r1, a_steps, a_r0e, a_r1e, a_r0a, a_r1a = simulate(
        "策略 A (camp): r0=seek_charger, r1=seek_charger",
        r0_start, r1_start,
        camp_charger_policy, seek_charger_policy,
        r0_first_order=[True, False],  # alternating
    )

    # ── Strategy B: r0 active pursuit ──
    print("\n" + "▓" * 70)
    print("  策略 B：r0 主動出擊追殺 r1")
    print("▓" * 70)
    b_r0, b_r1, b_steps, b_r0e, b_r1e, b_r0a, b_r1a = simulate(
        "策略 B (pursuit): r0=active_pursuit, r1=seek_charger",
        r0_start, r1_start,
        active_pursuit_policy, seek_charger_policy,
        r0_first_order=[True, False],  # alternating
    )

    # ── Also test different starting position ──
    print("\n" + "▓" * 70)
    print("  策略 A'：r1 從 (2,0) 出發（更近）")
    print("▓" * 70)
    a2_r0, a2_r1, *_ = simulate(
        "策略 A' (camp): r1 starts at (2,0)",
        r0_start, (2, 0),
        camp_charger_policy, seek_charger_policy,
        r0_first_order=[True, False],
    )

    print("\n" + "▓" * 70)
    print("  策略 B'：r1 從 (2,0) 出發（更近）")
    print("▓" * 70)
    b2_r0, b2_r1, *_ = simulate(
        "策略 B' (pursuit): r1 starts at (2,0)",
        r0_start, (2, 0),
        active_pursuit_policy, seek_charger_policy,
        r0_first_order=[True, False],
    )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  總 結")
    print("=" * 70)
    print(f"\n  r1 從 (0,0) 出發:")
    print(f"    策略 A (蹲充電座): r0 reward = {a_r0:+.2f}")
    print(f"    策略 B (主動追殺): r0 reward = {b_r0:+.2f}")
    print(f"    差距: B - A = {b_r0 - a_r0:+.2f}")
    print(f"\n  r1 從 (2,0) 出發:")
    print(f"    策略 A (蹲充電座): r0 reward = {a2_r0:+.2f}")
    print(f"    策略 B (主動追殺): r0 reward = {b2_r0:+.2f}")
    print(f"    差距: B - A = {b2_r0 - a2_r0:+.2f}")


if __name__ == '__main__':
    main()
