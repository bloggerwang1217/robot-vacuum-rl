#!/usr/bin/env python3
"""
掃描 docking_steps 和 stun_steps 對策略 A vs B reward 差距的影響。
重用 strategy_comparison.py 的模擬引擎。
"""

# ── Parameters (same as strategy_comparison.py) ──
GRID = 5
CHARGER = (2, 2)
E_INIT = [100.0, 100.0]
SPEED = [2, 1]
ATK = [30.0, 2.0]
E_MOVE = 0.0
E_CHARGE = 8.0
E_DECAY = 0.5
E_CAP = [100.0, 100.0]
MAX_STEPS = 500
REWARD_ALPHA = 0.05

DY = [-1, 1, 0, 0, 0]
DX = [0, 0, -1, 1, 0]

# These will be overridden per sweep
DOCK = [2, 0]
STUN = [2, 0]


class Robot:
    def __init__(self, rid, y, x, dock, stun):
        self.rid = rid
        self.y, self.x = y, x
        self.energy = E_INIT[rid]
        self.alive = True
        self.stun_counter = 0
        self.docking_counter = 0
        self.prev_energy = E_INIT[rid]
        self.total_reward = 0.0
        self._dock = dock
        self._stun = stun


def move_toward(fy, fx, ty, tx):
    dy, dx = ty - fy, tx - fx
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1 if dy > 0 else 4
    else:
        return 2 if dx < 0 else 3 if dx > 0 else 4


def on_charger(y, x):
    return y == CHARGER[0] and x == CHARGER[1]


def in_bounds(y, x):
    return 0 <= y < GRID and 0 <= x < GRID


def do_substep(mover, other, action, is_last, dock_cfg, stun_cfg):
    rid = mover.rid
    if not mover.alive:
        return 0.0

    if mover.stun_counter > 0:
        action = 4
        mover.stun_counter -= 1

    if E_DECAY > 0 and is_last:
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

    if is_move:
        mover.energy -= E_MOVE

    is_boundary = is_move and not in_bounds(py, px)
    if is_boundary:
        is_move = False

    collision = False
    if is_move and other.alive and py == other.y and px == other.x:
        collision = True
        kby, kbx = other.y + dy, other.x + dx
        if in_bounds(kby, kbx):
            mover.y, mover.x = py, px
            other.y, other.x = kby, kbx
        other.energy -= ATK[rid]
        other.docking_counter = 0
        other.stun_counter = stun_cfg[other.rid]
        other.energy = max(other.energy, 0.0)
        if other.energy <= 0:
            other.alive = False
    elif is_move and not collision:
        mover.y, mover.x = py, px

    if is_last:
        if on_charger(mover.y, mover.x) and mover.alive and mover.stun_counter == 0:
            mover.docking_counter += 1
        else:
            mover.docking_counter = 0

    charged = 0.0
    if is_last and on_charger(mover.y, mover.x) and mover.alive:
        if mover.docking_counter >= dock_cfg[rid]:
            other_on = other.alive and on_charger(other.y, other.x)
            if not other_on:
                mover.energy += E_CHARGE
                charged = E_CHARGE

    reward = (mover.energy - mover.prev_energy) * REWARD_ALPHA
    if not mover.alive:
        reward -= 100.0

    mover.energy = min(mover.energy, E_CAP[rid])
    mover.prev_energy = mover.energy
    mover.total_reward += reward
    return reward


def simulate(r0_start, r1_start, r0_policy, r1_policy, dock_cfg, stun_cfg):
    r0 = Robot(0, *r0_start, dock_cfg[0], stun_cfg[0])
    r1 = Robot(1, *r1_start, dock_cfg[1], stun_cfg[1])

    r1_death_step = None

    for step in range(MAX_STEPS):
        if not r0.alive and not r1.alive:
            break

        r0_first = (step % 2 == 0)
        first = (r0, r1) if r0_first else (r1, r0)

        for mover, other in [first, (first[1], first[0])]:
            rid = mover.rid
            speed = SPEED[rid]
            for sub in range(speed):
                is_last = (sub == speed - 1)
                if rid == 0:
                    action = r0_policy(r0, r1)
                else:
                    action = r1_policy(r1, r0)
                do_substep(mover, other, action, is_last, dock_cfg, stun_cfg)

        if not r1.alive and r1_death_step is None:
            r1_death_step = step

    return r0.total_reward, r1.total_reward, r1_death_step, r0.energy


def seek_charger(me, other):
    if me.y == CHARGER[0] and me.x == CHARGER[1]:
        return 4
    return move_toward(me.y, me.x, CHARGER[0], CHARGER[1])


def active_pursuit(me, other):
    if not other.alive:
        return seek_charger(me, other)
    return move_toward(me.y, me.x, other.y, other.x)


def main():
    r0_start = (2, 2)
    r1_start = (0, 0)

    print("=" * 90)
    print("  參數掃描：docking_steps / stun_steps 對策略 A vs B 的影響")
    print("  r0: E=100, spd=2, atk=30 | r1: E=100, spd=1, atk=2")
    print("  e_charge=8, e_decay=0.5, exclusive_charging, max_steps=500")
    print("=" * 90)

    # ── Sweep docking_steps ──
    print(f"\n{'─'*90}")
    print(f"  掃描 r0 docking_steps（r0 stun 固定 = 2）")
    print(f"{'─'*90}")
    print(f"{'dock':>6} │ {'A:reward':>10} {'A:r1死步':>10} {'A:r0殘血':>10} │ "
          f"{'B:reward':>10} {'B:r1死步':>10} {'B:r0殘血':>10} │ {'B-A':>8} {'比例':>8}")
    print(f"{'─'*6}─┼{'─'*34}─┼{'─'*34}─┼{'─'*18}")

    for dock_val in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        dock_cfg = [dock_val, 0]
        stun_cfg = [2, 0]

        a_r0, _, a_death, a_e = simulate(r0_start, r1_start, seek_charger, seek_charger, dock_cfg, stun_cfg)
        b_r0, _, b_death, b_e = simulate(r0_start, r1_start, active_pursuit, seek_charger, dock_cfg, stun_cfg)
        diff = b_r0 - a_r0
        pct = diff / max(abs(a_r0), 0.01) * 100

        print(f"{dock_val:>6} │ {a_r0:>+10.2f} {a_death or 'alive':>10} {a_e:>10.1f} │ "
              f"{b_r0:>+10.2f} {b_death or 'alive':>10} {b_e:>10.1f} │ {diff:>+8.2f} {pct:>+7.1f}%")

    # ── Sweep stun_steps ──
    print(f"\n{'─'*90}")
    print(f"  掃描 r0 stun_steps（r0 docking 固定 = 2）")
    print(f"{'─'*90}")
    print(f"{'stun':>6} │ {'A:reward':>10} {'A:r1死步':>10} {'A:r0殘血':>10} │ "
          f"{'B:reward':>10} {'B:r1死步':>10} {'B:r0殘血':>10} │ {'B-A':>8} {'比例':>8}")
    print(f"{'─'*6}─┼{'─'*34}─┼{'─'*34}─┼{'─'*18}")

    for stun_val in [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
        dock_cfg = [2, 0]
        stun_cfg = [stun_val, 0]

        a_r0, _, a_death, a_e = simulate(r0_start, r1_start, seek_charger, seek_charger, dock_cfg, stun_cfg)
        b_r0, _, b_death, b_e = simulate(r0_start, r1_start, active_pursuit, seek_charger, dock_cfg, stun_cfg)
        diff = b_r0 - a_r0
        pct = diff / max(abs(a_r0), 0.01) * 100

        print(f"{stun_val:>6} │ {a_r0:>+10.2f} {a_death or 'alive':>10} {a_e:>10.1f} │ "
              f"{b_r0:>+10.2f} {b_death or 'alive':>10} {b_e:>10.1f} │ {diff:>+8.2f} {pct:>+7.1f}%")

    # ── Combined sweep: high stun + high dock ──
    print(f"\n{'─'*90}")
    print(f"  組合掃描（dock, stun 同時增加）")
    print(f"{'─'*90}")
    print(f"{'dock,stun':>10} │ {'A:reward':>10} {'A:r1死步':>10} {'A:r0殘血':>10} │ "
          f"{'B:reward':>10} {'B:r1死步':>10} {'B:r0殘血':>10} │ {'B-A':>8} {'比例':>8}")
    print(f"{'─'*10}─┼{'─'*34}─┼{'─'*34}─┼{'─'*18}")

    for dock_val, stun_val in [(2, 2), (3, 3), (4, 4), (5, 5), (3, 6), (4, 8), (5, 10), (2, 10), (2, 20)]:
        dock_cfg = [dock_val, 0]
        stun_cfg = [stun_val, 0]

        a_r0, _, a_death, a_e = simulate(r0_start, r1_start, seek_charger, seek_charger, dock_cfg, stun_cfg)
        b_r0, _, b_death, b_e = simulate(r0_start, r1_start, active_pursuit, seek_charger, dock_cfg, stun_cfg)
        diff = b_r0 - a_r0
        pct = diff / max(abs(a_r0), 0.01) * 100

        label = f"{dock_val},{stun_val}"
        print(f"{label:>10} │ {a_r0:>+10.2f} {a_death or 'alive':>10} {a_e:>10.1f} │ "
              f"{b_r0:>+10.2f} {b_death or 'alive':>10} {b_e:>10.1f} │ {diff:>+8.2f} {pct:>+7.1f}%")


if __name__ == '__main__':
    main()
