"""
Unit tests for alliance energy-sharing mechanism.

Tests:
  1. mode=none: rewards are identical to baseline (no sharing)
  2. charge sharing: r0 gets charge=+8, r1 gets 0 → after sharing r0 keeps 8*2/3, r1 gets 8*1/3
  3. collision sharing: r0 gets collision=-30, r1 gets 0 → sharing distributes the loss
  4. no alliance_groups: degrades to no sharing, no crash
  5. outsider (r2) final energy equals base energy (unaffected by r0+r1 alliance)
"""
import sys
import os

# Make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from gym import RobotVacuumGymEnv


def _make_env(num_robots=3, sharing_mode='none', alliance_groups=None,
              sharing_events=None, w_self=2/3, w_ally=1/3):
    if sharing_events is None:
        sharing_events = ['charge', 'collision']
    config = {
        'n': 5,
        'num_robots': num_robots,
        'initial_energy': 100,
        'robot_energies': [100] * num_robots,
        'e_move': 1,
        'e_charge': 5,
        'e_collision': 3,
        'e_boundary': 50,
        'e_collision_active_one_sided': None,
        'e_collision_active_two_sided': None,
        'e_collision_passive': None,
        'n_steps': 500,
        'epsilon': 0.2,
        'charger_positions': [(2, 2)],
        'dust_max': 10.0,
        'dust_rate': 0.5,
        'dust_epsilon': 0.5,
        'charger_dust_max_ratio': 0.3,
        'charger_dust_rate_ratio': 0.5,
        'dust_reward_scale': 0.05,
        'dust_enabled': True,
        'exclusive_charging': False,
        'charger_range': 1,
        'robot_speeds': [1] * num_robots,
        'random_start_robots': set(),
        'robot_start_positions': {},
        'agent_types_mode': 'off',
        'triangle_agent_id': None,
        'heterotype_charge_mode': 'off',
        'heterotype_charge_factor': 1.0,
        'energy_cap': None,
        'e_decay': 0.0,
        'robot_attack_powers': None,
        'thief_spawn': False,
        'legacy_obs': False,
        # Alliance sharing params
        'alliance_groups': alliance_groups,
        'energy_sharing_mode': sharing_mode,
        'energy_sharing_events': sharing_events,
        'energy_sharing_self_weight': w_self,
        'energy_sharing_ally_weight': w_ally,
    }
    return RobotVacuumGymEnv(config=config)


# ---------------------------------------------------------------------------
# Test 1: mode=none — rewards identical to baseline
# ---------------------------------------------------------------------------
def test_mode_none_rewards_unchanged():
    """When mode=none, energy sharing must not affect rewards at all."""
    env_base = _make_env(num_robots=2, sharing_mode='none', alliance_groups=None)
    env_share = _make_env(num_robots=2, sharing_mode='none', alliance_groups=[{0, 1}])

    import random as _r
    import numpy as _np

    _r.seed(0)
    _np.random.seed(0)
    obs_base, _ = env_base.reset()

    _r.seed(0)
    _np.random.seed(0)
    obs_share, _ = env_share.reset()

    actions = [4, 4]  # both STAY
    _, rewards_base, _, _, _ = env_base.step(actions)
    _, rewards_share, _, _, _ = env_share.step(actions)

    for agent_id in rewards_base:
        assert abs(rewards_base[agent_id] - rewards_share[agent_id]) < 1e-6, (
            f"Rewards differ for {agent_id}: base={rewards_base[agent_id]}, "
            f"share={rewards_share[agent_id]}"
        )


# ---------------------------------------------------------------------------
# Test 2: charge sharing
# ---------------------------------------------------------------------------
def test_charge_sharing():
    """r0 charges (+8), r1 does not. After sharing: r0 keeps 8*2/3, r1 gets 8*1/3."""
    env = _make_env(num_robots=2, sharing_mode='event_only',
                    alliance_groups=[{0, 1}], sharing_events=['charge'],
                    w_self=2/3, w_ally=1/3)
    env.reset()

    # Directly invoke _apply_energy_sharing with fabricated events
    energy_events = [
        {'charge': 8.0, 'collision': 0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0, 'collision': 0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
    ]
    adjustments = env._apply_energy_sharing(
        energy_events,
        [{0, 1}],
        'event_only',
        ['charge'],
        2/3,
        1/3,
    )
    # r0 gets: 8*2/3 - 8 = -8/3 (gives away 1/3 of charge)
    # r1 gets: 8*1/3 = +8/3
    expected_r0 = 8.0 * (2/3) - 8.0  # = -8/3
    expected_r1 = 8.0 * (1/3)         # = +8/3

    assert abs(adjustments[0] - expected_r0) < 1e-6, \
        f"r0 adjustment: expected {expected_r0:.4f}, got {adjustments[0]:.4f}"
    assert abs(adjustments[1] - expected_r1) < 1e-6, \
        f"r1 adjustment: expected {expected_r1:.4f}, got {adjustments[1]:.4f}"

    # Net energy should be conserved within the alliance
    assert abs(adjustments[0] + adjustments[1]) < 1e-6, \
        "Net energy change within alliance should be zero"


# ---------------------------------------------------------------------------
# Test 3: collision sharing
# ---------------------------------------------------------------------------
def test_collision_sharing():
    """r0 takes collision=-30, r1 takes 0. After sharing r0 keeps -30*2/3, r1 gets -30*1/3."""
    env = _make_env(num_robots=2, sharing_mode='event_only',
                    alliance_groups=[{0, 1}], sharing_events=['collision'],
                    w_self=2/3, w_ally=1/3)
    env.reset()

    energy_events = [
        {'charge': 0.0, 'collision': -30.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0, 'collision':   0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
    ]
    adjustments = env._apply_energy_sharing(
        energy_events,
        [{0, 1}],
        'event_only',
        ['collision'],
        2/3,
        1/3,
    )
    # r0 shares 1/3 of its -30 pain to r1
    # adjustments[0] = -30*(2/3) - (-30) = -20 + 30 = +10 (r0 gets relief)
    # adjustments[1] = -30*(1/3) = -10  (r1 absorbs some pain)
    expected_r0 = (-30.0) * (2/3) - (-30.0)   # = +10
    expected_r1 = (-30.0) * (1/3)              # = -10

    assert abs(adjustments[0] - expected_r0) < 1e-6, \
        f"r0 adjustment: expected {expected_r0:.4f}, got {adjustments[0]:.4f}"
    assert abs(adjustments[1] - expected_r1) < 1e-6, \
        f"r1 adjustment: expected {expected_r1:.4f}, got {adjustments[1]:.4f}"

    # Conservation
    assert abs(adjustments[0] + adjustments[1]) < 1e-6, \
        "Net energy change within alliance should be zero"


# ---------------------------------------------------------------------------
# Test 4: no alliance_groups — degrades to no sharing, no crash
# ---------------------------------------------------------------------------
def test_no_alliance_groups_no_crash():
    """When alliance_groups=None, _apply_energy_sharing returns all-zeros and doesn't crash."""
    env = _make_env(num_robots=3, sharing_mode='event_only', alliance_groups=None)
    env.reset()

    energy_events = [
        {'charge': 5.0, 'collision': -10.0, 'move': -1.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0, 'collision': 0.0,   'move': -1.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0, 'collision': 0.0,   'move': 0.0,  'decay': 0.0, 'boundary': 0.0},
    ]
    adjustments = env._apply_energy_sharing(
        energy_events,
        None,
        'event_only',
        ['charge', 'collision'],
        2/3,
        1/3,
    )
    for i in range(3):
        assert adjustments[i] == 0.0, f"Expected 0.0 adjustment for r{i}, got {adjustments[i]}"


# ---------------------------------------------------------------------------
# Test 5: outsider (r2) energy unaffected by r0+r1 alliance
# ---------------------------------------------------------------------------
def test_outsider_unaffected():
    """r2 is not in any alliance, so its energy must be unaffected by r0+r1 sharing."""
    env = _make_env(num_robots=3, sharing_mode='event_only',
                    alliance_groups=[{0, 1}], sharing_events=['charge', 'collision'],
                    w_self=2/3, w_ally=1/3)
    env.reset()

    energy_events = [
        {'charge': 8.0,  'collision': -30.0, 'move': -1.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0,  'collision':   0.0, 'move': -1.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 10.0, 'collision': -5.0,  'move': -1.0, 'decay': 0.0, 'boundary': 0.0},
    ]
    adjustments = env._apply_energy_sharing(
        energy_events,
        [{0, 1}],
        'event_only',
        ['charge', 'collision'],
        2/3,
        1/3,
    )
    assert adjustments[2] == 0.0, \
        f"Outsider r2 should have 0 adjustment, got {adjustments[2]}"

    # r0 and r1 net should cancel
    assert abs(adjustments[0] + adjustments[1]) < 1e-6, \
        "Net energy change within r0+r1 alliance should be zero"


# ---------------------------------------------------------------------------
# Test 6: dead ally does NOT receive shared energy
# ---------------------------------------------------------------------------
def test_dead_ally_no_sharing():
    """If r1 is dead, r0's charge should NOT be shared to r1.
    r0 must keep its full charge, and r1's adjustment must be 0."""
    env = _make_env(num_robots=2, sharing_mode='event_only',
                    alliance_groups=[{0, 1}], sharing_events=['charge'],
                    w_self=2/3, w_ally=1/3)
    env.reset()

    # Kill r1 by setting energy to 0 and marking inactive
    env.env.robots[1]['energy'] = 0.0
    env.env.robots[1]['is_active'] = False

    energy_events = [
        {'charge': 8.0, 'collision': 0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0, 'collision': 0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
    ]
    adjustments = env._apply_energy_sharing(
        energy_events,
        [{0, 1}],
        'event_only',
        ['charge'],
        2/3,
        1/3,
    )
    # r1 is dead: should receive nothing
    assert adjustments[1] == 0.0, \
        f"Dead r1 should get 0 adjustment, got {adjustments[1]}"
    # r0 has no alive allies: should keep full charge (adjustment = 0)
    assert adjustments[0] == 0.0, \
        f"r0 with no alive allies should keep full charge (adj=0), got {adjustments[0]}"


# ---------------------------------------------------------------------------
# Test 7: 3-robot alliance, one dead — sharing among alive only
# ---------------------------------------------------------------------------
def test_3robot_one_dead_sharing():
    """In a {0,1,2} alliance, if r2 is dead, r0's charge should be shared
    only between r0 and r1 (not split 3-ways)."""
    env = _make_env(num_robots=3, sharing_mode='event_only',
                    alliance_groups=[{0, 1, 2}], sharing_events=['charge'],
                    w_self=2/3, w_ally=1/3)
    env.reset()

    # Kill r2
    env.env.robots[2]['energy'] = 0.0
    env.env.robots[2]['is_active'] = False

    energy_events = [
        {'charge': 9.0, 'collision': 0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0, 'collision': 0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
        {'charge': 0.0, 'collision': 0.0, 'move': 0.0, 'decay': 0.0, 'boundary': 0.0},
    ]
    adjustments = env._apply_energy_sharing(
        energy_events,
        [{0, 1, 2}],
        'event_only',
        ['charge'],
        2/3,
        1/3,
    )
    # r2 is dead: gets nothing
    assert adjustments[2] == 0.0, f"Dead r2 got {adjustments[2]}"
    # With only 1 alive ally (r1), r0 shares: 9*(1-2/3) = 3.0 to r1
    # r0 adj = 9*2/3 - 9 = -3.0
    # r1 adj = +3.0
    assert abs(adjustments[0] - (-3.0)) < 1e-6, f"r0 adj expected -3.0, got {adjustments[0]}"
    assert abs(adjustments[1] - 3.0) < 1e-6, f"r1 adj expected 3.0, got {adjustments[1]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
