# Emergent Aggression via Two-Phase Curriculum

## Research Goal

Can a robot learn to **kill a competitor to monopolize a charger** — without any explicit attack reward?

Attack behavior must be **emergent**: the agent chooses to attack purely because it maximizes long-term reward under resource scarcity. The reward function contains no attack bonus; only indirect signals (energy change, death penalty).

---

## Why Naive Joint Training Fails

If both robots train simultaneously from scratch, the system converges to a **"peaceful dictatorship" equilibrium**:

1. r0 is stronger → r1 learns to avoid the charger (learned helplessness)
2. r1 never contests the charger → r0 has no reason to leave and chase
3. Attacking costs energy (18 steps away ≈ 92 reward lost) with no gain → r0 stays put
4. Stable: neither aggression nor competition emerges

The necessary condition for aggression is that **r1 must actively compete for the resource**. Otherwise, r0 already monopolizes the charger without killing anyone.

---

## Two-Phase Curriculum

### Phase 1 — r1 Learns to Seek the Charger

| Parameter | Value |
|-----------|-------|
| r0 | Random walk (not trained) |
| r1 | DQN, trains from scratch |
| Goal | r1 learns to occupy the charger |
| Result | r1 charger occupancy: **95%+** |
| Model | `stun5_r1_seek_v2` |

r1 learns a stable **pest strategy**: navigate to the charger and stay there regardless of r0's behavior. With r0 doing random walk, r1 faces no adversarial pressure — it can focus entirely on learning the seek-and-hold behavior.

### Phase 2 — r0 Learns to Attack

| Parameter | Value |
|-----------|-------|
| r0 | DQN, trains from scratch |
| r1 | **Frozen** at Phase 1 checkpoint |
| Goal | r0 learns to reclaim the charger |
| Result | r0 develops offensive pursuit behavior |
| Model | `stun5_r0_vs_pest_v2` |

With r1 **frozen** at its pest policy, r0 faces a fixed obstacle: the charger is always occupied. The only way to access the charger is to displace r1. Over millions of episodes, r0 discovers that attacking r1 (stunning it for 5 steps) creates a window to charge — and that **double-hitting** (hitting again before r1 recovers) can kill r1 outright.

**Why freezing r1 is critical**: if r1 were also training, it would adapt — fleeing, counter-attacking, or learning to avoid the charger entirely. This mutual adaptation prevents the attacker from finding a stable aggressive policy. Freezing r1 creates a consistent pressure that forces r0 to solve the problem.

---

## Key Metrics

### Attack Rate
Fraction of game steps where r0 lands at least one hit on r1. Rises as r0 learns to pursue.

### Offensive Hit Rate
Among all hits, the fraction where **r0 is not on the charger** immediately after hitting. Distinguishes active pursuit from passive defense (hitting r1 while sitting on the charger).

> *Offensive hit rate: the fraction of all hits where robot 0 is not on the charger immediately after landing the hit — indicating active pursuit rather than passive defense from the charger tile.*

### Double-Hit Rate
Among attack steps, the fraction where r0 lands two consecutive hits. Double-hitting is the mechanism for killing r1 (second hit while r1 is still stunned from the first).

---

## Environment Parameters (stun5 series)

| Parameter | Value |
|-----------|-------|
| Grid | 5×5 |
| Charger | (2,2), exclusive |
| r0 speed / r1 speed | 2 / 1 |
| r0 attack / r1 attack | 30 / 2 |
| r0 stun steps / r1 stun steps | 5 / 1 |
| r0 docking steps | 2 (0 in v3) |
| e_move | 1 |
| e_charge | 8 |
| e_collision | 30 |
| e_decay | 0.5 |
| n-step return | 20 |
| γ | 0.999 |
| Max steps/episode | 500 |

**Stun asymmetry** (r0 stuns r1 for 5 steps, r1 stuns r0 for only 1) is what makes attack viable: r0 can land a follow-up hit while r1 is immobilized, enabling kills.

---

## Experiment Lineage

```
stun5_r1_seek_charger_p2  (r1 learns charger, e_move=0, buggy stun)
        │
        └── stun5_r1_seek_v2  (continued, e_move=1, stun bug fixed, 3M ep)
                │   checkpoint: episode_2130001
                │
                ├── stun5_r0_vs_pest_v2  (r0 learns attack, 5M ep, dock=2)
                │       → offensive hit rate rises to 70-80% after 5M ep
                │
                └── stun5_r0_vs_pest_v3_nodock  (same, dock=0 for both)
                        → ongoing
```

---

## Results Summary (stun5_r0_vs_pest_v2)

- **2M–5M episodes**: attack rate rises to ~12.5%, but offensive rate near 0% (r0 only hits from the charger)
- **5M+ episodes**: attack rate drops to ~0.5%, but **offensive rate jumps to 70–80%** and double-hit rate to 60–78%
- Interpretation: r0 converges to a low-frequency but high-quality aggressive strategy — rarely leaving the charger, but when it does, it actively pursues and double-hits r1
