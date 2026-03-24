#!/bin/bash
# Thief Scenario v7: Scripted Camper — Strong Must Displace
# v6 lesson: thief_spawn puts r1 ADJACENT (not ON) charger → charger empty
#   when r0 arrives → no displacement needed → peaceful dictator
#
# v7 fix: r1 starts ON charger (2,2) and is SCRIPTED to STAY.
#   r0 starts at (0,0), must learn: navigate → collide (×2) → camp.
#
# Both HP=100, Strong attack=50 (2-hit kill), Weak attack=1 (negligible)
# 4x4 grid, Charger: (2,2), range=0 (must stand ON tile)
# Decay: 0.3/step, Move: 0.1, Charge: 3.0, No energy cap
#
# Key math:
#   r1 on charger: net +2.7/step (unkillable unless r0 hits twice)
#   r0 travel cost (0,0)→(2,2): 4 steps × 0.4 = 1.6 HP
#   r0 must collide r1 twice: -0.3 decay × ~2 steps = 0.6 HP during attack
#   After kill: r0 camps charger, net +2.7/step for ~496 remaining steps
#   Expected r0 reward: ~496 × 2.7 × 0.05 - 1.6 × 0.05 ≈ +66.9
#   Without attacking: r0 dies at step 250 (100/0.4) → reward -105
#   Gap: +66.9 vs -105 = massive incentive to attack

set -e

RUN_NAME="${1:-thief_v7}"

echo "=== Thief Scenario v7: $RUN_NAME ==="
echo "r1 SCRIPTED on charger (2,2), r0 at (0,0), must displace"
echo "HP=100, attack=50/1, decay=0.3, move=0.1, charge=3.0"

python train_dqn_vec.py \
  --env-n 4 --num-robots 2 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-0-attack-power 50 --robot-1-attack-power 1 \
  --charger-positions "2,2" --charger-range 0 \
  --robot-start-positions "0,0;2,2" \
  --scripted-robots 1 \
  --e-decay 0.3 --e-move 0.1 --e-charge 3.0 \
  --e-collision 50 --e-boundary 0 \
  --no-dust \
  --no-noisy \
  --v-min -120 --v-max 80 \
  --epsilon-schedule exponential \
  --n-step 20 --gamma 0.99 \
  --max-episode-steps 500 \
  --num-episodes 2000000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --no-eval-after-training --save-frequency 10000 \
  --wandb-mode online \
  --wandb-run-name "$RUN_NAME"

echo "=== Training complete ==="
