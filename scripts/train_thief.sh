#!/bin/bash
# Thief Scenario v6: Single-Charger Asymmetric Survival (4x4)
# v1-v5 lessons:
#   - HP must die within episode (HP/decay < max_steps)
#   - boundary penalty kills exploration → e_boundary=0
#   - dead-robot replay buffer flooding → fixed with alive_mask
#   - decay=1.0 too harsh (67 steps), decay=0.3 gives 334 steps
#   - NoisyNet correlated noise fails multi-step navigation → use epsilon-greedy
#
# Both HP=100, Strong attack=50 (2-hit kill), Weak attack=1 (negligible)
# 4x4 grid, Charger: (2,2), range=0 (must stand ON tile)
# Decay: 0.3/step, Move: 0.1, Charge: 3.0, No energy cap
#
# Key math:
#   On charger: net +2.7/step → camp 500 steps → reward +67.5
#   Do nothing: die step 334 → reward -105
#   STAY survival (334 steps) vs random walk (263 steps): mild gap
#   Charger reward signal 9x stronger than STAY penalty

set -e

RUN_NAME="${1:-thief_v6}"

echo "=== Thief Scenario v6: $RUN_NAME ==="
echo "HP=100, attack=50/1, decay=0.3, move=0.1, charge=3.0"
echo "4x4 grid, Charger=(2,2), range=0, epsilon-greedy (no NoisyNet)"

python train_dqn_vec.py \
  --env-n 4 --num-robots 2 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-0-attack-power 50 --robot-1-attack-power 1 \
  --charger-positions "2,2" --charger-range 0 \
  --thief-spawn \
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
