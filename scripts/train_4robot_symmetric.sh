#!/bin/bash
# 4-Robot Symmetric Sanity Check: Ruling Behavior Emergence
# 5x5 grid, 4 equal robots at corners, 1 charger at center (2,2)
# All attack=50 (2-hit kill), HP=100, cap=100
# HP-ratio reward — no explicit attack bonus
# Goal: verify "kill-to-monopolize" strategy emerges in symmetric 4-agent setting
#
# Key math:
#   Each corner → (2,2): Manhattan distance 4, travel cost 4×0.4=1.6 HP
#   On charger: net +2.7/step, capped at 100
#   Without charger: die at step ~333 (100/0.3)
#   2-hit kill: 50×2=100, any robot can kill any other in 2 collisions
#   Expected: all 4 learn aggressive charger-seeking, winner ~25% each

set -e

RUN_NAME="${1:-4robot_symmetric}"

echo "=== 4-Robot Symmetric: $RUN_NAME ==="
echo "5x5, 4 corners, charger=(2,2), all attack=50, HP-ratio"

python train_dqn_vec.py \
  --env-n 5 --num-robots 4 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-2-energy 100 --robot-3-energy 100 \
  --robot-0-attack-power 50 --robot-1-attack-power 50 \
  --robot-2-attack-power 50 --robot-3-attack-power 50 \
  --charger-positions "2,2" --charger-range 0 \
  --robot-start-positions "0,0;0,4;4,0;4,4" \
  --energy-cap 100 \
  --e-decay 0.3 --e-move 0.1 --e-charge 3.0 \
  --e-collision 50 --e-boundary 0 \
  --no-dust \
  --no-noisy \
  --reward-mode hp-ratio --reward-alpha 0.2 \
  --v-min -120 --v-max 20 \
  --epsilon-schedule exponential \
  --n-step 20 --gamma 0.99 \
  --max-episode-steps 500 \
  --num-episodes 1500000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --no-eval-after-training --save-frequency 50000 \
  --wandb-mode online \
  --wandb-run-name "$RUN_NAME"

echo "=== Training complete ==="
