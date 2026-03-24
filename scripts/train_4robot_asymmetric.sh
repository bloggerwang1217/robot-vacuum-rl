#!/bin/bash
# 4-Robot Asymmetric: One Strong Ruler Candidate
# Same as symmetric but r0 attack=50 (2-hit kill), others attack=34 (3-hit kill)
# Question: does this mild asymmetry produce a stable ruler (r0 win rate >> 25%)?
#
# Key math:
#   r0 kills anyone in 2 hits (50×2=100)
#   Others kill anyone in 3 hits (34×3=102 > 100)
#   r0 advantage: kills 33% faster → can chain kills more efficiently
#   But r0 still takes same damage when hit (50 from r0, 34 from others)

set -e

RUN_NAME="${1:-4robot_asymmetric}"

echo "=== 4-Robot Asymmetric: $RUN_NAME ==="
echo "5x5, 4 corners, charger=(2,2), r0=50 vs r1/r2/r3=34, HP-ratio"

python train_dqn_vec.py \
  --env-n 5 --num-robots 4 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-2-energy 100 --robot-3-energy 100 \
  --robot-0-attack-power 50 --robot-1-attack-power 34 \
  --robot-2-attack-power 34 --robot-3-attack-power 34 \
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
