#!/bin/bash
# 4R-B: Symmetric 4-robot with RANDOMIZED stepping order (sanity check)
# Same as 4robot_symmetric but --shuffle-step-order eliminates fixed r0>r1>r2>r3 bias
# 3M episodes (longer run for thorough learning)
#
# Goal: verify that without stepping-order exploit, agents can still learn
#       kill-to-monopolize strategy, and that win rates are ~25% each (symmetric)

set -e

RUN_NAME="${1:-4robot_sym_shuffle}"

echo "=== 4R-B Symmetric Shuffle: $RUN_NAME ==="
echo "5x5, 4 corners, charger=(2,2), all attack=50, HP-ratio, SHUFFLE order"

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
  --shuffle-step-order \
  --reward-mode hp-ratio --reward-alpha 0.2 \
  --v-min -120 --v-max 20 \
  --epsilon-schedule exponential \
  --n-step 20 --gamma 0.99 \
  --max-episode-steps 500 \
  --num-episodes 3000000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --no-eval-after-training --save-frequency 100000 \
  --wandb-mode online \
  --wandb-run-name "$RUN_NAME"

echo "=== Training complete ==="
