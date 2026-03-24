#!/bin/bash
# 4R-A v9 Extension: Strong attacker (r0) vs 3 weak defenders clustered at charger
# Direct extension of v9 (2-robot thief scenario) to 4 robots
#
# Layout (5x5):
#   r0(0,0) far from charger (Manhattan distance 4)
#   r1(2,2) ON charger
#   r2(2,1) adjacent left of charger
#   r3(3,2) adjacent below charger
#
# Key: all 3 kills within ~10-12 steps → within n-step=20 credit assignment window
# r0 attack=50 (2-hit kill), r1/r2/r3 attack=1 (negligible)
# All 4 agents LEARN (no scripted)
# Shuffle step order to avoid sequential bias

set -e

RUN_NAME="${1:-4robot_v9ext}"

echo "=== 4R-A v9 Extension: $RUN_NAME ==="
echo "5x5, r0=(0,0) atk=50, r1=(2,2) r2=(2,1) r3=(3,2) atk=1, HP-ratio, shuffle"

python train_dqn_vec.py \
  --env-n 5 --num-robots 4 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-2-energy 100 --robot-3-energy 100 \
  --robot-0-attack-power 50 --robot-1-attack-power 1 \
  --robot-2-attack-power 1 --robot-3-attack-power 1 \
  --charger-positions "2,2" --charger-range 0 \
  --robot-start-positions "0,0;2,2;2,1;3,2" \
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
