#!/bin/bash
# 2-Robot Symmetric Evolution Study
# Two identical agents, random spawn, observe learning & behavioral evolution
# Both learn, same attack, same HP — pure symmetric competition
#
# Goal: observe the full learning trajectory — avoidance, aggression, charging, etc.
# Rich wandb logging: charger occupancy, distance to charger, distance between robots,
#   first collision step, outcome rates, per-robot survival, collision patterns

set -e

RUN_NAME="${1:-2robot_sym_evolution}"

echo "=== 2-Robot Symmetric Evolution: $RUN_NAME ==="
echo "6x6, random spawn, both attack=50, HP-ratio, shuffle order"

python train_dqn_vec.py \
  --env-n 6 --num-robots 2 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-0-attack-power 50 --robot-1-attack-power 50 \
  --charger-positions "3,3" --charger-range 0 \
  --random-start-robots "0,1" \
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
