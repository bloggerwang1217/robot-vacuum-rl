#!/bin/bash
# Pest Experiment v2: r0 learns vs tanky seek-charger r1
#
# r0: HP=100, attack=50, speed=2, delta-energy reward, exclusive charging
# r1: HP=500, attack=0, speed=1, scripted seek-charger, takes 10 hits to kill
#
# Key fixes from v1:
#   - No energy cap → r1 keeps its 500 HP (won't be clipped to 100)
#   - delta-energy reward (not hp-ratio) → each lost charge cycle = -0.15 reward (clear signal)
#
# r1 can push r0 off charger (0 damage) but blocks r0 from charging.
# r0 must learn: invest 10 hits to permanently kill r1, then charge in peace.

set -e

RUN_NAME="${1:-pest_v2}"

echo "=== Pest Experiment v2: $RUN_NAME ==="
echo "5x5, r0(HP=100,atk=50,spd=2) vs r1(HP=500,atk=0,seek-charger), exclusive, delta-energy"

python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --robot-0-energy 100 --robot-1-energy 500 \
  --robot-0-attack-power 50 --robot-1-attack-power 0 \
  --robot-0-speed 2 --robot-1-speed 1 \
  --charger-positions "2,2" --charger-range 0 \
  --random-start-robots "0,1" \
  --e-decay 0.3 --e-move 0.1 --e-charge 3.0 \
  --e-collision 50 --e-boundary 0 \
  --no-dust \
  --no-noisy \
  --exclusive-charging \
  --shuffle-step-order \
  --seek-charger-robots 1 \
  --n-step 20 --gamma 0.99 \
  --max-episode-steps 500 \
  --num-episodes 3000000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --no-eval-after-training --save-frequency 100000 \
  --wandb-mode online \
  --wandb-run-name "$RUN_NAME"

echo "=== Training complete ==="
