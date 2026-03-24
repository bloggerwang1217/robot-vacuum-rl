#!/bin/bash
# Arms Race: Speed=3 (r0 triple speed advantage)
# r0: HP=100, attack=50, speed=3, delta-energy reward, exclusive charging
# r1: HP=500, attack=0, speed=1, scripted seek-charger

set -e
RUN_NAME="${1:-pest_spd3}"
echo "=== Arms Race speed=3: $RUN_NAME ==="

python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --robot-0-energy 100 --robot-1-energy 500 \
  --robot-0-attack-power 50 --robot-1-attack-power 0 \
  --robot-0-speed 3 --robot-1-speed 1 \
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
