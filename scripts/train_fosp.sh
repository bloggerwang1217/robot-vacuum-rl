#!/bin/bash
# Frozen-Opponent Self-Play (FOSP) — stabilize emergent aggression
#
# Problem: IDQN co-adaptation cycling makes aggression transient
# Solution: Train agents in alternating phases with frozen opponents
#
# Phase 1: r0 learns vs random r1        (r0 learns pursuit/kill)
# Phase 2: r1 learns vs frozen r0        (r1 learns survival vs aggressor)
# Phase 3: r0 learns vs frozen r1        (r0 refines pursuit of fleeing r1)
# Phase 4: both learn from stable init   (fine-tune together)
#
# Usage: bash scripts/train_fosp.sh [RUN_NAME] [GPU]

set -e
source ~/.venv/bin/activate

RUN_NAME=${1:-fosp_r4}
GPU=${2:-0}
BASE_DIR="./models"

# Common R4 parameters (proven to produce aggression)
COMMON_ARGS="--env-n 5 --num-robots 2 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-0-speed 1 --robot-1-speed 1 \
  --charger-positions 2,2 --charger-range 1 --no-dust \
  --e-move 0 --e-charge 8 --e-collision 100 --e-boundary 0 --e-decay 5 \
  --n-step 20 --gamma 0.99 --max-episode-steps 200 \
  --num-envs 256 --no-dueling --no-noisy --no-c51 \
  --batch-env --no-eval-after-training --save-frequency 50000 \
  --random-start-robots 0,1 \
  --gpu $GPU --wandb-mode disabled"

# Helper: find highest episode checkpoint in a model dir
find_checkpoint() {
    local dir="$1"
    ls -d "$dir"/episode_* 2>/dev/null | sort -t_ -k2 -n | tail -1
}

echo "=========================================="
echo "FOSP Experiment: $RUN_NAME (GPU $GPU)"
echo "=========================================="

# --- Phase 1: r0 learns vs random r1 (300k episodes) ---
P1_NAME="${RUN_NAME}_p1_r0vsRandom"
echo ""
echo "=== Phase 1: r0 learns vs random r1 (300k) ==="
echo "Run name: $P1_NAME"
python train_dqn_vec.py $COMMON_ARGS \
    --random-robots 1 \
    --num-episodes 300000 \
    --wandb-run-name "$P1_NAME"

P1_CKPT=$(find_checkpoint "$BASE_DIR/$P1_NAME")
echo "Phase 1 done. Checkpoint: $P1_CKPT"

# --- Phase 2: r1 learns vs frozen r0 (300k episodes) ---
P2_NAME="${RUN_NAME}_p2_r1vsFrozenR0"
echo ""
echo "=== Phase 2: r1 learns vs frozen r0 (300k) ==="
echo "Loading r0 from: $P1_CKPT"
echo "Run name: $P2_NAME"
python train_dqn_vec.py $COMMON_ARGS \
    --frozen-robots 0 \
    --load-model-dir "$P1_CKPT" \
    --num-episodes 300000 \
    --wandb-run-name "$P2_NAME"

P2_CKPT=$(find_checkpoint "$BASE_DIR/$P2_NAME")
echo "Phase 2 done. Checkpoint: $P2_CKPT"

# --- Phase 3: r0 learns vs frozen r1 (300k episodes) ---
P3_NAME="${RUN_NAME}_p3_r0vsFrozenR1"
echo ""
echo "=== Phase 3: r0 learns vs frozen r1 (300k) ==="
echo "Loading both from: $P2_CKPT"
echo "Run name: $P3_NAME"
python train_dqn_vec.py $COMMON_ARGS \
    --frozen-robots 1 \
    --load-model-dir "$P2_CKPT" \
    --num-episodes 300000 \
    --wandb-run-name "$P3_NAME"

P3_CKPT=$(find_checkpoint "$BASE_DIR/$P3_NAME")
echo "Phase 3 done. Checkpoint: $P3_CKPT"

# --- Phase 4: both learn (200k episodes, fine-tune) ---
P4_NAME="${RUN_NAME}_p4_bothLearn"
echo ""
echo "=== Phase 4: both learn together (200k) ==="
echo "Loading both from: $P3_CKPT"
echo "Run name: $P4_NAME"
python train_dqn_vec.py $COMMON_ARGS \
    --num-episodes 200000 \
    --load-model-dir "$P3_CKPT" \
    --wandb-run-name "$P4_NAME"

P4_CKPT=$(find_checkpoint "$BASE_DIR/$P4_NAME")
echo ""
echo "=========================================="
echo "FOSP complete!"
echo "Final checkpoint: $P4_CKPT"
echo "=========================================="
echo ""
echo "Evaluate with:"
echo "  python evaluate_models.py --model-dir $P4_CKPT --env-n 5 --charger-positions 2,2 --charger-range 1 --no-dust --e-charge 8 --e-collision 100 --e-decay 5 --max-steps 200 --eval-epsilon 0 --random-start-robots 0,1"
