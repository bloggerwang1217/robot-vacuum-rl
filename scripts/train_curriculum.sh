#!/bin/bash
# Curriculum training: 4 phases for robot_1
#   Phase 1: robot_1 STAY     (robot_0 learns to hunt a stationary target)
#   Phase 2: robot_1 RANDOM   (robot_0 learns to chase a moving target)
#   Phase 3: robot_1 FLEE     (robot_0 learns to catch an escaping target)
#   Phase 4: both learn       (full IDQN from a strong initialization)
#
# Usage:
#   bash scripts/train_curriculum.sh [RUN_NAME]
#
# Examples:
#   bash scripts/train_curriculum.sh                     # auto-named
#   bash scripts/train_curriculum.sh my_curriculum       # custom name

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

# ── Base run name ──────────────────────────────────────────────────────────────
BASE_NAME="${1:-curriculum_$(date +%Y%m%d_%H%M%S)}"

# ── Env params (edit these to match your experiment) ──────────────────────────
ENV_N=5
CHARGER="2,2"
R0_ENERGY=100
R1_ENERGY=100
E_MOVE=1
E_CHARGE=10
E_COLLISION=30
E_BOUNDARY=5       # low boundary penalty so robot_0 can explore safely
N_STEP=20
GAMMA=0.999
MAX_STEPS=1000
NUM_ENVS=32
SAVE_FREQ=1000
WANDB_MODE=disabled

# ── Episodes per phase (total = sum) ──────────────────────────────────────────
P1_EP=3000   # robot_1 STAY
P2_EP=3000   # robot_1 RANDOM
P3_EP=2000   # robot_1 FLEE
P4_EP=5000   # both learn

# ── Shared training flags ─────────────────────────────────────────────────────
COMMON="--env-n $ENV_N \
  --num-robots 2 \
  --robot-0-energy $R0_ENERGY \
  --robot-1-energy $R1_ENERGY \
  --charger-positions $CHARGER \
  --exclusive-charging \
  --no-dust \
  --e-move $E_MOVE \
  --e-charge $E_CHARGE \
  --e-collision $E_COLLISION \
  --e-boundary $E_BOUNDARY \
  --n-step $N_STEP \
  --gamma $GAMMA \
  --max-episode-steps $MAX_STEPS \
  --num-envs $NUM_ENVS \
  --save-frequency $SAVE_FREQ \
  --wandb-mode $WANDB_MODE \
  --no-eval-after-training"

# Helper: find highest episode checkpoint in a model save dir
find_checkpoint() {
    local dir="$1"
    local ep=$(ls "$dir" | grep '^episode_' | sed 's/episode_//' | sort -n | tail -1)
    echo "$dir/episode_$ep"
}

echo "======================================================"
echo "  Curriculum Training: $BASE_NAME"
echo "======================================================"
echo "  Env: ${ENV_N}x${ENV_N}  Charger: ($CHARGER)  exclusive_charging=True"
echo "  R0 energy=$R0_ENERGY  R1 energy=$R1_ENERGY"
echo "  e_collision=$E_COLLISION  e_boundary=$E_BOUNDARY  e_charge=$E_CHARGE"
echo "  Phase episodes: P1=$P1_EP  P2=$P2_EP  P3=$P3_EP  P4=$P4_EP"
echo "  Total: $((P1_EP + P2_EP + P3_EP + P4_EP)) episodes"
echo "======================================================"
echo ""

# ── Phase 1: robot_1 STAY ──────────────────────────────────────────────────────
P1_NAME="${BASE_NAME}_p1_stay"
echo "[Phase 1/$P1_EP ep] robot_1 STAY  →  $P1_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P1_EP \
  --scripted-robots 1 \
  --wandb-run-name "$P1_NAME" \
  --save-dir "./models/$P1_NAME"
P1_CKPT=$(find_checkpoint "./models/$P1_NAME")
echo "  checkpoint: $P1_CKPT"
echo ""

# ── Phase 2: robot_1 RANDOM ───────────────────────────────────────────────────
P2_NAME="${BASE_NAME}_p2_random"
echo "[Phase 2/$P2_EP ep] robot_1 RANDOM  →  $P2_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P2_EP \
  --random-robots 1 \
  --load-model-dir "$P1_CKPT" \
  --wandb-run-name "$P2_NAME" \
  --save-dir "./models/$P2_NAME"
P2_CKPT=$(find_checkpoint "./models/$P2_NAME")
echo "  checkpoint: $P2_CKPT"
echo ""

# ── Phase 3: robot_1 FLEE ─────────────────────────────────────────────────────
P3_NAME="${BASE_NAME}_p3_flee"
echo "[Phase 3/$P3_EP ep] robot_1 FLEE  →  $P3_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P3_EP \
  --flee-robots 1 \
  --load-model-dir "$P2_CKPT" \
  --wandb-run-name "$P3_NAME" \
  --save-dir "./models/$P3_NAME"
P3_CKPT=$(find_checkpoint "./models/$P3_NAME")
echo "  checkpoint: $P3_CKPT"
echo ""

# ── Phase 4: both learn ───────────────────────────────────────────────────────
P4_NAME="${BASE_NAME}_p4_full"
echo "[Phase 4/$P4_EP ep] both learn  →  $P4_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P4_EP \
  --load-model-dir "$P3_CKPT" \
  --wandb-run-name "$P4_NAME" \
  --save-dir "./models/$P4_NAME" \
  --eval-after-training    # 最後一個 phase 才做 eval

echo ""
echo "======================================================"
echo "  Curriculum done: $BASE_NAME"
echo "  Final model: ./models/$P4_NAME"
echo "======================================================"
