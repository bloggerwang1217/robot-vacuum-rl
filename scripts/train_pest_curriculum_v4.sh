#!/bin/bash
# Pest Curriculum v4: 雙方先各自學充電 → 放在一起競爭
#
# P1: r0 單獨學充電 (r1 STAY) — 直接沿用 v3 的 checkpoint
# P2: r1 單獨學充電 (r0 STAY) — 新跑
# P3: 載入 P1的r0 + P2的r1，epsilon-start=0.3，兩邊一起學
#
# 目標：觀察 r1 從「搶充電座」→「被打退」→「開始退卻」過程中
#       r0 是否會離開充電座主動追殺 r1
#
# Usage: bash scripts/train_pest_curriculum_v4.sh [RUN_NAME] [GPU]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

BASE_NAME="${1:-pest_curri_v4_$(date +%Y%m%d_%H%M%S)}"
GPU="${2:-0}"

# ── Env params (same as v3) ──────────────────────────────────────────────────
ENV_N=5
CHARGER="2,2"
CHARGER_RANGE=0
R0_ENERGY=100
R1_ENERGY=100
R0_SPEED=2
R1_SPEED=1
R0_ATK=30
R1_ATK=2
R0_DOCK=2
R1_DOCK=0
R0_STUN=2
R1_STUN=0
E_MOVE=0
E_CHARGE=8
E_DECAY=0.5
E_COLLISION=30
E_BOUNDARY=0
N_STEP=20
GAMMA=0.99
MAX_STEPS=500
NUM_ENVS=256

# ── Episodes per phase ───────────────────────────────────────────────────────
P1_CKPT_DIR="./models/pest_curri_v3_p1/episode_200000"   # 沿用 v3
P2_EP=150000      # r1 學充電 (沒 docking，收斂更快)
P3_EP=3000000     # 雙方競爭

# ── Save frequency ───────────────────────────────────────────────────────────
P2_SAVE=50000
P3_SAVE=5000      # 密集存，抓追殺窗口

# ── Shared flags ─────────────────────────────────────────────────────────────
COMMON="--env-n $ENV_N \
  --num-robots 2 \
  --robot-0-energy $R0_ENERGY --robot-1-energy $R1_ENERGY \
  --robot-0-speed $R0_SPEED --robot-1-speed $R1_SPEED \
  --robot-0-attack-power $R0_ATK --robot-1-attack-power $R1_ATK \
  --robot-0-docking-steps $R0_DOCK --robot-1-docking-steps $R1_DOCK \
  --robot-0-stun-steps $R0_STUN --robot-1-stun-steps $R1_STUN \
  --charger-positions $CHARGER --charger-range $CHARGER_RANGE \
  --exclusive-charging --no-dust \
  --e-move $E_MOVE --e-charge $E_CHARGE --e-decay $E_DECAY \
  --e-collision $E_COLLISION --e-boundary $E_BOUNDARY \
  --n-step $N_STEP --gamma $GAMMA \
  --max-episode-steps $MAX_STEPS \
  --num-envs $NUM_ENVS \
  --batch-env --no-noisy \
  --no-dueling --no-c51 \
  --random-start-robots 0,1 \
  --shuffle-step-order \
  --gpu $GPU \
  --wandb-mode online \
  --no-eval-after-training"

find_checkpoint() {
    local dir="$1"
    local ep=$(ls "$dir" | grep '^episode_' | sed 's/episode_//' | sort -n | tail -1)
    echo "$dir/episode_$ep"
}

echo "======================================================"
echo "  Pest Curriculum v4: $BASE_NAME"
echo "======================================================"
echo "  Grid: ${ENV_N}x${ENV_N}  Charger: ($CHARGER) range=$CHARGER_RANGE"
echo "  r0: E=$R0_ENERGY spd=$R0_SPEED atk=$R0_ATK dock=$R0_DOCK stun=$R0_STUN"
echo "  r1: E=$R1_ENERGY spd=$R1_SPEED atk=$R1_ATK dock=$R1_DOCK stun=$R1_STUN"
echo "  P1: reuse $P1_CKPT_DIR"
echo "  P2: $P2_EP ep (r1 learns charger)"
echo "  P3: $P3_EP ep (both compete, ε-start=0.3)"
echo "  GPU=$GPU"
echo "======================================================"
echo ""

# ── Verify P1 checkpoint exists ─────────────────────────────────────────────
if [ ! -f "$P1_CKPT_DIR/robot_0.pt" ]; then
    echo "ERROR: P1 checkpoint not found: $P1_CKPT_DIR/robot_0.pt"
    exit 1
fi
echo "[Phase 1] Using existing r0 checkpoint: $P1_CKPT_DIR"
echo ""

# ── Phase 2: r1 learns charger alone (r0 scripted STAY) ─────────────────────
P2_NAME="${BASE_NAME}_p2"
echo "[Phase 2 / $P2_EP ep] r1 learns to charge (r0 STAY)  →  $P2_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P2_EP \
  --scripted-robots 0 \
  --save-frequency $P2_SAVE \
  --wandb-run-name "$P2_NAME" \
  --save-dir "./models/$P2_NAME"
P2_CKPT=$(find_checkpoint "./models/$P2_NAME")
echo "  P2 checkpoint: $P2_CKPT"
echo ""

# ── Merge checkpoints: P1/robot_0.pt + P2/robot_1.pt → combined dir ─────────
MERGED_DIR="./models/${BASE_NAME}_merged"
mkdir -p "$MERGED_DIR"
cp "$P1_CKPT_DIR/robot_0.pt" "$MERGED_DIR/robot_0.pt"
cp "$P2_CKPT/robot_1.pt"     "$MERGED_DIR/robot_1.pt"
echo "[Merge] Combined checkpoint: $MERGED_DIR"
echo "  robot_0.pt ← $P1_CKPT_DIR"
echo "  robot_1.pt ← $P2_CKPT"
echo ""

# ── Phase 3: both compete (loaded from merged checkpoint) ───────────────────
P3_NAME="${BASE_NAME}_p3"
echo "[Phase 3 / $P3_EP ep] both compete (ε-start=0.3)  →  $P3_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P3_EP \
  --load-model-dir "$MERGED_DIR" \
  --epsilon-start 0.3 \
  --save-frequency $P3_SAVE \
  --wandb-run-name "$P3_NAME" \
  --save-dir "./models/$P3_NAME"

echo ""
echo "======================================================"
echo "  Curriculum v4 done: $BASE_NAME"
echo "  Final model: ./models/$P3_NAME"
echo "  Run eval_evolution.py on ./models/$P3_NAME to plot"
echo "======================================================"
