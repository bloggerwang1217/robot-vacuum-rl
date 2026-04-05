#!/bin/bash
# Pest Curriculum: 讓 r0 先學會充電 → r1 開始學（發現可以撞 r0）→ 雙方共同演化
# 目標：捕捉 r1 從「衝充電座」→「開始撤退」過程中 r0 追殺的短暫窗口
#
# P1: r0 alone (r1 scripted STAY), NO docking — r0 學會充電（無 docking 限制）
# P2: both learn (r1 從零開始), WITH docking — r1 發現充電座、學會撞 r0、r0 反擊、r1 開始退
#
# 配置：pest design (r0 atk=30, r1 atk=2), docking=2 for r0, stun=2 for r0
#        charger_range=0, exclusive_charging, no dust
#
# Usage: bash scripts/train_pest_curriculum.sh [RUN_NAME] [GPU]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

BASE_NAME="${1:-pest_curriculum_$(date +%Y%m%d_%H%M%S)}"
GPU="${2:-0}"

# ── Env params ───────────────────────────────────────────────────────────────
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

# ── Episodes per phase ──────────────────────────────────────────────────────
P1_EP=200000     # r0 learns charger (~25k to converge, 200k to solidify)
P2_EP=3000000    # both learn — 密集存 checkpoint 抓過渡期

# ── Save frequency ──────────────────────────────────────────────────────────
P1_SAVE=50000    # P1 粗粒度就好
P2_SAVE=10000    # P2 密集存，抓那個窗口

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
echo "  Pest Curriculum: $BASE_NAME"
echo "======================================================"
echo "  Grid: ${ENV_N}x${ENV_N}  Charger: ($CHARGER) range=$CHARGER_RANGE"
echo "  r0: E=$R0_ENERGY spd=$R0_SPEED atk=$R0_ATK dock=$R0_DOCK stun=$R0_STUN"
echo "  r1: E=$R1_ENERGY spd=$R1_SPEED atk=$R1_ATK dock=$R1_DOCK stun=$R1_STUN"
echo "  P1=$P1_EP ep (r0 alone)  P2=$P2_EP ep (both learn)"
echo "  GPU=$GPU"
echo "======================================================"
echo ""

# ── Phase 1: r0 learns charger (r1 scripted STAY) ───────────────────────────
P1_NAME="${BASE_NAME}_p1"
echo "[Phase 1 / $P1_EP ep] r0 learns to charge (r1 STAY)  →  $P1_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P1_EP \
  --scripted-robots 1 \
  --save-frequency $P1_SAVE \
  --wandb-run-name "$P1_NAME" \
  --save-dir "./models/$P1_NAME"
P1_CKPT=$(find_checkpoint "./models/$P1_NAME")
echo "  P1 checkpoint: $P1_CKPT"
echo ""

# ── Phase 2: both learn (r1 starts from scratch) ────────────────────────────
P2_NAME="${BASE_NAME}_p2"
echo "[Phase 2 / $P2_EP ep] both learn (r1 from zero)  →  $P2_NAME"
python3 "$SCRIPT_DIR/train_dqn_vec.py" \
  $COMMON \
  --num-episodes $P2_EP \
  --load-model-dir "$P1_CKPT" \
  --save-frequency $P2_SAVE \
  --wandb-run-name "$P2_NAME" \
  --save-dir "./models/$P2_NAME"

echo ""
echo "======================================================"
echo "  Curriculum done: $BASE_NAME"
echo "  Final model: ./models/$P2_NAME"
echo "  Run eval_evolution.py on ./models/$P2_NAME to plot"
echo "======================================================"
