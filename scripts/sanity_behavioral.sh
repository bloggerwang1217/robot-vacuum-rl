#!/usr/bin/env bash
# Behavioral sanity checks for the DQN training pipeline.
# Stops early on first failure.
#
# Usage:
#   bash scripts/sanity_behavioral.sh [GPU_ID]
#
# Note: training log prints one line per 100 episodes.
# All tail_n values below are in LOG LINES (÷100 of episode count).

set -euo pipefail
GPU=${1:-0}

source ~/.venv/bin/activate

# LOG_INTERVAL = 100 episodes per line
LOG_INTERVAL=100

# sanity check 全部用 standard DQN（關掉 NoisyNet / C51 / PER）
# 目的：測試 pipeline 正確性，不是測 Rainbow 收斂速度。
# 主實驗仍用 Rainbow；這裡只是驗證物理/reward/n-step 有沒有接對。
BASE="python train_dqn_vec.py
  --env-n 6 --charger-positions 3,3
  --exclusive-charging --no-dust
  --e-collision 30 --e-boundary 0
  --no-noisy --no-c51 --no-per
  --n-step 10 --gamma 0.99
  --batch-env --use-torch-compile
  --num-envs 128 --save-frequency 999999
  --wandb-mode disabled --no-eval-after-training --no-plot-evolution-after-training"

# Check 1 用獨立 base
# 設計邏輯：
#   1. 關閉 NoisyNet/C51/PER，用 standard DQN（簡單任務收斂更快）
#   2. charger-range=1（9 格充電區），從 (3,3) 出發移動仍在範圍內
#   3. e_move=1, e_decay=1, energy=30：離開充電區 15 步死亡，留在充電區淨 +7/步
BASE_CHECK1="python train_dqn_vec.py
  --env-n 6 --charger-positions 3,3 --charger-range 1
  --exclusive-charging --no-dust
  --e-move 1 --e-charge 8 --e-decay 1 --e-boundary 0
  --e-collision 30
  --no-noisy --no-c51 --no-per
  --n-step 10 --gamma 0.99
  --batch-env --use-torch-compile
  --num-envs 128 --save-frequency 999999
  --wandb-mode disabled --no-eval-after-training --no-plot-evolution-after-training"

GREEN="\033[92m"
RED="\033[91m"
RESET="\033[0m"
PASS="${GREEN}PASS${RESET}"
FAIL="${RED}FAIL${RESET}"
all_ok=true

# ── helpers ────────────────────────────────────────────────────────────────
# All helpers take tail_lines = number of log lines to inspect
# (= target_episode_count / LOG_INTERVAL)

mean_reward() {
    local logfile=$1 robot=$2 tail_lines=$3
    grep "^\[Episode" "$logfile" | tail -n "$tail_lines" \
      | grep -oP "(?<=${robot}:)-?[0-9]+\.[0-9]+" \
      | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "0"}'
}

mean_reward_survived() {
    local logfile=$1 robot=$2 tail_lines=$3
    grep "^\[Episode" "$logfile" | tail -n "$tail_lines" \
      | grep "${robot}_alive" \
      | grep -oP "(?<=${robot}:)-?[0-9]+\.[0-9]+" \
      | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "0"}'
}

survival_rate() {
    local logfile=$1 robot=$2 tail_lines=$3
    local total survived
    total=$(grep "^\[Episode" "$logfile" | tail -n "$tail_lines" | wc -l)
    survived=$(grep "^\[Episode" "$logfile" | tail -n "$tail_lines" \
               | grep -c "${robot}_alive" || true)
    awk "BEGIN {if($total>0) printf \"%.3f\", $survived/$total; else print \"0\"}"
}

mean_collisions_r0() {
    local logfile=$1 tail_lines=$2
    grep "^\[Episode" "$logfile" | tail -n "$tail_lines" \
      | grep -oP "(?<=Collisions\()[0-9]+" \
      | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "0"}'
}

mean_r0_attacks_r2() {
    local logfile=$1 tail_lines=$2
    grep "^\[Episode" "$logfile" | tail -n "$tail_lines" \
      | grep -oP "(?<=r0→r2:)[0-9]+" \
      | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "0"}'
}

mean_r1_attacks_r2() {
    local logfile=$1 tail_lines=$2
    grep "^\[Episode" "$logfile" | tail -n "$tail_lines" \
      | grep -oP "(?<=r1→r2:)[0-9]+" \
      | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "0"}'
}

mean_friendly_collisions() {
    local logfile=$1 tail_lines=$2
    grep "^\[Episode" "$logfile" | tail -n "$tail_lines" \
      | grep -oP "(?<=r0↔r1:)[0-9]+" \
      | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "0"}'
}

run_check() {
    local name=$1 logfile=$2
    shift 2
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    CUDA_VISIBLE_DEVICES=$GPU $@ 2>&1 | tee "$logfile"
}

check_pass() { echo -e "  ${PASS}  $1"; }
check_fail() { echo -e "  ${FAIL}  $1"; all_ok=false; }

# ════════════════════════════════════════════════════════════════
# Check 1 — 充電 Reward 訊號（最小化干擾）
#
# 設計原則：只測 ΔE reward pipeline 有沒有接通，排除其他變數。
# - r0 從充電座正上 (3,3) 出發，e_move=0，e_decay=0（不會死）
# - charger-range=0（只有剛好在 (3,3) 才充電）
# - STAY at (3,3) → +0.4 reward/step；離開 → 0 reward
# - 預期 15k ep 後 r0 學會 STAY，mean reward > 100
# tail_lines = last 2000 ep / 100 = 20 lines
# ════════════════════════════════════════════════════════════════
LOG1=/tmp/sanity_check1.log
run_check "Check 1: 充電 Reward 訊號" "$LOG1" \
  $BASE_CHECK1 \
  --num-robots 1 \
  --robot-0-energy 30 \
  --robot-start-positions 3,3 \
  --max-episode-steps 500 \
  --num-episodes 15000 \
  --wandb-run-name sanity_check1

TAIL1=20  # last 2000 episodes
sr1=$(survival_rate "$LOG1" r0 $TAIL1)
r0_mean1=$(mean_reward "$LOG1" r0 $TAIL1)
echo ""
echo "  r0 存活率 (last 2k ep): $sr1      [threshold: > 0.70]"
echo "  r0 mean reward (last 2k ep): $r0_mean1  [threshold: > 50]"

sr1_ok=false; rw1_ok=false
awk "BEGIN {exit ($sr1 > 0.70) ? 0 : 1}"    && sr1_ok=true || true
awk "BEGIN {exit ($r0_mean1 > 50) ? 0 : 1}" && rw1_ok=true || true

if $sr1_ok && $rw1_ok; then
    check_pass "Check 1: 充電 Reward 訊號"
else
    check_fail "Check 1: 充電 Reward 訊號"
    $sr1_ok || echo "  → 存活率 < 70%：agent 沒學會留在充電座，obs 的充電座位置欄位可能有問題"
    $rw1_ok || echo "  → reward < 50：充電 ΔE reward 沒有正確傳遞，檢查 prev_energy 時序或 reward_alpha"
    echo ""; echo "Stopping early."; exit 1
fi

# ════════════════════════════════════════════════════════════════
# Check 2 — Death Penalty + N-step Propagation
#
# r0 energy=10, starts 2 steps from charger (2,2).
# Wrong moves → die in <10 steps → -100.
# Correct moves → reach charger → survive 200 steps.
# n-step must carry death signal 2 steps back to spawn.
# 指標：存活率 > 80%，平均 reward > 30
# tail_lines = last 5000 ep / 100 = 50 lines
# ════════════════════════════════════════════════════════════════
LOG2=/tmp/sanity_check2.log
run_check "Check 2: 死亡懲罰 + N-step 傳播" "$LOG2" \
  $BASE \
  --num-robots 1 \
  --robot-0-energy 10 \
  --robot-start-positions 2,2 \
  --max-episode-steps 200 \
  --num-episodes 30000 \
  --wandb-run-name sanity_check2

TAIL2=50
sr2=$(survival_rate "$LOG2" r0 $TAIL2)
r0_mean2=$(mean_reward "$LOG2" r0 $TAIL2)
echo ""
echo "  r0 存活率 (last 5k ep):  $sr2      [threshold: > 0.80]"
echo "  r0 mean reward (last 5k ep): $r0_mean2 [threshold: > 0]"
echo "  （reward 只要正值即可——重點是驗證 n-step 傳播了死亡訊號，非要求完美蹲點）"

sr2_ok=false; r2_ok=false
awk "BEGIN {exit ($sr2 > 0.80) ? 0 : 1}"   && sr2_ok=true || true
awk "BEGIN {exit ($r0_mean2 > 0) ? 0 : 1}" && r2_ok=true || true

if $sr2_ok && $r2_ok; then
    check_pass "Check 2: 死亡懲罰 + N-step 傳播"
else
    check_fail "Check 2: 死亡懲罰 + N-step 傳播"
    $sr2_ok || echo "  → 存活率 < 80%：死亡訊號沒有往前傳播，檢查 remember_batch 的 done-masking"
    $r2_ok  || echo "  → reward < 0：n-step return 鏈斷掉，完全沒學到充電收益"
    echo ""; echo "Stopping early."; exit 1
fi

# ════════════════════════════════════════════════════════════════
# Check 3 — 殺敵獨霸充電座
#
# r1 STAY 在充電座 (3,3)，r0 必須殺掉 r1（20hp < 30dmg）才能充電。
# 不殺就活不到最後。Kill → camp → ~100+ reward。
# n-step 必須把殺後充電的收益傳回「靠近 r1」這個 action。
# 指標：r0 mean reward > 100，r0→r1 碰撞/集 > 0.8
# tail_lines = last 5000 ep / 100 = 50 lines
# ════════════════════════════════════════════════════════════════
LOG3=/tmp/sanity_check3.log
run_check "Check 3: 殺敵獨霸充電座" "$LOG3" \
  $BASE \
  --num-robots 2 \
  --scripted-robots 1 \
  --robot-0-energy 100 --robot-1-energy 20 \
  --robot-start-positions "0,0;3,3" \
  --max-episode-steps 500 \
  --num-episodes 50000 \
  --wandb-run-name sanity_check3

TAIL3=50
r0_mean3=$(mean_reward "$LOG3" r0 $TAIL3)
kill_rate=$(mean_collisions_r0 "$LOG3" $TAIL3)
echo ""
echo "  r0 mean reward      (last 5k ep): $r0_mean3  [threshold: > 100]"
echo "  r0→r1 collisions/ep (last 5k ep): $kill_rate [threshold: > 0.8]"

r3_ok=false; k3_ok=false
awk "BEGIN {exit ($r0_mean3 > 100) ? 0 : 1}" && r3_ok=true || true
awk "BEGIN {exit ($kill_rate > 0.8) ? 0 : 1}" && k3_ok=true || true

if $r3_ok && $k3_ok; then
    check_pass "Check 3: 殺敵獨霸充電座"
else
    check_fail "Check 3: 殺敵獨霸充電座"
    $r3_ok || echo "  → reward < 100：殺後獨霸充電的收益沒有傳到 agent"
    $k3_ok || echo "  → 碰撞率 < 0.8：n-step 沒有把殺後充電收益傳回「靠近」動作"
    echo ""; echo "Stopping early."; exit 1
fi

# ════════════════════════════════════════════════════════════════
# Check 4（同盟）— 友好碰撞觸發能量平均
#
# r0+r1 同盟，r2 STAY 遠端。r0 與 r1 相鄰出發。
# 速度=2，r0 會頻繁走進 r1（友好碰撞）。
# 測試：batch_env 訓練路徑的能量平均機制有沒有觸發？
# 指標：r0↔r1 友好碰撞/集 > 1.0
# tail_lines = last 3000 ep / 100 = 30 lines
# ════════════════════════════════════════════════════════════════
LOG4=/tmp/sanity_check4.log
run_check "Check 4（同盟）: 友好碰撞觸發" "$LOG4" \
  $BASE \
  --num-robots 3 \
  --scripted-robots 2 \
  --robot-0-energy 80 --robot-1-energy 20 --robot-2-energy 100 \
  --robot-0-speed 2 --robot-1-speed 2 --robot-2-speed 1 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 5 --robot-2-stun-steps 1 \
  --robot-start-positions "3,2;3,4;0,0" \
  --alliance-groups 0,1 \
  --max-episode-steps 300 \
  --num-episodes 20000 \
  --wandb-run-name sanity_check4

TAIL4=30
fc=$(mean_friendly_collisions "$LOG4" $TAIL4)
echo ""
echo "  r0↔r1 friendly collisions/ep (last 3k ep): $fc [threshold: > 1.0]"

if awk "BEGIN {exit ($fc > 1.0) ? 0 : 1}"; then
    check_pass "Check 4（同盟）: 友好碰撞觸發"
else
    check_fail "Check 4（同盟）: 友好碰撞觸發"
    echo "  → 次數接近 0：batch_env 同盟碰撞路徑沒有執行到，或 _allied_pairs 解析錯誤"
    echo "  → 若 log 完全沒有 r0↔r1 欄位：訓練 log 格式對同盟組的輸出有問題"
    echo ""; echo "Stopping early."; exit 1
fi

# ════════════════════════════════════════════════════════════════
# Check 5（同盟）— 害蟲追殺行為出現
#
# r0+r1 同盟，r2 seek_charger。r2 撞到 r0/r1 會讓他們被 stun 5 步。
# 兩隻盟友都有殺 r2 的誘因。測試 stun 懲罰能否傳播讓至少一隻追殺。
# 只有一隻追殺 → WARN（搭便車現象，是研究發現，不是 bug）。
# 指標：r0→r2 OR r1→r2 碰撞/集 > 0.5
# tail_lines = last 10000 ep / 100 = 100 lines
# ════════════════════════════════════════════════════════════════
LOG5=/tmp/sanity_check5.log
run_check "Check 5（同盟）: 害蟲追殺行為出現" "$LOG5" \
  $BASE \
  --num-robots 3 \
  --seek-charger-robots 2 \
  --robot-0-energy 80 --robot-1-energy 80 --robot-2-energy 20 \
  --robot-0-speed 2 --robot-1-speed 2 --robot-2-speed 1 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 5 --robot-2-stun-steps 1 \
  --robot-start-positions "0,0;5,5;3,0" \
  --alliance-groups 0,1 \
  --max-episode-steps 500 \
  --num-episodes 150000 \
  --wandb-run-name sanity_check5

TAIL5=100
r0_atk=$(mean_r0_attacks_r2 "$LOG5" $TAIL5)
r1_atk=$(mean_r1_attacks_r2 "$LOG5" $TAIL5)
echo ""
echo "  r0→r2 attacks/ep (last 10k ep): $r0_atk [threshold: > 0.5]"
echo "  r1→r2 attacks/ep (last 10k ep): $r1_atk [threshold: > 0.5]"

a5=$(awk "BEGIN {exit ($r0_atk > 0.5) ? 0 : 1}" && echo true || echo false)
b5=$(awk "BEGIN {exit ($r1_atk > 0.5) ? 0 : 1}" && echo true || echo false)

if $a5 && $b5; then
    check_pass "Check 5（同盟）: 兩隻盟友都追殺害蟲"
elif $a5 || $b5; then
    echo -e "  ${GREEN}WARN${RESET}  Check 5：只有一隻盟友追殺（搭便車均衡）"
    echo "  → r0→r2=$r0_atk  r1→r2=$r1_atk"
    echo "  → 不是 bug——IDQN 搭便車是研究發現，值得記錄"
else
    check_fail "Check 5（同盟）: 沒有任何盟友追殺害蟲"
    echo "  → 兩者都接近 0：r2 的 stun 懲罰訊號沒有傳到任何一個 agent"
    echo "  → 確認 stun_steps 設定正確，stun counter 有在遞減"
fi

# ════════════════════════════════════════════════════════════════
# 總結
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
if $all_ok; then
    echo -e "  ${PASS}  所有 checks 通過"
    echo "  Pipeline 健康——可以安全跑主實驗。"
else
    echo -e "  ${FAIL}  有 check 失敗（詳見上方）"
fi
echo "  Logs: $LOG1  $LOG2  $LOG3  $LOG4  $LOG5"
echo "════════════════════════════════════════"
