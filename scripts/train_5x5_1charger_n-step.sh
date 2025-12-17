#!/bin/bash

# 機器人吸塵器 RL 批次訓練腳本
# 虐殺模式：高血量機器人 vs 低血量機器人

set -e

# 參數解析：第一個參數為 episodes 數量，預設 5000
NUM_EPISODES=${1:-5000}

# 根據 episodes 數量動態設定 epsilon decay
if [ $NUM_EPISODES -ge 5000 ]; then
  EPSILON_DECAY=0.998  # 5000 episodes: 在 ~2000 ep 降到 0.01
else
  EPSILON_DECAY=0.999  # 預設: 在 ~900 ep 降到 0.01
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
MODELS_DIR="${SCRIPT_DIR}/models/massacre_5x5_center_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "機器人吸塵器強化學習 - 虐殺模式訓練 (5x5 + 中心充電器)"
echo "=========================================="
echo "開始時間: $(date)"
echo "模型儲存目錄: $MODELS_DIR"
echo "地圖配置: 5x5，充電器位置: (2,2)（中心）"
echo "訓練 Episodes: $NUM_EPISODES"
echo "=========================================="
echo ""

# 定義 n-step sweep 參數
declare -a n_steps=(2 3 4 5)

# 定義訓練配置
# 格式: "配置名稱|訓練參數"
declare -a configs=(
  "massacre-1v3-5x5-center|--env-n 5 --charger-positions 2,2 --robot-0-energy 1000 --robot-1-energy 100 --robot-2-energy 100 --robot-3-energy 100 --e-collision 20 --e-charge 20 --use-epsilon-decay --epsilon-decay 0.998"
)

total_configs=$((${#configs[@]} * ${#n_steps[@]}))
echo "共有 $total_configs 個訓練配置 - 虐殺模式"
echo ""
echo "地圖設置："
echo "  - 大小: 5×5"
echo "  - 充電器: 中心位置 (2,2)"
echo ""
echo "能量機制："
echo "  - 生存成本: 每回合所有活著的機器人扣 e_move"
echo "  - 碰撞傷害: 被推人或互撞時額外扣 e_collision"
echo "  - 充電獎勵: +e_charge"
echo ""
echo "碰撞機制："
echo "  - 單方面推人: 攻擊者無額外傷害，被推方受 e_collision 傷害"
echo "  - 互撞 (swap/contested): 雙方都停留且受 e_collision 傷害"
echo "  - 撞牆: 停留且受 e_collision 傷害"
echo ""
echo "訓練參數："
echo "  - e-move: 1 (生存成本)"
echo "  - e-collision: 20 (推人傷害，降低敵人進行風險)"
echo "  - e-charge: 20 (充電獎勵，提升敵人去充電座的動機)"
echo "  - 機制: 先充電後扣生存成本（強化充電座吸引力）"
echo "  - batch-size: 128"
echo "  - epsilon decay: 0.998 (較慢的探索衰減，在 ep~2000 時降到 0.01)"
echo ""
echo "血量配置 (虐殺模式)："
echo "  [1v3] Robot 0: 1000 | Robots 1,2,3: 100 (一個強者 vs 三個弱者)"
echo ""
echo "N-step Sweep 配置："
echo "  - 測試 n-step: ${n_steps[@]}"
echo ""
echo "理論預測："
echo "  - 敵人傷害低 (20)，充電獎勵高 (20) → 敵人會主動競爭充電座"
echo "  - 期望競爭效率 α: 40-50%（從 94% 習得無助 → 真實競爭）"
echo "  - 臨界值 α_critical: 74%"
echo "  - 由於 α < α_critical，殺人策略應成為最優 ✓"
echo "  - R0 應學會主動消滅敵人以獲得獨佔充電座 ✓"
echo ""

# 遍歷每個配置和每個 n-step 值並執行訓練
total_counter=0
for ((i=0; i<${#configs[@]}; i++)); do
  config="${configs[$i]}"
  config_name="${config%|*}"
  config_params="${config#*|}"

  for n in "${n_steps[@]}"; do
    total_counter=$((total_counter + 1))

    # 生成此次訓練的名稱
    run_name="${config_name}-nstep${n}"

    echo "=========================================="
    echo "[$total_counter/$total_configs] 執行訓練: $run_name"
    echo "參數: $config_params --n-step $n"
    echo "開始時間: $(date)"
    echo "=========================================="

    # 建立此次訓練的模型目錄
    run_dir="$MODELS_DIR/$run_name"
    mkdir -p "$run_dir"

    # 執行訓練
    python3 "$SCRIPT_DIR/train_dqn.py" \
      $config_params \
      --num-episodes $NUM_EPISODES \
      --n-step $n \
      --max-episode-steps 1000 \
      --save-dir "$run_dir" \
      --save-frequency 500 \
      --wandb-entity lazyhao-national-taiwan-university \
      --wandb-project robot-vacuum-rl \
      --wandb-run-name "$run_name-ep$NUM_EPISODES" \
      --wandb-mode online \
      2>&1 | tee "$run_dir/training.log"

    echo ""
    echo "✓ 訓練完成: $run_name"
    echo "結果已保存到: $run_dir"
    echo "結束時間: $(date)"
    echo ""

    # 訓練間隔 - 讓系統稍作休息
    if [ $total_counter -lt $total_configs ]; then
      echo "等待 5 秒後開始下一個訓練..."
      sleep 5
    fi
  done
done

echo "=========================================="
echo "所有虐殺模式訓練完成！"
echo "結束時間: $(date)"
echo "所有結果已保存到: $MODELS_DIR"
echo "=========================================="

# 列出所有生成的結果
echo ""
echo "訓練結果概覽："
ls -lh "$MODELS_DIR"
