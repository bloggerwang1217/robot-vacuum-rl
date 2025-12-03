#!/bin/bash

# 機器人吸塵器 RL 批次訓練腳本
# 使用 imbalanced collision 參數訓練

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
MODELS_DIR="${SCRIPT_DIR}/models/1charger-imbalanced_collision_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "機器人吸塵器強化學習 - Imbalanced Collision 訓練"
echo "=========================================="
echo "開始時間: $(date)"
echo "模型儲存目錄: $MODELS_DIR"
echo "=========================================="
echo ""

# 定義訓練配置
declare -a configs=(
  "1charger-collision-50-energy-100-balanced|--e-collision 50 --initial-energy 100 --use-epsilon-decay"
  "1charger-collision-50-energy-100-imbalanced|--e-collision-active-one-sided 25 --e-collision-active-two-sided 50 --e-collision-passive 50 --initial-energy 100 --use-epsilon-decay"
)

total_configs=${#configs[@]}
echo "共有 $total_configs 個訓練配置（使用 imbalanced collision）"
echo ""

# 遍歷每個配置並執行訓練
for ((i=0; i<${#configs[@]}; i++)); do
  config="${configs[$i]}"
  config_name="${config%|*}"
  config_params="${config#*|}"

  # 計算進度
  current=$((i+1))

  echo "=========================================="
  echo "[$current/$total_configs] 執行訓練: $config_name"
  echo "參數: $config_params"
  echo "開始時間: $(date)"
  echo "=========================================="

  # 建立此次訓練的模型目錄
  run_dir="$MODELS_DIR/$config_name"
  mkdir -p "$run_dir"

  # 執行訓練
  python3 "$SCRIPT_DIR/train_dqn.py" \
    $config_params \
    --charger-positions "1,1" \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university \
    --wandb-project robot-vacuum-rl \
    --wandb-run-name "$config_name" \
    --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"

  echo ""
  echo "✓ 訓練完成: $config_name"
  echo "結果已保存到: $run_dir"
  echo "結束時間: $(date)"
  echo ""

  # 訓練間隔 - 讓系統稍作休息
  if [ $current -lt $total_configs ]; then
    echo "等待 5 秒後開始下一個訓練..."
    sleep 5
  fi
done

echo "=========================================="
echo "所有訓練完成！"
echo "結束時間: $(date)"
echo "所有結果已保存到: $MODELS_DIR"
echo "=========================================="

# 列出所有生成的結果
echo ""
echo "訓練結果概覽："
ls -lh "$MODELS_DIR"
