#!/bin/bash

# 機器人吸塵器 RL 批次訓練腳本
# 虐殺模式：高血量機器人 vs 低血量機器人

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
MODELS_DIR="${SCRIPT_DIR}/models/massacre_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "機器人吸塵器強化學習 - 虐殺模式訓練 (Default 4 chargers)"
echo "=========================================="
echo "開始時間: $(date)"
echo "模型儲存目錄: $MODELS_DIR"
echo "=========================================="
echo ""

# 定義訓練配置
# 格式: "配置名稱|訓練參數"
declare -a configs=(
  "massacre-1v3|--robot-0-energy 1000 --robot-1-energy 20 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-2v2|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-3v1|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
)

total_configs=${#configs[@]}
echo "共有 $total_configs 個訓練配置 - 虐殺模式"
echo "固定參數："
echo "  - e-collision: 10"
echo "  - e-move: 1 (預設)"
echo "  - batch-size: 128 (加速訓練)"
echo "  - 使用 epsilon decay"
echo ""
echo "血量配置 (虐殺模式)："
echo "  1. [1v3] Robot 0: 1000 | Robots 1,2,3: 20 (一個強者 vs 三個弱者)"
echo "  2. [2v2] Robots 0,1: 1000 | Robots 2,3: 20 (兩強 vs 兩弱)"
echo "  3. [3v1] Robots 0,1,2: 1000 | Robot 3: 20 (三強圍殺一個弱者)"
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
echo "所有虐殺模式訓練完成！"
echo "結束時間: $(date)"
echo "所有結果已保存到: $MODELS_DIR"
echo "=========================================="

# 列出所有生成的結果
echo ""
echo "訓練結果概覽："
ls -lh "$MODELS_DIR"
