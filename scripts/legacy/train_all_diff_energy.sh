#!/bin/bash

# 機器人吸塵器 RL 批次訓練腳本
# 整合所有血量配置實驗：漸進式不平衡（collision 10/50）+ 虐殺模式

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
MODELS_DIR="${SCRIPT_DIR}/models/all_energy_configs_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "機器人吸塵器強化學習 - 完整血量配置訓練"
echo "=========================================="
echo "開始時間: $(date)"
echo "模型儲存目錄: $MODELS_DIR"
echo "=========================================="
echo ""

# 定義訓練配置
# 格式: "配置名稱|訓練參數"
declare -a configs=(
  # 漸進式不平衡 (collision=10)
  "collision-10-robot0-1000-others-200|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  "collision-10-robot01-1000-others-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  "collision-10-robot012-1000-robot3-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  
  # 漸進式不平衡 (collision=50)
  "collision-50-robot0-1000-others-200|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  "collision-50-robot01-1000-others-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  "collision-50-robot012-1000-robot3-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  
  # 虐殺模式 (collision=10, 極端血量差距)
  "massacre-1v3|--robot-0-energy 1000 --robot-1-energy 20 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-2v2|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-3v1|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
)

total_configs=${#configs[@]}
echo "共有 $total_configs 個訓練配置"
echo ""
echo "固定參數："
echo "  - batch-size: 128 (加速訓練)"
echo "  - num-episodes: 2000"
echo "  - e-move: 1 (預設)"
echo "  - 使用 epsilon decay"
echo ""
echo "配置分類："
echo ""
echo "【漸進式不平衡 - collision=10】(configs 1-3)"
echo "  1. Robot 0: 1000 | Robots 1,2,3: 200"
echo "  2. Robots 0,1: 1000 | Robots 2,3: 200"
echo "  3. Robots 0,1,2: 1000 | Robot 3: 200"
echo ""
echo "【漸進式不平衡 - collision=50】(configs 4-6)"
echo "  4. Robot 0: 1000 | Robots 1,2,3: 200"
echo "  5. Robots 0,1: 1000 | Robots 2,3: 200"
echo "  6. Robots 0,1,2: 1000 | Robot 3: 200"
echo ""
echo "【虐殺模式 - collision=10】(configs 7-9)"
echo "  7. [1v3] Robot 0: 1000 | Robots 1,2,3: 20"
echo "  8. [2v2] Robots 0,1: 1000 | Robots 2,3: 20"
echo "  9. [3v1] Robots 0,1,2: 1000 | Robot 3: 20"
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
    --batch-size 128 \
    --save-dir "$run_dir" \
    --save-frequency 100 \
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
