#!/bin/bash

# 機器人吸塵器 RL 批次評估腳本 (長時間版本)
# 評估不同能量配置：不同機器人組合的訓練結果
# 50000 步長期觀察

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

# 模型目錄 (使用已訓練好的模型)
MODELS_DIR="${SCRIPT_DIR}/models/all_energy_configs_20251125_211122"

# 選擇要評估的 episode checkpoint (推薦 2000 = 完全收斂)
EPISODE="2000"

# 最大步數 (長期觀察)
MAX_STEPS=50000

echo "=========================================="
echo "機器人吸塵器強化學習 - 不同能量配置評估 (長期版)"
echo "=========================================="
echo "開始時間: $(date)"
echo "模型來源目錄: $MODELS_DIR"
echo "評估 Episode: $EPISODE"
echo "最大步數: $MAX_STEPS (長期觀察)"
echo "=========================================="
echo ""

# 定義評估配置 (只跑 e-collision 10 的三個配置)
# 格式: "配置名稱|評估參數 (energy 要與訓練時一致)"
declare -a configs=(
  "collision-10-robot0-1000-others-200|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10"
  "collision-10-robot01-1000-others-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10"
  "collision-10-robot012-1000-robot3-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 10"
)

total_configs=${#configs[@]}
echo "共有 $total_configs 個評估配置 - 長期觀察版"
echo ""
echo "能量配置 (e-collision 10)："
echo "  1. [robot0-1000] Robot 0: 1000 | Robots 1,2,3: 200"
echo "  2. [robot01-1000] Robots 0,1: 1000 | Robots 2,3: 200"
echo "  3. [robot012-1000] Robots 0,1,2: 1000 | Robot 3: 200"
echo ""
echo "終止條件: 存活數 ≤ 1 或達到最大步數"
echo ""

# 遍歷每個配置並執行評估
for ((i=0; i<${#configs[@]}; i++)); do
  config="${configs[$i]}"
  config_name="${config%|*}"
  config_params="${config#*|}"

  # 計算進度
  current=$((i+1))

  # 模型路徑
  model_dir="$MODELS_DIR/$config_name/episode_$EPISODE"

  # 評估 log 儲存路徑 (與模型放在同一目錄)
  eval_log="$MODELS_DIR/$config_name/eval_ep${EPISODE}_long.log"

  echo "=========================================="
  echo "[$current/$total_configs] 評估: $config_name (長期)"
  echo "模型路徑: $model_dir"
  echo "參數: $config_params"
  echo "Log 儲存: $eval_log"
  echo "開始時間: $(date)"
  echo "=========================================="

  # 檢查模型是否存在
  if [ ! -d "$model_dir" ]; then
    echo "⚠️  警告: 模型目錄不存在: $model_dir"
    echo "跳過此配置..."
    echo ""
    continue
  fi

  # 執行評估並儲存 log (使用 deterministic policy, epsilon=0)
  python3 "$SCRIPT_DIR/evaluate_models.py" \
    --model-dir "$model_dir" \
    $config_params \
    --max-steps $MAX_STEPS \
    --seed 42 \
    --wandb-entity lazyhao-national-taiwan-university \
    --wandb-project robot-vacuum-eval \
    --wandb-run-name "eval-$config_name-ep$EPISODE-long" \
    --wandb-mode online \
    2>&1 | tee "$eval_log"

  echo ""
  echo "✓ 評估完成: $config_name"
  echo "Log 已儲存到: $eval_log"
  echo "結束時間: $(date)"
  echo ""

  # 評估間隔
  if [ $current -lt $total_configs ]; then
    echo "等待 3 秒後開始下一個評估..."
    sleep 3
  fi
done

echo "=========================================="
echo "所有長期評估完成！"
echo "結束時間: $(date)"
echo "=========================================="
echo ""
echo "查看結果: https://wandb.ai/lazyhao-national-taiwan-university/robot-vacuum-eval"
