#!/bin/bash

# 機器人吸塵器 RL 批次評估腳本 (長時間版本)
# 評估 1-charger Imbalanced Collision (Imbalanced 版本)
# 50000 步長期觀察

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

# 模型目錄 (使用已訓練好的模型)
MODELS_DIR="${SCRIPT_DIR}/models/1charger-imbalanced_collision_20251203_102459"

# 選擇要評估的 episode checkpoint (推薦 2000 = 完全收斂)
EPISODE="2000"

# 最大步數 (長期觀察)
MAX_STEPS=50000

echo "=========================================="
echo "機器人吸塵器強化學習 - 1-Charger Imbalanced Collision 評估 (Imbalanced 版本)"
echo "=========================================="
echo "開始時間: $(date)"
echo "模型來源目錄: $MODELS_DIR"
echo "評估 Episode: $EPISODE"
echo "最大步數: $MAX_STEPS (長期觀察)"
echo "=========================================="
echo ""

# 定義評估配置 (只跑 imbalanced)
# 格式: "配置名稱|評估參數 (energy 要與訓練時一致)"
declare -a configs=(
  "1charger-collision-50-energy-100-imbalanced|--initial-energy 100 --e-collision-active-one-sided 25 --e-collision-active-two-sided 50 --e-collision-passive 50"
)

total_configs=${#configs[@]}
echo "共有 $total_configs 個評估配置 - 長期觀察版"
echo ""
echo "碰撞參數設定 (Imbalanced 版本)："
echo "  1. active-one-sided: 25, active-two-sided: 50, passive: 50"

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
  eval_log="$MODELS_DIR/$config_name/eval_ep${EPISODE}.log"

  echo "=========================================="
  echo "[$current/$total_configs] 評估: $config_name"
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
    --charger-positions "1,1" \
    --max-steps $MAX_STEPS \
    --seed 42 \
    --eval-epsilon 0.0 \
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
