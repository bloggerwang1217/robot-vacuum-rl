#!/bin/bash

# 機器人吸塵器 RL 批次評估腳本 (長時間版本)
# 評估剩餘 6 個配置：collision-50 (3組) + 同能量 (3組)
# 50000 步長期觀察

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

# 選擇要評估的 episode checkpoint (推薦 2000 = 完全收斂)
EPISODE="2000"

# 最大步數 (長期觀察)
MAX_STEPS=50000

echo "=========================================="
echo "機器人吸塵器強化學習 - 剩餘 6 組評估 (長期版)"
echo "=========================================="
echo "開始時間: $(date)"
echo "評估 Episode: $EPISODE"
echo "最大步數: $MAX_STEPS (長期觀察)"
echo "=========================================="
echo ""

# 定義評估配置 (剩餘 6 個)
# 格式: "模型來源目錄|配置名稱|評估參數"
declare -a configs=(
  # 標準二後半：不同能量配置 (collision-50, 3 組)
  "models/all_energy_configs_20251125_211122|collision-50-robot0-1000-others-200|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50"
  "models/all_energy_configs_20251125_211122|collision-50-robot01-1000-others-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50"
  "models/all_energy_configs_20251125_211122|collision-50-robot012-1000-robot3-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 50"

  # 標準三：同能量配置 (3 組)
  "models/epsilon_decay_20251118_112056|collision-10-energy-150-epsilon-decay|--robot-0-energy 150 --robot-1-energy 150 --robot-2-energy 150 --robot-3-energy 150 --e-collision 10"
  "models/epsilon_decay_20251118_112056|collision-10-energy-200-epsilon-decay|--robot-0-energy 200 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10"
  "models/epsilon_decay_20251118_112056|collision-50-energy-100-epsilon-decay|--robot-0-energy 100 --robot-1-energy 100 --robot-2-energy 100 --robot-3-energy 100 --e-collision 50"
)

total_configs=${#configs[@]}
echo "共有 $total_configs 個評估配置"
echo ""
echo "配置組成："
echo "  標準二後半 (collision-50, 3 組)："
echo "    1. [robot0-1000] Robot 0: 1000 | Robots 1,2,3: 200"
echo "    2. [robot01-1000] Robots 0,1: 1000 | Robots 2,3: 200"
echo "    3. [robot012-1000] Robots 0,1,2: 1000 | Robot 3: 200"
echo "  標準三 (同能量, 3 組)："
echo "    4. collision-10-energy-150"
echo "    5. collision-10-energy-200"
echo "    6. collision-50-energy-100"
echo ""
echo "終止條件: 存活數 ≤ 1 或達到最大步數"
echo ""

# 遍歷每個配置並執行評估
for ((i=0; i<${#configs[@]}; i++)); do
  config="${configs[$i]}"

  # 解析配置 (三欄位)
  IFS='|' read -r models_dir config_name config_params <<< "$config"
  models_dir="$SCRIPT_DIR/$models_dir"

  # 計算進度
  current=$((i+1))

  # 模型路徑
  model_dir="$models_dir/$config_name/episode_$EPISODE"

  # 評估 log 儲存路徑 (與模型放在同一目錄)
  eval_log="$models_dir/$config_name/eval_ep${EPISODE}_long.log"

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
echo "所有剩餘 6 組評估完成！"
echo "結束時間: $(date)"
echo "=========================================="
echo ""
echo "查看結果: https://wandb.ai/lazyhao-national-taiwan-university/robot-vacuum-eval"
