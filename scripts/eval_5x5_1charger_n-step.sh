#!/bin/bash

# 機器人吸塵器 RL 批次評估腳本 (長時間版本)
# 評估虐殺模式：高血量機器人 vs 低血量機器人 的訓練結果
# 50000 步長期觀察

set -e

# 參數解析：
# $1: 模型目錄名稱（必填，例如：massacre_5x5_center_nstep*_20251216_003734）
# $2: episode checkpoint（選填，預設 5000）
if [ -z "$1" ]; then
  echo "錯誤：請提供模型目錄名稱"
  echo "用法: $0 <模型目錄名稱> [episode數量]"
  echo "範例: $0 massacre_5x5_center_nstep2_20251216_111427 2000"
  echo ""
  echo "可用的模型目錄："
  ls -d "${SCRIPT_DIR:-./}"/models/massacre_5x5_center_nstep* 2>/dev/null | head -5 || echo "  (無)"
  exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

# 基礎模型目錄 (使用已訓練好的模型)
BASE_MODELS_DIR="${SCRIPT_DIR}/models/$1"

# 選擇要評估的 episode checkpoint (預設 5000)
EPISODE="${2:-5000}"

# 定義 n-step 值
declare -a n_steps=(2 3 4 5)

# 最大步數 (敵人初始能量 100，每步 -1，足夠讓自然死亡完成)
MAX_STEPS=5000

echo "=========================================="
echo "機器人吸塵器強化學習 - 虐殺模式評估 (N-step Sweep 版)"
echo "=========================================="
echo "開始時間: $(date)"
echo "基礎模型目錄: $BASE_MODELS_DIR"
echo "評估 Episode: $EPISODE"
echo "N-step 值: ${n_steps[@]}"
echo "最大步數: $MAX_STEPS (長期觀察)"
echo "=========================================="
echo ""

# 定義評估配置 (只跑 1v3)
# 格式: "配置名稱|評估參數 (energy 要與訓練時一致)"
declare -a configs=(
  "massacre-1v3-5x5-center|--env-n 5 --charger-positions 2,2 --robot-0-energy 1000 --robot-1-energy 100 --robot-2-energy 100 --robot-3-energy 100 --e-collision 20 --e-charge 20"
)

total_configs=$((${#configs[@]} * ${#n_steps[@]}))
echo "共有 $total_configs 個評估配置 - 長期觀察版"
echo ""
echo "血量配置 (虐殺模式)："
echo "  1. [1v3] Robot 0: 1000 | Robots 1,2,3: 100 (一強 vs 三弱)"
echo ""
echo "評估參數（與訓練時相同）："
echo "  - e-collision: 20"
echo "  - e-charge: 20"
echo "  - 地圖: 5x5，充電器: (2,2)（中心）"
echo ""
echo "終止條件: 存活數 ≤ 1 或達到最大步數"
echo ""

# 遍歷每個配置和每個 n-step 值並執行評估
total_counter=0
for ((i=0; i<${#configs[@]}; i++)); do
  config="${configs[$i]}"
  config_name="${config%|*}"
  config_params="${config#*|}"

  for n in "${n_steps[@]}"; do
    total_counter=$((total_counter + 1))

    # 生成此次評估的名稱
    run_name="${config_name}-nstep${n}"

    # 模型路徑 (找對應的 nstep 子目錄)
    # 例如: $BASE_MODELS_DIR/massacre-1v3-5x5-center-nstep3/episode_5000
    model_dir="$BASE_MODELS_DIR/$run_name/episode_$EPISODE"

    # 評估 log 儲存路徑 (與模型放在同一目錄)
    eval_log="$BASE_MODELS_DIR/$run_name/eval_ep${EPISODE}_long.log"

    echo "=========================================="
    echo "[$total_counter/$total_configs] 評估: $run_name (長期)"
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
      --charger-positions "2,2" \
      --max-steps $MAX_STEPS \
      --seed 42 \
      --wandb-entity lazyhao-national-taiwan-university \
      --wandb-project robot-vacuum-eval \
      --wandb-run-name "eval-$run_name-ep$EPISODE-long" \
      --wandb-mode online \
      2>&1 | tee "$eval_log"

    echo ""
    echo "✓ 評估完成: $run_name"
    echo "Log 已儲存到: $eval_log"
    echo "結束時間: $(date)"
    echo ""

    # 評估間隔
    if [ $total_counter -lt $total_configs ]; then
      echo "等待 3 秒後開始下一個評估..."
      sleep 3
    fi
  done
done

echo "=========================================="
echo "所有 N-step 長期評估完成！"
echo "結束時間: $(date)"
echo "=========================================="
echo ""
echo "評估完成的 N-step 模型："
echo "  - nstep2, nstep3, nstep4, nstep5"
echo ""
echo "查看結果: https://wandb.ai/lazyhao-national-taiwan-university/robot-vacuum-eval"
