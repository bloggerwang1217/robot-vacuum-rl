#!/bin/bash
# This script runs a series of sanity check experiments with 1 charger at center (1,1).
# These configurations are useful for establishing baseline performance
# by testing edge cases of the learning parameters (e.g., epsilon=0, epsilon=1, gamma=0).

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
MODELS_DIR="${SCRIPT_DIR}/models/1charger-massacre_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="]
echo "機器人吸塵器強化學習 - 虐殺模式訓練"
echo "=========================================="
echo "開始時間: $(date)"
echo "模型儲存目錄: $MODELS_DIR"
echo "=========================================="
echo ""

# 定義訓練配置
# 格式: "配置名稱|訓練參數"
declare -a configs=(
  "1charger-massacre-3v1|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
)

for config_str in "${configs[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running 虐殺模式訓練 with 1 charger: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
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
done

echo "==========================================================="
echo "虐殺模式訓練 with 1 charger runs complete."
echo "Results saved to: ${MODELS_DIR}"
echo "==========================================================="
