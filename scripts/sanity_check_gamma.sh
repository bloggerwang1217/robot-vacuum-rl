#!/bin/bash
# This script runs a series of sanity check experiments.
# These configurations are useful for establishing baseline performance
# by testing edge cases of the learning parameters (e.g., epsilon=0, epsilon=1, gamma=0).

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
BASE_MODELS_DIR="${SCRIPT_DIR}/models/sanity_checks_$(date +%Y%m%d_%H%M%S)"

echo "==========================================================="
echo "Running Sanity Check Experiments"
echo "Global settings: --num-episodes 2000, --max-episode-steps 1000"
echo "Results will be saved under: ${BASE_MODELS_DIR}"
echo "==========================================================="

declare -a configs=(
  "sanity-c10-e150-eps0-gamma0|--e-collision 10 --initial-energy 150 --epsilon 0 --gamma 0"
)

for config_str in "${configs[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running Sanity Check: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
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
done

echo "==========================================================="
echo "All sanity check runs complete."
echo "Results saved to: ${BASE_MODELS_DIR}"
echo "==========================================================="
