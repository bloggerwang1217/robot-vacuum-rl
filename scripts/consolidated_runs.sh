#!/bin/bash
# This is a consolidated script containing all training runs from various .sh files.
# The original loop structure is preserved for clarity and manageability.
# All runs have been updated to use --num-episodes 2000 and --max-episode-steps 1000.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
BASE_MODELS_DIR="${SCRIPT_DIR}/models/consolidated_runs_$(date +%Y%m%d_%H%M%S)"

echo "==========================================================="
echo "Consolidated Training Runs - FULL VERSION"
echo "Global settings: --num-episodes 2000, --max-episode-steps 1000"
echo "Results will be saved under: ${BASE_MODELS_DIR}"
echo "==========================================================="

# --- From scripts/train_all_diff_energy.sh ---
echo ""
echo ">>> Section: Running experiments from train_all_diff_energy.sh <<<"
declare -a configs_all_diff=(
  "collision-10-robot0-1000-others-200|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  "collision-10-robot01-1000-others-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  "collision-10-robot012-1000-robot3-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  "collision-50-robot0-1000-others-200|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  "collision-50-robot01-1000-others-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  "collision-50-robot012-1000-robot3-200|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  "massacre-1v3|--robot-0-energy 1000 --robot-1-energy 20 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-2v2|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-3v1|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
)
for config_str in "${configs_all_diff[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
    $config_params \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --batch-size 128 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-rl --wandb-run-name "$config_name" --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"
done

# --- From scripts/train_batch.sh ---
echo ""
echo ">>> Section: Running experiments from train_batch.sh <<<"
declare -a configs_batch=(
  "collision-10-energy-150|--e-collision 10 --initial-energy 150"
  "collision-50-energy-100-epsilon-0|--e-collision 50 --initial-energy 100 --epsilon 0"
  "collision-50-energy-100-epsilon-0.5|--e-collision 50 --initial-energy 100 --epsilon 0.5"
  "collision-50-energy-100-epsilon-1.0|--e-collision 50 --initial-energy 100 --epsilon 1.0"
  "collision-50-energy-100-gamma-0|--e-collision 50 --initial-energy 100 --gamma 0"
  "collision-10-energy-200-epsilon-0.3|--e-collision 10 --initial-energy 200 --epsilon 0.3"
)
for config_str in "${configs_batch[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
    $config_params \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-rl --wandb-run-name "$config_name" --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"
done

# --- From scripts/train_different_energy_colli_10.sh ---
echo ""
echo ">>> Section: Running experiments from train_different_energy_colli_10.sh <<<"
declare -a configs_diff_10=(
  "diff-energy-10-robot0-1000|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  "diff-energy-10-robot01-1000|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
  "diff-energy-10-robot012-1000|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 10 --use-epsilon-decay"
)
for config_str in "${configs_diff_10[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
    $config_params \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-rl --wandb-run-name "$config_name" --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"
done

# --- From scripts/train_different_energy_colli_50.sh ---
echo ""
echo ">>> Section: Running experiments from train_different_energy_colli_50.sh <<<"
declare -a configs_diff_50=(
  "diff-energy-50-robot0-1000|--robot-0-energy 1000 --robot-1-energy 200 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  "diff-energy-50-robot01-1000|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 200 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
  "diff-energy-50-robot012-1000|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 200 --e-collision 50 --use-epsilon-decay"
)
for config_str in "${configs_diff_50[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
    $config_params \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-rl --wandb-run-name "$config_name" --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"
done

# --- From scripts/train_epsilon_decay.sh ---
echo ""
echo ">>> Section: Running experiments from train_epsilon_decay.sh <<<"
declare -a configs_eps_decay=(
  "eps-decay-c10-e200|--e-collision 10 --initial-energy 200 --use-epsilon-decay"
  "eps-decay-c50-e100|--e-collision 50 --initial-energy 100 --use-epsilon-decay"
  "eps-decay-c10-e150|--e-collision 10 --initial-energy 150 --use-epsilon-decay"
)
for config_str in "${configs_eps_decay[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
    $config_params \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-rl --wandb-run-name "$config_name" --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"
done

# --- From scripts/train_high_move.sh ---
echo ""
echo ">>> Section: Running experiments from train_high_move.sh <<<"
declare -a configs_high_move=(
  "high-move-m10-c50-e100|--e-move 10 --e-collision 50 --initial-energy 100 --use-epsilon-decay"
  "high-move-m25-c50-e100|--e-move 25 --e-collision 50 --initial-energy 100 --use-epsilon-decay"
  "high-move-m50-c50-e100|--e-move 50 --e-collision 50 --initial-energy 100 --use-epsilon-decay"
  "high-move-m10-c50-e150|--e-move 10 --e-collision 50 --initial-energy 150 --use-epsilon-decay"
  "high-move-m50-c50-e150|--e-move 50 --e-collision 50 --initial-energy 150 --use-epsilon-decay"
)
for config_str in "${configs_high_move[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
    $config_params \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-rl --wandb-run-name "$config_name" --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"
done

# --- From scripts/train_massacre.sh ---
echo ""
echo ">>> Section: Running experiments from train_massacre.sh <<<"
declare -a configs_massacre=(
  "massacre-1v3-new|--robot-0-energy 1000 --robot-1-energy 20 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-2v2-new|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 20 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
  "massacre-3v1-new|--robot-0-energy 1000 --robot-1-energy 1000 --robot-2-energy 1000 --robot-3-energy 20 --e-collision 10 --use-epsilon-decay"
)
for config_str in "${configs_massacre[@]}"; do
  config_name="${config_str%|*}"
  config_params="${config_str#*|}"
  run_dir="${BASE_MODELS_DIR}/${config_name}"
  mkdir -p "$run_dir"
  echo "--- Running: ${config_name} ---"
  python3 "${SCRIPT_DIR}/train_dqn.py" \
    $config_params \
    --num-episodes 2000 \
    --max-episode-steps 1000 \
    --batch-size 128 \
    --save-dir "$run_dir" \
    --save-frequency 500 \
    --wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-rl --wandb-run-name "$config_name" --wandb-mode online \
    2>&1 | tee "$run_dir/training.log"
done


echo "==========================================================="
echo "All consolidated training configurations are in this script."
echo "==========================================================="
