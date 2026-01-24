# Evaluation Scripts Summary

This document summarizes the evaluation scripts in `scripts/` and the parameters they run with.

## Common Settings (All Evaluations)

- **Episode Checkpoint**: 2000 (fully converged)
- **Seed**: 42 (deterministic)
- **Policy**: Deterministic (epsilon = 0.0)
- **WandB Settings**:
  - Entity: `lazyhao-national-taiwan-university`
  - Project: `robot-vacuum-eval`
  - Mode: `online`

---

## 1. Massacre Mode Evaluation (High vs Low Energy)

**Model Directory**: `models/all_energy_configs_20251125_211122`

Energy split: High-energy robots (1000) vs Low-energy robots (20)

**Scripts**: `scripts/eval_massacre.sh`, `scripts/eval_massacre_long.sh`

| Config Name | Robots 0-3 Energy | e-collision | Description | Max Steps | Actual Steps | Log |
|---|---|---|---|---|---|---|
| `massacre-1v3` | 1000, 20, 20, 20 | 10 | One strong (robot 0) vs three weak | 10,000 | 321 | `massacre-1v3/eval_ep2000.log` |
| `massacre-2v2` | 1000, 1000, 20, 20 | 10 | Two strong (0-1) vs two weak (2-3) | 50,000 | 50,000 | `massacre-2v2/eval_ep2000.log` |
| `massacre-3v1` | 1000, 1000, 1000, 20 | 10 | Three strong (0-2) vs one weak (robot 3) | 50,000 | 50,000 | `massacre-3v1/eval_ep2000.log` |

---

## 2. Different Energy Config Evaluation

**Model Directory**: `models/all_energy_configs_20251125_211122`

Energy split: High-energy robot(s) (1000) vs Medium-energy robots (200)

**Scripts**: `scripts/eval_diff_energy.sh`, `scripts/eval_diff_energy_long.sh`

| Config Name | Robots 0-3 Energy | e-collision | Description | Max Steps | Actual Steps | Log |
|---|---|---|---|---|---|---|
| `collision-10-robot0-1000-others-200` | 1000, 200, 200, 200 | 10 | One strong vs three medium | 50,000 | 50,000 | `collision-10-robot0-1000-others-200/eval_ep2000.log` |
| `collision-10-robot01-1000-others-200` | 1000, 1000, 200, 200 | 10 | Two strong vs two medium | 50,000 | 50,000 | `collision-10-robot01-1000-others-200/eval_ep2000.log` |
| `collision-10-robot012-1000-robot3-200` | 1000, 1000, 1000, 200 | 10 | Three strong vs one medium | 50,000 | 33,923 | `collision-10-robot012-1000-robot3-200/eval_ep2000.log` |
| `collision-50-robot0-1000-others-200` | 1000, 200, 200, 200 | 50 | One strong vs three medium (higher collision) | 10,000 | 2,047 | `collision-50-robot0-1000-others-200/eval_ep2000.log` |
| `collision-50-robot01-1000-others-200` | 1000, 1000, 200, 200 | 50 | Two strong vs two medium (higher collision) | 10,000 | 522 | `collision-50-robot01-1000-others-200/eval_ep2000.log` |
| `collision-50-robot012-1000-robot3-200` | 1000, 1000, 1000, 200 | 50 | Three strong vs one medium (higher collision) | 10,000 | 5,854 | `collision-50-robot012-1000-robot3-200/eval_ep2000.log` |

---

## 3. Same Energy Evaluation (Epsilon Decay)

**Model Directory**: `models/epsilon_decay_20251118_112056` (restored)

All robots have equal energy (homogeneous teams)

**Script**: `scripts/eval_same_energy.sh`

| Config Name | Robots 0-3 Energy | e-collision | Description | Max Steps | Actual Steps | Log |
|---|---|---|---|---|---|---|
| `collision-10-energy-150-epsilon-decay` | 150, 150, 150, 150 | 10 | All robots same energy (150) | 10,000 | 352 | `collision-10-energy-150-epsilon-decay/eval_ep2000.log` |
| `collision-10-energy-200-epsilon-decay` | 200, 200, 200, 200 | 10 | All robots same energy (200) | 10,000 | 3,875 | `collision-10-energy-200-epsilon-decay/eval_ep2000.log` |
| `collision-50-energy-100-epsilon-decay` | 100, 100, 100, 100 | 50 | All robots same energy (100) | 10,000 | 627 | `collision-50-energy-100-epsilon-decay/eval_ep2000.log` |

## Notes
- All scripts call `evaluate_models.py` with `--seed 42` to increase reproducibility and set deterministic policy behavior (epsilon is specified per config or defaults to 0.0 in the evaluation script).
- Scripts set `--wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-eval` and use `--wandb-mode online` by default (they upload runs to WandB). Change `--wandb-mode offline` to avoid uploads.
- Directory layout expected by scripts: `models/<config>/episode_<EPISODE>/robot_*.pt`.
- Long-run scripts use `MAX_STEPS=50000` to observe emergent dynamics over longer horizons; short-run use `10000` steps.

## Quick actions

To run a single config locally (example):

```bash
bash scripts/eval_same_energy.sh
# or run a specific script
bash scripts/eval_massacre.sh
```