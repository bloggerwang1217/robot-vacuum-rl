# Evaluation Scripts Summary

This document summarizes the evaluation scripts in `scripts/` and the parameters they run with. Use it to share with collaborators or your adviser.

| Script | Models Dir | Episode | Max Steps | Configs (short) | e-collision | Seed | WandB mode | Log path pattern | WandB run name pattern |
|---|---|---:|---:|---|---:|---:|---|---|---|
| `scripts/eval_massacre.sh` | `models/all_energy_configs_20251125_211122` | 2000 | 10000 | `massacre-1v3`; `massacre-2v2`; `massacre-3v1` — high-vs-low energy splits (1k vs 20) | 10 | 42 | `online` | `.../<config>/eval_ep2000.log` | `eval-<config>-ep2000` |
| `scripts/eval_massacre_long.sh` | `models/all_energy_configs_20251125_211122` | 2000 | 50000 | `massacre-2v2`; `massacre-3v1` — long-run variants | 10 | 42 | `online` | `.../<config>/eval_ep2000_long.log` | `eval-<config>-ep2000-long` |
| `scripts/eval_diff_energy.sh` | `models/all_energy_configs_20251125_211122` | 2000 | 10000 | Six configs: three high-energy groupings × `e-collision` in {10,50} (examples: `collision-10-robot0-1000-others-200`, etc.) | 10 / 50 | 42 | `online` | `.../<config>/eval_ep2000.log` | `eval-<config>-ep2000` |
| `scripts/eval_diff_energy_long.sh` | `models/all_energy_configs_20251125_211122` | 2000 | 50000 | Three long-run configs (only `e-collision=10`) — same energy splits as `eval_diff_energy.sh` | 10 | 42 | `online` | `.../<config>/eval_ep2000_long.log` | `eval-<config>-ep2000-long` |
| `scripts/eval_same_energy.sh` | `models/epsilon_decay_20251118_112056` (restored) | 2000 | 10000 | Three same-energy checkpoints from `epsilon_decay`: `collision-10-energy-150-epsilon-decay` (all robots energy=150); `collision-10-energy-200-epsilon-decay` (all=200); `collision-50-energy-100-epsilon-decay` (all=100) | 10 / 50 | 42 | `online` | `.../<config>/eval_ep2000.log` | `eval-<config>-ep2000` |


## Notes
- All scripts call `evaluate_models.py` with `--seed 42` to increase reproducibility and set deterministic policy behavior (epsilon is specified per config or defaults to 0.0 in the evaluation script).
- Scripts set `--wandb-entity lazyhao-national-taiwan-university --wandb-project robot-vacuum-eval` and use `--wandb-mode online` by default (they upload runs to WandB). Change `--wandb-mode offline` to avoid uploads.
- Directory layout expected by scripts: `models/<config>/episode_<EPISODE>/robot_*.pt`.
- Long-run scripts use `MAX_STEPS=50000` to observe emergent dynamics over longer horizons; short-run use `10000` steps.
- The `epsilon_decay` models were restored from git history and placed under `models/epsilon_decay_20251118_112056`. Confirm the `episode_2000` folders exist inside each listed config before running.

## Quick actions
- To commit this summary to git:

```bash
git add EVAL_SCRIPTS_SUMMARY.md
git commit -m "docs: add evaluation scripts summary"
```

- To run a single config locally (example):

```bash
bash scripts/eval_same_energy.sh
# or run a specific script
bash scripts/eval_massacre.sh
```