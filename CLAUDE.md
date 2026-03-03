# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Goal

**核心問題**：在資源稀缺的多智能體環境中，能否讓強者自然學會「消滅弱者以獨佔資源」的攻擊行為——而不在 reward 或 loss 中明確寫入任何攻擊獎勵？

攻擊行為必須是 **emergent behavior**：agent 純粹因為攻擊在數學上是最大化長期 reward 的最優策略，才選擇攻擊。

### 設計邏輯

Reward 只包含三個 **間接** 訊號：
1. **能量變化**（`ΔE × 0.05`）：充電賺錢、受傷虧損
2. **灰塵收割**（`dust × scale`）：鼓勵移動、探索地圖
3. **死亡懲罰**（`-100`）：活下去有價值

「消滅對手」本身沒有 reward。但若消滅對手後能獨佔充電座或灰塵資源，長期 Q-value 會更高——攻擊行為因此可能自然浮現。

### 核心挑戰：習得無助（Learned Helplessness）

理論分析顯示，攻擊性出現的 **必要條件** 是「弱者必須真正競爭資源」。若弱者因實力差距太大而選擇放棄（不靠近充電座），強者實質已獨佔資源，此時攻擊的機會成本（離開充電座 18 步 ≈ 損失 92 reward）大於收益，等待策略反而更優。

這形成一個循環：弱者因強者太強而不敢搶 → 強者因無人搶而不需殺 → 穩定在「和平獨裁」均衡。

**打破此均衡的嘗試**：
- 灰塵系統：提供充電座以外的資源，迫使雙方在地圖上競爭
- 充電座格灰塵弱化：降低蹲坐充電座的相對收益，逼迫機器人外出
- 不對稱能量配置（massacre 系列）：研究 1強 vs N弱 的動態
- N-step return：讓 agent 更能規劃長期策略（攻擊的收益是遠期的）

詳細數學分析見 `docs/emergent_aggression_analysis.md`。

---

## Environment Setup

```bash
source ~/.venv/bin/activate
pip install torch numpy gymnasium wandb pygame
pip install opencv-python  # optional, for replay recording
```

## Common Commands

**Train (main entry point):**
```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --charger-positions 2,2 \
  --num-episodes 10000 --num-envs 8 \
  --wandb-mode disabled
```

**Evaluate (loads saved models, generates replay JSON):**
```bash
python evaluate_models.py \
  --model-dir ./models/{run_name}/episode_N \
  --env-n 5 --charger-positions 2,0 \
  --max-steps 1000 --eval-epsilon 0
```

**Visualize replay:**
```bash
python replay.py --replay-file ./models/{run_name}/{run_name}-eval_replay.json
```

**Quick sanity check (validates DQN implementation):**
```bash
python sanity_check.py
```

**Preset experiment scripts** are in `scripts/` (e.g. `bash scripts/train_5x5_1charger.sh`).

## Architecture

The system has a 4-layer stack:

```
train_dqn_vec.py  (VectorizedMultiAgentTrainer + SharedBufferDQNAgent)
    └── vec_env.py           (VectorizedRobotVacuumEnv: N envs in parallel)
            └── gym.py       (RobotVacuumGymEnv: obs space, reward function)
                    └── robot_vacuum_env.py  (physics: collision, energy, dust)
```

### Core Modules

**`robot_vacuum_env.py`** — Physical engine. Handles:
- Collision rules (5 types; attacker never takes damage in sequential mode)
- Energy: move costs `e_move`, charging splits `e_charge/n` among robots in 3×3 charger range, no energy cap (overcharge allowed)
- Dust: sigmoid (logistic) growth per cell; robots harvest all dust when stepping on a cell; charger cells have weakened dust parameters

**`gym.py`** — Gymnasium wrapper. Defines:
- Observation vector: `[self_pos(2), self_energy(1), others(N-1)*3, chargers C*2, dust_grid n²]` — total `3 + (N-1)*3 + C*2 + n²`
- Reward: `ΔEnergy × 0.05 + dust_collected × dust_reward_scale - 100 × died`
- Two step modes: `step(actions)` simultaneous (used during training via vec_env), `step_single(robot_id, action)` sequential (used during evaluation)

**`vec_env.py`** — Wraps N `RobotVacuumGymEnv` instances, batches observations for GPU inference, handles auto-reset on episode end.

**`dqn.py`** — MLP: `obs_dim → 128 → 256 → 256 → 128 → 5 actions`. Kaiming init.

**`train_dqn_vec.py`** — `VectorizedMultiAgentTrainer` manages:
- One `SharedBufferDQNAgent` per robot (q_net + target_net + Adam)
- Independent replay buffers per agent (trainer-managed, not agent-owned)
- N-step return buffer per (env, agent) pair
- Sequential robot action order within each step

### Model Persistence

- Auto-named: `nstep{n}_episode{N}` if `--wandb-run-name` not set
- Saved to: `./models/{run_name}/episode_{N}/robot_{i}.pt` every `--save-frequency` episodes
- Post-training eval output: `./models/{run_name}/{run_name}-eval_replay.json`

### Key Design Decisions

- **IDQN**: each robot has its own independent Q-network (no shared parameters)
- **Sequential training mode**: robots act one at a time via `step_single()`; this avoids simultaneous-action edge cases (swap, contested)
- **No survival cost**: staying still costs 0 energy; only movement costs `e_move`
- **Dynamic energy normalization**: obs energy divided by current max energy across all alive robots (handles overcharge)
- **Charger dust weakening**: charger cells have lower `dust_max` and `dust_rate` to discourage camping
