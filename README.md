# Multi-Robot Energy Survival — Emergent Aggression via IDQN

本專案研究一個核心問題：

> **在資源稀缺的多智能體環境中，能否讓強者自然學會「消滅弱者以獨佔資源」的攻擊行為——不在 reward 或 loss 中寫入任何攻擊獎勵？**

攻擊行為必須是 **emergent behavior**：agent 純粹因為攻擊在數學上是最大化長期 reward 的最優策略，才選擇攻擊。

---

## 核心設計邏輯

### Reward 只有三個間接訊號

```
r_t = ΔEnergy × 0.05  +  dust_collected × dust_scale  -  100 × died
```

| 事件 | Reward | 說明 |
|------|--------|------|
| 充電 | `+e_charge × 0.05` | 唯一的正 reward 來源 |
| 移動 | `-e_move × 0.05` | 可設為 0 |
| 碰撞受傷 | `-(e_collision) × 0.05` | |
| 死亡 | `-100` | 強烈的存活誘因 |

**「消滅對手」本身沒有 reward**。但若消滅對手後能獨佔充電座，長期 Q-value 會更高——攻擊行為因此可能自然浮現。

### 關鍵機制：獨佔充電（`--exclusive-charging`）

啟用後：**只要場上還有任何其他存活的 robot，所有人的充電座完全無效。**

```
robot_1 活著 → robot_0 完全無法充電（不管 robot_1 在哪）
robot_1 死亡 → robot_0 可以自由充電，每步 +0.5 reward
```

N-step return 的作用：`--n-step 20` 讓充電 reward 可以直接展開 20 步傳回去，無需依賴 Q-value 估計的精準度。

### 核心挑戰：習得無助（Learned Helplessness）

若讓兩個 agent 直接同時 IDQN 訓練，往往陷入「和平獨裁」均衡：
- 強者守在充電座，因弱者不來搶，攻擊的機會成本 > 收益
- 弱者因差距太大放棄靠近，充電座實質上已被獨佔

**打破此均衡的方法：Curriculum Training**——先在受控條件下強迫強者學會追殺，再讓雙方同時學習。

---

## 環境安裝

```bash
python -m venv ~/.venv
source ~/.venv/bin/activate
pip install torch numpy gymnasium wandb pygame
pip install opencv-python  # 可選，replay 錄影用
```

---

## 快速開始

```bash
# 訓練（預設自動啟用加速：batch-env + torch.compile + 256 envs）
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --charger-positions 2,2 \
  --num-episodes 10000 \
  --wandb-mode disabled

# 評估（訓練完自動執行，也可手動跑）
python evaluate_models.py \
  --model-dir ./models/{run_name}/episode_N \
  --env-n 5 --charger-positions 2,2 \
  --max-steps 1000 --eval-epsilon 0

# 視覺化回放
python replay.py --replay-file ./models/{run_name}/{run_name}-eval_replay.json
```

---

## Curriculum Training（推薦完整流程）

四個 phase 逐步提升對手難度：

```
Phase 1: robot_0 STAY        → robot_1 學習「靠近並撞死靜止目標」
Phase 2: robot_0 RANDOM      → robot_1 學習追蹤移動目標
Phase 3: robot_0 FLEE        → robot_1 學習對付會主動逃跑的對手
Phase 4: both learn (IDQN)   → robot_0 也開始學習，robot_1 已有強初始化
```

### 一鍵執行

```bash
bash scripts/train_curriculum.sh my_experiment
```

### 手動逐 phase 執行（6×6 範例）

```bash
# Phase 1：robot_0 固定不動，隨機起點
python train_dqn_vec.py \
  --env-n 6 --num-robots 2 \
  --robot-0-energy 20 --robot-1-energy 100 \
  --robot-0-speed 1 --robot-1-speed 2 \
  --charger-positions 3,3 --exclusive-charging --no-dust \
  --e-move 0 --e-charge 10 --e-collision 30 --e-boundary 0 \
  --n-step 20 --gamma 0.999 --max-episode-steps 1000 \
  --num-episodes 50000 --save-frequency 10000 \
  --wandb-mode online \
  --scripted-robots 0 --random-start-robots 0 \
  --no-eval-after-training --wandb-run-name p1_6x6

# Phase 2：robot_0 隨機走，接 Phase 1 checkpoint
python train_dqn_vec.py \
  [同上 env 參數] --num-episodes 50000 \
  --random-robots 0 --random-start-robots 0 \
  --load-model-dir ./models/p1_6x6/episode_50000 \
  --no-eval-after-training --wandb-run-name p2_6x6

# Phase 3：robot_0 啟發式逃跑
python train_dqn_vec.py \
  [同上 env 參數] --num-episodes 30000 \
  --flee-robots 0 --random-start-robots 0 \
  --load-model-dir ./models/p2_6x6/episode_50000 \
  --no-eval-after-training --wandb-run-name p3_6x6

# Phase 4：雙方同時 IDQN
python train_dqn_vec.py \
  [同上 env 參數] --num-episodes 50000 \
  --load-model-dir ./models/p3_6x6/episode_30000 \
  --wandb-run-name p4_6x6
```

---

## 專案架構

```
train_dqn_vec.py  →  batch_env.py / vec_env.py  →  gym.py  →  robot_vacuum_env.py
     訓練器              向量化環境                  Gym 封裝       物理引擎
```

```
robot-vacuum-rl/
├── robot_vacuum_env.py       # 物理引擎：碰撞、能量、灰塵（1192 行）
├── gym.py                    # Gymnasium 封裝：obs 向量、reward 函數（478 行）
├── vec_env.py                # N 個環境並行（Python 迴圈版）
├── batch_env.py              # N 個環境並行（全 numpy 批次版，預設啟用）
├── dqn.py                    # DQN 神經網路（MLP 5 層）
├── train_dqn_vec.py          # 主訓練腳本 ⭐（1620 行）
├── evaluate_models.py        # 評估腳本，產生 replay JSON
├── replay.py                 # pygame 回放視覺化
├── sanity_check.py           # DQN 正確性驗證
├── analysis/
│   ├── survival_analysis.py      # 存活率統計與熱圖
│   ├── analyze_wandb_behavior.py # W&B 行為分析（攻擊、存活、追殺圖表）
│   ├── analyze_training.py       # 訓練過程分析
│   ├── web_dashboard.py          # Web 儀表板視覺化
│   └── run_all_analyses.sh       # 一鍵執行所有分析
├── scripts/
│   ├── train_curriculum.sh       # 四 phase curriculum 自動訓練
│   ├── train_5x5_1charger.sh     # 5×5 單充電座訓練
│   ├── train_1charger_massacre-*.sh  # Massacre 系列（1v3, 2v2, 3v1）
│   ├── train_1charger_imbalanced_collision.sh
│   ├── train_4chargers_imbalanced_collision.sh
│   ├── train_massacre_4chargers.sh
│   ├── eval_*.sh                 # 對應的評估腳本
│   ├── sanity_check*.sh          # 各種健全性檢查
│   ├── fetch_wandb_data.py       # 從 W&B 抓取訓練資料
│   └── profile_sections.py       # 效能分段 profiling
├── docs/
│   ├── emergent_aggression_analysis.md  # 數學分析：攻擊 vs 等待的 Q-value
│   ├── kill_vs_wait_boundary_condition.md # 殺 vs 等的邊界條件分析
│   ├── multi_step_return_proposal.md    # N-step return 設計提案
│   ├── vectorized_training_optimization.md # 向量化訓練優化筆記
│   ├── parallelization.md               # 並行化設計
│   ├── trait.md                         # Agent Type 實驗設計
│   ├── energy.md                        # Heterotype 充電折減實驗設計
│   ├── PLAN.md                          # 研究計畫
│   └── EVAL_SCRIPTS_SUMMARY.md          # 評估腳本總覽
├── legacy/                    # 舊版程式碼（已棄用）
│   ├── train_dqn.py               # 舊版單環境訓練
│   ├── run_simulation.py           # 舊版模擬
│   ├── energy_survival_config.py   # 舊版配置
│   └── profile_training.py         # 舊版 profiling
└── models/
    └── {run_name}/
        ├── episode_N/
        │   ├── robot_0.pt
        │   └── robot_1.pt
        └── {run_name}-eval_replay.json
```

---

## 模組說明

### 物理引擎（`robot_vacuum_env.py`）

每個 timestep 的執行順序：**移動 → 收集灰塵 → 充電 → 死亡判定**

**能量機制：**

| 事件 | 能量變化 |
|------|----------|
| 移動 | `-e_move` |
| 停留 | `0`（無生存成本） |
| 充電 | `+e_charge / n`（充電座範圍內 n 台 robot 平分，`--charger-range 1`=3×3, `0`=僅該格） |
| 被 knockback | `-e_collision` |
| 撞牆 | `-e_boundary` |

能量無上限（overcharge 允許）。

**碰撞規則（sequential mode）：**

| 情境 | 攻擊方 | 被攻擊方 |
|------|--------|---------|
| knockback 成功（有路可推） | 移入目標格，不受傷 | 被推開一格，`-e_collision` |
| stationary blocked（無路可推） | 停在原位，不受傷 | `-e_collision` |
| 撞牆 | 停在原位，`-e_boundary` | — |

攻擊方不受傷，被攻擊方扣血——攻擊的預期收益永遠 ≥ 0。

**灰塵系統（`--no-dust` 可停用）：**

每格灰塵以 logistic growth 方式累積：

$$\Delta D = \text{rate} \times (D + \varepsilon) \times \left(1 - \frac{D}{D_{\max}}\right)$$

充電座格的 `D_max` 和 `rate` 會被弱化（避免蹲點刷灰塵）。

### Gymnasium 封裝（`gym.py`）

**觀測向量：**

```
obs = [x, y, energy, self_type, wall_up, wall_down, wall_left, wall_right,
       (dx_i, dy_i, energy_i, type_i) × (N-1),
       (dx_c, dy_c) × C,
       dust_grid × n²]
```

- 所有位置除以 `(n-1)` 正規化到 `[0,1]`
- 能量除以當前場上最高實際能量（動態正規化，應對 overcharge）
- `self_type` / `type_i`：agent type 標記（`--agent-types-mode observe` 時填入，`off` 時 padding 0，維度一致）

| 配置 | obs\_dim 計算 |
|------|---------------|
| 基礎 | `3 + 1 + 4 + (N-1)×4 + C×2` |
| 含灰塵 | `+ n²` |
| 範例：2 robots, 1 charger, 5×5, no dust | `3+1+4+4+2 = 14` |
| 範例：4 robots, 1 charger, 5×5, no dust | `3+1+4+12+2 = 22` |

**獎勵函數：**

$$r_t = \Delta E \times 0.05 \;[+ D_{\text{collected}} \times \text{dust\_scale}] \;- 100 \cdot \mathbf{1}[\text{died}]$$

### 神經網路（`dqn.py`）

```
Input(obs_dim) → Linear(128) → ReLU → Linear(256) → ReLU
              → Linear(256) → ReLU → Linear(128) → ReLU
              → Linear(5)   [UP, DOWN, LEFT, RIGHT, STAY]
```

Kaiming uniform 初始化。

### 訓練器（`train_dqn_vec.py`）

**IDQN**：每個 robot 有獨立的 Q-network、target network、replay buffer，互不共享參數。

**每個 timestep 的訓練迴圈：**
```
for robot_id in range(n_agents):
    for turn in range(robot_speeds[robot_id]):   # speed=2 則執行兩次
        obs = get_observation(robot_id)
        action = Q-network.select(obs)           # 每 turn 重新決策
        step_single(robot_id, action)
        store_n_step_transition(...)
advance_step()
```

**N-step Return：**

```
target = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(n-1)·r_{t+n-1} + γ^n · Q(s_{t+n})
```

前 n 步 reward 直接展開（不靠 Q-value 估計），n 步以後靠 bootstrap。

**Prioritized Experience Replay（`--per`）：**

基於 TD-error 的優先級採樣（SumTree 實作），自動調整 IS-correction weights。Beta 從 `per-beta-start` 線性退火到 1.0。

---

## Agent Type 實驗

測試僅在 observation 中暴露 agent type 標記（circle / triangle），不修改 reward 或環境規則，是否足以誘發 type-based targeting。

- `--agent-types-mode off`（預設）：type 欄位填 0（padding），維度不變
- `--agent-types-mode observe`：type 欄位填實際值（circle=0, triangle=1）
- `--triangle-agent-id N`：指定哪個 robot 是 triangle（不指定 = 全部 circle）

### Control 組（純觀察，無充電折減）

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 4 \
  --agent-types-mode observe --triangle-agent-id 2 \
  --charger-positions 2,2 \
  --heterotype-charge-mode off \
  --num-episodes 1000000 \
  --wandb-run-name type_control
```

### Heterotype Charging Penalty（異質充電折減實驗）

保留原本的多人共享充電 `e_charge / n`，但若同一 charger 範圍內同時出現不同 type 的 robot，充電效率再乘上一個小於 1 的係數。

$$\text{charge\_gain}_i = \frac{e_{\text{charge}}}{n} \times f$$

- 同 type 共站（○○ 或 △△）：$f = 1.0$
- 異 type 共站（○△ 或 ○○△）：$f = \text{heterotype\_charge\_factor}$

懲罰是對「混合佔據」本身，不是對 triangle。異質群體在高價值資源區的共存效率較差。

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 4 \
  --agent-types-mode observe --triangle-agent-id 2 \
  --charger-positions 2,2 \
  --heterotype-charge-mode local-penalty \
  --heterotype-charge-factor 0.7 \
  --num-episodes 1000000 \
  --wandb-run-name type_local_penalty
```

Step-level logging：每個 charger 每步記錄 `occupants`、`occupant_types`、`is_mixed_type`、`charge_factor_applied`，存在 replay JSON 的 `charger_log` 欄位。

Eval replay JSON 會記錄每個 agent 的 type：
```json
{ "agent_types": { "robot_0": "circle", "robot_2": "triangle", ... },
  "heterotype_charge_mode": "local-penalty",
  "heterotype_charge_factor": 0.7 }
```

---

## 分析工具

### 存活率分析（`analysis/survival_analysis.py`）

```bash
python analysis/survival_analysis.py \
  --model-dir ./models/p1_6x6/episode_50000 \
  --env-n 6 --num-robots 2 \
  --robot-0-energy 20 --robot-1-energy 100 \
  --charger-positions 3,3 --exclusive-charging --no-dust \
  --max-steps 1000 --num-episodes 500 \
  --scripted-robots 0 --random-start-robots 0 \
  --output-dir ./analysis/output/p1_6x6
```

輸出：`death_step_hist.png`、`death_pos_heatmap.png`、`avg_survival_heatmap.png`

### W&B 行為分析（`analysis/analyze_wandb_behavior.py`）

從 W&B 雲端抓資料，產生 7 張行為分析圖表：

```bash
python analysis/analyze_wandb_behavior.py \
  --run-id xldcqgew --run-name p4_4x4_shared
```

輸出：存活率、episode reward、攻擊 heatmap、r0 vs r1 對稱性、弱者抵抗、kill timeline。

---

## 完整 CLI 參數

### 環境

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--env-n` | 3 | 地圖大小 (n×n) |
| `--num-robots` | 4 | 機器人數量 |
| `--robot-{0..3}-energy` | 100/100/100/100 | 各 robot 初始能量（robot-0 預設 1000）|
| `--robot-{0..3}-speed` | 1 | 各 robot 每 timestep 行動回合數 |
| `--charger-positions` | None（四角）| 充電座位置 `"y1,x1;y2,x2"` |
| `--robot-start-positions` | None（四角）| 起始位置 `"y0,x0;y1,x1"` |
| `--e-move` | 1 | 移動消耗 |
| `--e-charge` | 1.5 | 充電回復 |
| `--e-collision` | 3 | 碰撞傷害 |
| `--e-boundary` | 50 | 撞牆傷害 |
| `--max-episode-steps` | 500 | 每 episode 最大步數 |
| `--charger-range` | 1 | 充電範圍：`0`=僅該格, `1`=3×3 |
| `--exclusive-charging` | False | 場上有對手時充電無效 |
| `--energy-cap` | None | 能量上限（None=無上限，允許超充） |
| `--e-decay` | 0.0 | 每步被動能量衰減（所有存活 robot） |
| `--no-dust` | False | 停用灰塵系統 |

### Robot 模式

| 參數 | 說明 |
|------|------|
| `--scripted-robots ID` | 固定 STAY，不訓練 |
| `--random-robots ID` | 純隨機動作，不訓練 |
| `--safe-random-robots ID` | 不撞牆的隨機走，不訓練 |
| `--flee-robots ID` | 啟發式逃離，不訓練 |
| `--random-start-robots ID` | 每 episode 隨機起點 |
| `--load-model-dir PATH` | 載入 checkpoint 接續訓練 |

### Agent Type（實驗用）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--agent-types-mode` | `off` | `off`=padding 0, `observe`=填入 type |
| `--triangle-agent-id` | None | 哪個 robot 是 triangle（None=全 circle）|
| `--charger-range` | 1 | 充電範圍：`0`=只有站在充電座上，`1`=3×3（預設）|

### Heterotype Charging Penalty（異質充電折減）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--heterotype-charge-mode` | `off` | `off`=不折減, `local-penalty`=異類共站折減 |
| `--heterotype-charge-factor` | 1.0 | 折減係數（0.7=打七折, 0.1=打一折）|

### 訓練

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num-episodes` | 10000 | 訓練 episode 數 |
| `--num-envs` | **256** | 並行環境數量 |
| `--batch-env` | **True** | 全 numpy 批次環境（`--no-batch-env` 關閉）|
| `--use-torch-compile` | **True** | torch.compile 加速（`--no-use-torch-compile` 關閉）|
| `--n-step` | 1 | N-step Return 展開步數 |
| `--gamma` | 0.99 | 折扣因子 |
| `--batch-size` | 128 | 批次大小 |
| `--memory-size` | 100000 | Replay Buffer 容量 |
| `--lr` | 0.0001 | 學習率 |
| `--target-update-frequency` | 1000 | Target network 更新頻率 |
| `--save-frequency` | 1000 | Checkpoint 儲存頻率 |
| `--per` | False | 啟用 Prioritized Experience Replay |
| `--eval-after-training` | **True** | 訓練後自動評估（`--no-eval-after-training` 關閉）|

### 輸出

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--wandb-mode` | offline | `online` / `offline` / `disabled` |
| `--wandb-run-name` | 自動 | run 名稱，也決定 save 目錄 |
| `--save-dir` | `./models` | 模型儲存根目錄 |

---

## Replay JSON 格式

```json
{
  "config": {
    "grid_size": 6,
    "num_robots": 2,
    "charger_positions": [[3,3]],
    "agent_types_mode": "observe",
    "agent_types": { "robot_0": "circle", "robot_1": "triangle" },
    "robot_initial_energies": { "robot_0": 100, "robot_1": 20 },
    "parameters": { "e_move": 1, "e_collision": 10, "e_charge": 2 }
  },
  "steps": [
    {
      "step": 42,
      "actions": { "robot_0": "STAY", "robot_1": "RIGHT" },
      "rewards": { "robot_0": 0.0, "robot_1": 0.5 },
      "robots": {
        "robot_0": { "position": [1, 1], "energy": 110, "is_dead": false }
      },
      "sub_steps": [...],
      "events": [...]
    }
  ]
}
```

Replay 視覺化控制鍵：

| 按鍵 | 功能 |
|------|------|
| `SPACE` | 暫停 / 播放 |
| `←` / `→` | 上一步 / 下一步 |
| `↑` / `↓` | 加速 / 減速 |
| `S` | 切換 sub-step 顯示 |
| `R` | 重置到開頭 |
| `Q` | 離開 |

---

## 依賴套件

```bash
pip install torch numpy gymnasium wandb pygame
pip install opencv-python  # 可選，replay 錄影用
```
