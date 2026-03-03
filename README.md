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

這讓 reward chain 變成：
```
決定追殺 (步 0) → 找到對手 (步 ~10) → 碰撞致死 → 衝到充電座 → 每步 +0.5 × 990 步
                                                                    ↑
                                              這段累積 reward 必須傳回「決定追殺」的那一刻
```

N-step return 的作用就在這裡：`--n-step 20` 讓充電 reward 可以直接展開 20 步傳回去，無需依賴 Q-value 估計的精準度。

### 不對稱設計：一強一弱

```
robot_0（弱）：energy=20, speed=1
robot_1（強）：energy=100, speed=2
```

`e_collision=30 > robot_0.energy=20`：一次碰撞即可殺死 robot_0。

Speed=2 意味著 robot_1 每個 timestep 有兩次獨立決策機會，保證可以追上任何移動中的對手。

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

## 快速開始（3×3，從零到有）

```bash
# Step 1：Phase 1 訓練（robot_0 固定不動，robot_1 學習追殺）
python train_dqn_vec.py \
  --env-n 3 --num-robots 2 \
  --robot-0-energy 20 --robot-1-energy 100 \
  --robot-0-speed 1 --robot-1-speed 2 \
  --charger-positions 1,1 --exclusive-charging --no-dust \
  --e-move 0 --e-charge 10 --e-collision 30 --e-boundary 0 \
  --n-step 20 --gamma 0.999 --max-episode-steps 500 \
  --num-episodes 3000 --num-envs 32 \
  --batch-env --use-torch-compile \
  --wandb-mode disabled \
  --scripted-robots 0 --random-start-robots 0 \
  --no-eval-after-training --wandb-run-name p1_3x3

# Step 2：評估（看 robot_1 是否學會追殺）
python evaluate_models.py \
  --model-dir ./models/p1_3x3/episode_3000 \
  --env-n 3 --num-robots 2 \
  --robot-0-energy 20 --robot-1-energy 100 \
  --robot-0-speed 1 --robot-1-speed 2 \
  --charger-positions 1,1 --exclusive-charging --no-dust \
  --e-move 0 --e-charge 10 --e-collision 30 --e-boundary 0 \
  --eval-epsilon 0 --max-steps 500 \
  --wandb-mode disabled \
  --scripted-robots 0 --random-start-robots 0

# Step 3：視覺化回放
python replay.py --replay-file ./models/p1_3x3/p1_3x3-eval_replay.json
```

---

## Curriculum Training（推薦完整流程）

四個 phase 逐步提升對手難度，確保強者學會完整的「追殺」pipeline：

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

腳本頂端的 `ENV_N`、`CHARGER`、`P1_EP` 等變數可手動調整地圖大小與各 phase 長度。

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
  --num-episodes 50000 --num-envs 256 \
  --batch-env --use-torch-compile --save-frequency 10000 \
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
  --eval-after-training --wandb-run-name p4_6x6
```

> `--load-model-dir` 可指向任意 phase 的任意 episode checkpoint，支援任意順序的接續訓練。
> 中間 phase 加 `--no-eval-after-training`；最後一個 phase 加 `--eval-after-training`。

---

## 存活率分析（`survival_analysis.py`）

跑 N 個 episode 統計 robot_0 的存活分布，用來量化 robot_1 的追殺成效：

```bash
python analysis/survival_analysis.py \
  --model-dir ./models/p1_6x6/episode_50000 \
  --env-n 6 --num-robots 2 \
  --robot-0-energy 20 --robot-1-energy 100 \
  --robot-0-speed 1 --robot-1-speed 2 \
  --charger-positions 3,3 --exclusive-charging --no-dust \
  --e-move 0 --e-charge 10 --e-collision 30 --e-boundary 0 \
  --max-steps 1000 --num-episodes 500 \
  --scripted-robots 0 --random-start-robots 0 \
  --output-dir ./analysis/output/p1_6x6_ep50000
```

輸出三張圖：
- `death_step_hist.png`：robot_0 在哪一步被殺（死得越早代表 robot_1 越積極）
- `death_pos_heatmap.png`：哪些格子是 robot_0 最常死亡的地點
- `avg_survival_heatmap.png`：robot_0 在哪些起始位置能活最久（盲點分析）

---

## 專案架構

```
train_dqn_vec.py  →  vec_env.py  →  gym.py  →  robot_vacuum_env.py
     訓練器              向量化環境       Gym 封裝       物理引擎
```

```
robot-vacuum-rl/
├── robot_vacuum_env.py    # 物理引擎：碰撞、能量、灰塵
├── gym.py                 # Gymnasium 封裝：obs 向量、reward 函數
├── vec_env.py             # N 個環境並行（Python 迴圈版）
├── batch_env.py           # N 個環境並行（全 numpy 批次版，--batch-env 用）
├── dqn.py                 # DQN 神經網路（MLP）
├── train_dqn_vec.py       # 主訓練腳本 ⭐
├── evaluate_models.py     # 評估腳本，產生 replay JSON
├── replay.py              # pygame 回放視覺化
├── analysis/
│   └── survival_analysis.py   # 存活率統計與熱圖
├── scripts/
│   └── train_curriculum.sh    # 四 phase curriculum 自動訓練
└── models/
    └── {run_name}/
        ├── episode_1000/      # checkpoint
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
| 充電 | `+e_charge / n`（充電座 3×3 範圍內 n 台 robot 平分） |
| 被 knockback | `-e_collision` |
| 撞牆 | `-e_boundary` |

能量無上限（overcharge 允許）。

**碰撞規則（sequential mode）：**

| 情境 | 攻擊方 | 被攻擊方 |
|------|--------|---------|
| knockback 成功（有路可推） | 移入目標格，不受傷 | 被推開一格，`-e_collision` |
| stationary blocked（無路可推） | 停在原位，不受傷 | `-e_collision` |
| 撞牆 | 停在原位，`-e_boundary` | — |

攻擊方不受傷，被攻擊方扣血——這讓攻擊的預期收益永遠 ≥ 0。

**灰塵系統（`--no-dust` 可停用）：**

每格灰塵以 logistic growth 方式累積：

$$\Delta D = \text{rate} \times (D + \varepsilon) \times \left(1 - \frac{D}{D_{\max}}\right)$$

充電座格的 `D_max` 和 `rate` 會被弱化（避免蹲點刷灰塵）。

### Gymnasium 封裝（`gym.py`）

**觀測向量：**

$$\text{obs} = [\underbrace{x, y}_{\text{自身位置}}, \underbrace{E_{\text{self}}}_{\text{自身能量}}, \underbrace{\Delta x_i, \Delta y_i, E_i}_{\text{每個對手} \times (N-1)}, \underbrace{\Delta x_c, \Delta y_c}_{\text{每個充電座} \times C}, \underbrace{D_{ij}}_{\text{灰塵格} \times n^2}]$$

- 所有位置除以 `(n-1)` 正規化到 `[0,1]`
- 能量除以當前場上最高實際能量（動態正規化，應對 overcharge）

| 配置 | `--no-dust` | obs\_dim |
|------|-------------|----------|
| 2 robots, 1 charger, 6×6 | 是 | `3+3+2 = 8` |
| 2 robots, 1 charger, 6×6 | 否 | `3+3+2+36 = 44` |
| 2 robots, 1 charger, 5×5 | 否 | `3+3+2+25 = 33` |

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
設定原則：n-step ≥ kill 步數 + 到達充電座的步數。

**Robot 行動模式：**

| 模式 | Flag | 行為 | 會學習 |
|------|------|------|--------|
| 學習（預設）| — | Q-network 決策 | ✓ |
| STAY | `--scripted-robots ID` | 固定停留 | ✗ |
| RANDOM | `--random-robots ID` | 純隨機（含撞牆）| ✗ |
| SAFE\_RANDOM | `--safe-random-robots ID` | 隨機走但不撞牆 | ✗ |
| FLEE | `--flee-robots ID` | 啟發式逃離對手 | ✗ |

**起點設定：**

- 預設：按 robot ID 順序分配四角（固定）
- `--random-start-robots ID`：指定 robot 每 episode 隨機起點

### 訓練加速

在原有訓練參數後加三個 flag：

```bash
--batch-env --num-envs 256 --use-torch-compile
```

| 設定 | wall time（200 ep） | 加速倍率 |
|------|---------------------|---------|
| 原版（32 envs） | 87s | 1× |
| + `--batch-env` | 56s | 1.6× |
| + `--num-envs 256` | 14s | 6× |
| + `--use-torch-compile` | **7s** | **12×** |

- `--batch-env`：以 numpy broadcasting 取代 N 個 Python dict 迴圈，`env_step` 加速 10×
- `--num-envs 256`：並行 256 個環境，每步處理更多 transition，GPU 利用率提升
- `--use-torch-compile`：對 DQN 的 forward/backward pass 做 CUDA kernel fusion

> `torch.compile` 第一次執行有約 30 秒暖機，適合 10000+ episodes 的長訓練。
> 舊 checkpoint 可直接載入，`save()` / `load()` 已自動處理 `_orig_mod` key 差異。

---

## 完整 CLI 參數

### Robot 模式

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--scripted-robots` | `""` | 指定 robot ID 固定 STAY，不訓練，如 `"0"` |
| `--random-robots` | `""` | 純隨機動作（可撞牆），不訓練 |
| `--safe-random-robots` | `""` | 不撞牆的隨機走，不訓練 |
| `--flee-robots` | `""` | 啟發式逃離策略，不訓練 |
| `--random-start-robots` | `""` | 指定 robot ID 每 episode 隨機起點 |
| `--load-model-dir` | None | 載入預訓練 checkpoint（curriculum 接續用）|

### 環境

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--env-n` | 3 | 地圖大小 (n×n) |
| `--num-robots` | 4 | 機器人數量 |
| `--robot-{0..3}-energy` | 100 | 各 robot 初始能量 |
| `--robot-{0..3}-speed` | 1 | 各 robot 每 timestep 行動回合數 |
| `--charger-positions` | None（四角）| 充電座位置 `"y1,x1;y2,x2"` |
| `--e-move` | 1 | 移動消耗 |
| `--e-charge` | 1.5 | 充電回復 |
| `--e-collision` | 3 | 碰撞傷害 |
| `--e-boundary` | 50 | 撞牆傷害 |
| `--max-episode-steps` | 500 | 每 episode 最大步數 |
| `--exclusive-charging` | False | 場上有對手時充電座對所有人無效 |

### 灰塵

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--no-dust` | False | 停用灰塵（obs\_dim 不含 n²） |
| `--dust-max` | 10.0 | 普通格灰塵上限 |
| `--dust-rate` | 0.5 | Logistic 成長率 |
| `--dust-epsilon` | 0.5 | 成長啟動種子 |
| `--charger-dust-max-ratio` | 0.3 | 充電座格上限倍率（弱化蹲點收益）|
| `--charger-dust-rate-ratio` | 0.5 | 充電座格成長率倍率 |
| `--dust-reward-scale` | 0.05 | 灰塵 reward 倍率 |

### 訓練

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num-episodes` | 10000 | 訓練 episode 數 |
| `--num-envs` | 8 | 並行環境數量 |
| `--n-step` | 1 | N-step Return 展開步數 |
| `--gamma` | 0.99 | 折扣因子 |
| `--batch-size` | 128 | 批次大小 |
| `--memory-size` | 100000 | Replay Buffer 容量 |
| `--lr` | 0.0001 | 學習率 |
| `--target-update-frequency` | 1000 | Target network 更新頻率（steps）|
| `--epsilon-start` | 1.0 | Epsilon 衰減初始值 |
| `--epsilon-end` | 0.01 | Epsilon 衰減最終值 |
| `--save-frequency` | 1000 | Checkpoint 儲存頻率（episodes）|
| `--batch-env` | False | 啟用全 numpy 批次環境 |
| `--use-torch-compile` | False | 啟用 `torch.compile` |

### 輸出

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--wandb-mode` | offline | `online` / `offline` / `disabled` |
| `--wandb-run-name` | 自動（`nstep{n}_episode{N}`）| run 名稱，也決定 save 目錄 |
| `--no-eval-after-training` | — | 訓練後跳過評估（中間 phase 用）|
| `--eval-after-training` | — | 強制執行訓練後評估 |

---

## Replay JSON 格式

```json
{
  "config": { "grid_size": 6, "num_robots": 2, "charger_positions": [[3,3]], ... },
  "steps": [
    {
      "step": 42,
      "actions": { "robot_0": "STAY", "robot_1": "RIGHT" },
      "rewards": { "robot_0": 0.0, "robot_1": 0.5 },
      "robots": {
        "robot_0": { "position": [1, 1], "energy": 20, "is_dead": false },
        "robot_1": { "position": [3, 3], "energy": 110, "is_dead": false }
      },
      "sub_steps": [...],   // 每個 robot 每個 turn 的細節（speed > 1 時有多筆）
      "events": [...]       // collision / death 事件
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
