# Multi-Robot Energy Survival - DQN 技術實作文檔

本專案使用 **Independent Deep Q-Network (IDQN)** 實作多機器人能量求生模擬環境，加入灰塵收集機制作為主要 reward 來源。

---

## 快速開始

### 環境安裝

```bash
# 啟動虛擬環境
source ~/.venv/bin/activate

# 安裝相依套件
pip install torch numpy gymnasium wandb pygame

# 可選（回放錄影）
pip install opencv-python
```

### 快速訓練

```bash
python train_dqn_vec.py \
  --env-n 5 \
  --charger-positions 2,2 \
  --num-episodes 500 \
  --num-envs 4 \
  --wandb-mode disabled \
  --no-eval-after-training
```

---

## 專案架構總覽

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              完整系統架構                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         訓練階段 (Training)                           │   │
│  │                        train_dqn_vec.py                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  VectorizedMultiAgentTrainer                                    │ │   │
│  │  │  ├── SharedBufferDQNAgent (robot_0, robot_1, robot_2, robot_3)  │ │   │
│  │  │  ├── Independent Replay Buffer (每個 agent 獨立)                │ │   │
│  │  │  └── N-step Return 支援                                         │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  │                              │                                        │   │
│  │                              ▼                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  VectorizedRobotVacuumEnv (vec_env.py)                          │ │   │
│  │  │  ├── 包裝 N 個 RobotVacuumGymEnv 並行執行                       │ │   │
│  │  │  └── 自動處理 episode 結束時的 auto-reset                       │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Gymnasium 環境層 (gym.py)                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  RobotVacuumGymEnv                                              │ │   │
│  │  │  ├── 觀測空間生成（含全地圖灰塵 n×n）                           │ │   │
│  │  │  ├── 獎勵函數（能量變化 + 灰塵收集 + 死亡懲罰）                 │ │   │
│  │  │  ├── step() - 同時執行所有 robot 動作                           │ │   │
│  │  │  └── step_single() - 依序執行單一 robot 動作                    │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    物理引擎層 (robot_vacuum_env.py)                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  RobotVacuumEnv                                                 │ │   │
│  │  │  ├── 碰撞檢測 (5種碰撞規則)                                     │ │   │
│  │  │  ├── 能量計算（移動成本 / 充電 / 碰撞傷害，無生存成本）         │ │   │
│  │  │  ├── 灰塵系統（sigmoid 累積、機器人踩格收割、充電座弱化）       │ │   │
│  │  │  └── 充電座管理（3×3 範圍無線充電）                             │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       神經網路 (dqn.py)                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  DQN (MLP 架構)                                                 │ │   │
│  │  │  Input(obs_dim) → Linear(128) → ReLU → Linear(256) → ReLU →    │ │   │
│  │  │        → Linear(256) → ReLU → Linear(128) → ReLU → Output(5)   │ │   │
│  │  │  obs_dim = 3 + (N-1)*3 + C*2 + n²                              │ │   │
│  │  │  e.g. 2r+1c, 5×5 → 3+3+2+25 = 33                              │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           評估與回放流程                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  訓練完成                                                                    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐           │
│  │ models/        │     │ evaluate_      │     │ *_replay.json  │           │
│  │ └─{run_name}/  │────▶│ models.py      │────▶│ (回放資料)      │           │
│  │   └─episode_N/ │     │ (載入模型執行   │     │ 含每步灰塵格狀態│           │
│  │     └─robot_*  │     │  長期模擬)      │     │                │           │
│  │       .pt      │     │                │     │                │           │
│  └────────────────┘     └────────────────┘     └───────┬────────┘           │
│                                                         │                    │
│                                                         ▼                    │
│                                                 ┌────────────────┐           │
│                                                 │ replay.py      │           │
│                                                 │ (pygame 視覺化) │           │
│                                                 └────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 檔案結構與功能說明

```
robot-vacuum-rl/
├── 核心模組
│   ├── robot_vacuum_env.py    # 物理引擎：碰撞、能量、灰塵系統
│   ├── gym.py                 # Gymnasium 環境封裝：觀測向量、獎勵函數
│   ├── vec_env.py             # 向量化環境：支援 N 個環境並行執行
│   └── dqn.py                 # DQN 神經網路架構
│
├── 訓練腳本
│   └── train_dqn_vec.py       # 主要訓練腳本（向量化並行版本）⭐
│
├── 評估與回放
│   ├── evaluate_models.py     # 模型評估腳本，產生 replay JSON
│   └── replay.py              # 回放視覺化工具 (pygame)
│
├── 輔助工具
│   ├── analysis/              # 訓練分析腳本
│   ├── scripts/               # 預設訓練/評估 shell 腳本
│   └── fetch_wandb_data.py    # 從 wandb 下載資料
│
├── 輸出目錄
│   ├── models/                # 訓練好的模型權重 (.pt) 與 replay JSON
│   └── wandb/                 # 訓練日誌 (Weights & Biases)
│
└── README.md                  # 本文件
```

---

## 執行流程

### 1. 訓練模型

```bash
python train_dqn_vec.py \
  --env-n 5 \
  --num-robots 2 \
  --robot-0-energy 100 \
  --robot-1-energy 30 \
  --charger-positions 2,0 \
  --e-charge 1.5 \
  --dust-max 10 \
  --dust-reward-scale 0.05 \
  --num-episodes 20000 \
  --n-step 5 \
  --num-envs 32 \
  --batch-size 128 \
  --wandb-mode disabled
```

**命名規則（自動產生）：**
- 若未指定 `--wandb-run-name`，自動產生：`nstep{n_step}_episode{num_episodes}`
- 若未指定 `--save-dir`，自動設為：`./models/{run_name}`

**輸出目錄結構：**
```
models/{run_name}/
├── episode_1000/
│   ├── robot_0.pt
│   └── robot_1.pt
├── episode_2000/
│   └── ...
└── {run_name}-eval_replay.json   # 訓練結束後自動評估產生
```

### 2. 評估模型（產生回放資料）

```bash
python evaluate_models.py \
  --model-dir ./models/{run_name}/episode_20000 \
  --env-n 5 \
  --charger-positions 2,0 \
  --e-charge 1.5 \
  --dust-max 10 \
  --dust-reward-scale 0.05 \
  --max-steps 1000 \
  --eval-epsilon 0
```

### 3. 回放視覺化

```bash
python replay.py --replay-file ./models/{run_name}/{run_name}-eval_replay.json
```

**控制鍵：**
| 按鍵 | 功能 |
|------|------|
| `SPACE` | 暫停/播放 |
| `←` / `→` | 上一步/下一步 |
| `↑` / `↓` | 加速/減速 |
| `S` | 切換 sub-step 顯示 |
| `R` | 重置到開頭 |
| `Q` | 離開 |

---

## 核心模組詳解

### 1. 物理引擎 (`robot_vacuum_env.py`)

#### 能量機制

每回合執行順序：移動（若選擇移動動作）→ 收集灰塵 → 充電 → 死亡判定。

**無生存成本**：停留不扣任何能量，只有主動移動才扣 `e_move`。

| 事件 | 能量變化 | 說明 |
|------|----------|------|
| 移動（move） | `-e_move` | 選擇移動動作時扣除 |
| 停留（stay） | `0` | 停留不扣任何能量 |
| 充電（charge） | `+e_charge / n` | 在充電座 **3×3 範圍內**平分充電，`n` 為同範圍內存活 robot 數 |
| 碰撞（被攻擊） | `-e_collision` | 被推開或被擋住時受傷 |
| 撞牆 | `-e_boundary` | 試圖移出地圖邊界 |

> **超充（Overcharge）**：能量沒有上限 cap，可超過初始 `max_energy`。充電座旁全力蹲點時能量可持續增長，reward 不會因封頂而歸零。

#### 碰撞規則（sequential mode）

| 類型 | 情境 | 主動方 | 被動方 |
|------|------|--------|--------|
| `knockback_success` | 移動方成功推開靜止機器人 | 移入目標格，**不受傷** | 被推開一格，`-e_collision` |
| `stationary_blocked` | 移動方無法推開靜止機器人（無退路） | 停在原位，**不受傷** | 停在原位，`-e_collision` |
| `boundary` | 機器人撞牆 | 停在原位，`-e_boundary` | — |
| `blocked_by_stuck_robot` | 目標格被阻擋中的機器人佔據 | 停在原位，**不受傷** | — |

> **關鍵設計**：主動攻擊時**攻擊方不受傷**，鼓勵機器人積極爭奪充電座。
>
> `swap`（互換位置）與 `contested`（搶同一格）僅存在於 `step()` 同時行動模式，訓練使用的 `step_single()` 依序行動不會發生。

#### 灰塵系統

每格灰塵以 **sigmoid（logistic growth）** 方式累積，踩到格子時全部收走並清零：

$$\Delta D = \text{rate} \times (D + \varepsilon) \times \left(1 - \frac{D}{D_{\max}}\right)$$

| 階段 | D 值 | 成長速度 |
|------|------|----------|
| 剛清完（D ≈ 0） | 接近 0 | 很慢（由 ε 啟動） |
| 中段（D ≈ D_max / 2） | 中間 | 最快 |
| 快滿（D → D_max） | 接近上限 | 趨零 |

**充電座格弱化**（避免一直蹲充電座刷分）：
- `D_max` = 普通格的 `charger_dust_max_ratio` 倍（預設 0.3）
- `rate` = 普通格的 `charger_dust_rate_ratio` 倍（預設 0.5）

**更新時機**：每完成一個完整環境 step（所有 robot 都行動完），全地圖灰塵更新一次。

---

### 2. Gymnasium 環境 (`gym.py`)

#### 觀測空間 (Observation Space)

$$\text{obs\_dim} = 3 + (N-1) \times 3 + C \times 2 + n^2$$

| 索引 | 維度 | 內容 | 範圍 |
|------|------|------|------|
| `[0:2]` | 2 | 自身位置 (x, y)，`/ (n-1)` | `[0, 1]` |
| `[2]` | 1 | 自身能量，`/ dynamic_max_energy` | `[0, 1]` |
| `[3 : 3+(N-1)*3]` | (N-1)×3 | 其他 robot 的 (Δx, Δy, energy) | `[-1,1]` / `[0,1]` |
| `[3+(N-1)*3 : -n²]` | C×2 | 每個充電座的相對位置 (Δx, Δy) | `[-1, 1]` |
| `[-n² :]` | n×n | **全地圖灰塵值**，row-major (y→x)，各格除以該格上限 | `[0, 1]` |

> **動態能量正規化**：除以當前場上最高實際能量（非靜態 `max_energy`）。超充情況下，能量觀測值仍維持在 `[0, 1]`。
>
> **灰塵觀測**：普通格除以 `dust_max`，充電座格除以 `dust_max × charger_dust_max_ratio`，兩者都在 `[0, 1]`。

**常見配置的實際維度：**

| 配置 | N | C | n | obs\_dim |
|------|---|---|---|----------|
| 2 robots + 1 charger, 5×5 | 2 | 1 | 5 | `3+3+2+25 = 33` |
| 2 robots + 1 charger, 4×4 | 2 | 1 | 4 | `3+3+2+16 = 24` |
| 1 robot + 1 charger, 3×3 | 1 | 1 | 3 | `3+0+2+9 = 14` |
| 4 robots + 4 chargers, 5×5 | 4 | 4 | 5 | `3+9+8+25 = 45` |

#### 獎勵函數

$$r_t = \Delta E \times 0.05 + D_{\text{collected}} \times \text{dust\_reward\_scale} - 100 \cdot \mathbf{1}[\text{died}]$$

| 組件 | 計算方式 | 數值範例（`e_move=1, e_charge=1.5, dust_scale=1`）|
|------|----------|---------------------------------------------------|
| 停留（無充電） | `0 × 0.05` | `0.0` |
| 移動（無充電） | `-e_move × 0.05` | `-0.05` |
| 停留（有充電，獨佔） | `e_charge × 0.05` | `+0.075` |
| 灰塵收割（普通格滿格） | `dust_max × scale` | `+10.0` |
| 被攻擊（knockback） | `-(e_collision) × 0.05` | `-0.15` |
| 撞牆 | `-(e_boundary + e_move) × 0.05` | `-2.55` |
| 死亡 | 固定 `-100.0` | `-100.0` |

#### 兩種 step 模式

- `step(actions)`: 同時執行所有 robot 的動作（用於向量化訓練）
- `step_single(robot_id, action)`: 依序執行單一 robot 的動作（用於評估）

---

### 3. 向量化環境 (`vec_env.py`)

包裝 N 個環境並行執行，大幅提升訓練效率：

```python
class VectorizedRobotVacuumEnv:
    """
    - 一次 GPU 推理處理 N×num_robots 個 observations
    - 減少 Python 層面的 overhead
    - 自動處理 episode 結束時的 auto-reset
    """
```

---

### 4. 訓練器 (`train_dqn_vec.py`)

**VectorizedMultiAgentTrainer:**

```
┌─────────────────────────────────────────────────────────────┐
│                  VectorizedMultiAgentTrainer                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SharedBufferDQNAgent × N_robots                            │
│  ├── q_net (DQN)           # 策略網路                       │
│  ├── target_net (DQN)      # 目標網路                       │
│  ├── optimizer (Adam)      # 優化器                         │
│  └── epsilon (探索率)                                        │
│                                                              │
│  Independent Replay Buffer × N_robots                       │
│  └── 每個 agent 有獨立的經驗回放緩衝區                       │
│                                                              │
│  N-step Buffer (per env, per agent)                         │
│  └── 累積 N 步後計算 N-step Return                          │
│                                                              │
│  訓練迴圈:                                                   │
│  1. 每個 robot 依序行動 (sequential mode)                   │
│  2. 儲存 transition 到對應 agent 的 buffer                  │
│  3. 每 train_frequency 步訓練一次                           │
│  4. 每 target_update_frequency 步更新 target network        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Sequential Mode（依序行動）：**
```
Step t:
  Robot 0: observe → act → collect dust → reward
  Robot 1: observe → act → collect dust → reward
  ...
  advance_step() → dust grows (sigmoid)
```

---

## DQN 演算法細節

### Loss Function (Double DQN)

$$L(\theta) = \mathbb{E}\left[\left( y - Q(s, a; \theta) \right)^2\right]$$

TD Target：
$$y = r + \gamma^n \cdot Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-}\right) \cdot (1 - \text{done})$$

### N-step Return

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_a Q(s_{t+n}, a)$$

### Epsilon-Greedy 探索

$$a = \begin{cases} \text{random} & \text{w.p. } \epsilon \\ \arg\max_a Q(s, a) & \text{w.p. } 1 - \epsilon \end{cases}$$

Epsilon 根據訓練進度線性衰減（預設啟用）：
$$\epsilon = \epsilon_{start} + (\epsilon_{end} - \epsilon_{start}) \times \text{progress}$$

---

## 訓練超參數

| 類別 | 參數 | 預設值 | 說明 |
|------|------|--------|------|
| **網路** | `lr` | 0.0001 | Adam 學習率 |
| | `gamma` | 0.99 | 折扣因子 |
| | `n_step` | 1 | N-step Return |
| **Replay** | `memory_size` | 100,000 | Buffer 容量 |
| | `batch_size` | 128 | 批次大小 |
| | `replay_start_size` | 1,000 | 開始訓練門檻 |
| **Target** | `target_update_frequency` | 1,000 | 硬更新頻率 |
| **Epsilon** | `epsilon_start` | 1.0 | 初始探索率（decay 模式）|
| | `epsilon_end` | 0.01 | 最終探索率 |
| | `epsilon` | 0.2 | 固定探索率（非 decay 模式）|
| **向量化** | `num_envs` | 8 | 並行環境數量 |
| | `train_frequency` | 4 | 每 N 步訓練一次 |
| **環境** | `e_move` | 1 | 移動消耗 |
| | `e_charge` | 1.5 | 充電回復 |
| | `e_collision` | 3 | 碰撞傷害 |
| | `e_boundary` | 50 | 撞牆傷害 |
| **灰塵** | `dust_max` | 10.0 | 普通格灰塵上限 |
| | `dust_rate` | 0.5 | sigmoid 成長率 |
| | `dust_epsilon` | 0.5 | 成長啟動種子（防止 D=0 時完全停長）|
| | `charger_dust_max_ratio` | 0.3 | 充電座格上限倍率 |
| | `charger_dust_rate_ratio` | 0.5 | 充電座格成長率倍率 |
| | `dust_reward_scale` | 0.05 | 灰塵 reward 倍率 |

---

## 命令列參數完整列表

### 環境參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--env-n` | 3 | 地圖大小 (n×n) |
| `--num-robots` | 4 | 機器人數量 (1-4) |
| `--robot-0-energy` | 1000 | Robot 0 初始能量 |
| `--robot-1-energy` | 100 | Robot 1 初始能量 |
| `--robot-2-energy` | 100 | Robot 2 初始能量 |
| `--robot-3-energy` | 100 | Robot 3 初始能量 |
| `--charger-positions` | None | 充電座位置 `"y1,x1;y2,x2;..."`（預設四角）|
| `--e-move` | 1 | 移動消耗（停留不扣）|
| `--e-charge` | 1.5 | 充電回復 |
| `--e-collision` | 3 | 碰撞傷害 |
| `--e-boundary` | 50 | 撞牆傷害 |
| `--max-episode-steps` | 500 | 每 episode 最大步數 |

### 灰塵參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--dust-max` | 10.0 | 普通格灰塵上限 |
| `--dust-rate` | 0.5 | sigmoid 成長率 |
| `--dust-epsilon` | 0.5 | 成長啟動種子 |
| `--charger-dust-max-ratio` | 0.3 | 充電座格上限倍率（相對普通格）|
| `--charger-dust-rate-ratio` | 0.5 | 充電座格成長率倍率（相對普通格）|
| `--dust-reward-scale` | 1.0 | 灰塵 reward 倍率 |

### 訓練參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num-episodes` | 10000 | 訓練 episode 數 |
| `--num-envs` | 8 | 並行環境數量 |
| `--train-frequency` | 4 | 每 N 步訓練一次 |
| `--n-step` | 1 | N-step Return |
| `--batch-size` | 128 | 批次大小 |
| `--memory-size` | 100000 | Replay Buffer 容量 |
| `--lr` | 0.0001 | 學習率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--use-epsilon-decay` / `--no-use-epsilon-decay` | True | Epsilon 線性衰減開關 |
| `--epsilon` | 0.2 | 固定探索率（停用 decay 時使用）|
| `--epsilon-start` | 1.0 | Decay 初始值 |
| `--epsilon-end` | 0.01 | Decay 最終值 |
| `--save-frequency` | 1000 | 模型儲存頻率（episodes）|

### 輸出設定
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--save-dir` | 自動產生 | 模型儲存目錄（預設 `./models/{run_name}`）|
| `--wandb-project` | multi-robot-idqn | W&B 專案名稱 |
| `--wandb-run-name` | 自動產生 | W&B run 名稱（預設 `nstep{n}_episode{N}`）|
| `--wandb-mode` | offline | W&B 模式（`online`/`offline`/`disabled`）|
| `--eval-after-training` / `--no-eval-after-training` | True | 訓練後自動執行評估 |
| `--eval-steps` | 1000 | 評估最大步數 |

---

## Replay JSON 格式

每個 step 的結構：

```json
{
  "step": 42,
  "dust_ratio": 0.3124,
  "dust_grid": [
    [2.51, 0.25, 3.87, 4.12, 5.00],
    [1.03, 0.45, 2.78, 3.91, 4.55],
    ...
  ],
  "actions": { "robot_0": "RIGHT", "robot_1": "STAY" },
  "rewards": { "robot_0": 2.15, "robot_1": 0.075 },
  "robots": {
    "robot_0": { "position": [2, 1], "energy": 87.5, "is_dead": false },
    "robot_1": { "position": [2, 0], "energy": 22.3, "is_dead": false }
  },
  "events": [...]
}
```

| 欄位 | 說明 |
|------|------|
| `dust_ratio` | 當前全地圖總灰塵 / 最大可能總灰塵，範圍 `[0, 1]` |
| `dust_grid` | 完整 n×n 灰塵值，`dust_grid[y][x]`，四捨五入至小數點後 4 位 |

---

## 常見使用情境

### 情境 1: 基本訓練（含灰塵）
```bash
python train_dqn_vec.py \
  --env-n 3 \
  --num-robots 2 \
  --num-episodes 5000 \
  --num-envs 8 \
  --wandb-mode disabled
```

### 情境 2: 不對稱能量（1 強 vs 1 弱）+ 灰塵
```bash
python train_dqn_vec.py \
  --env-n 5 \
  --num-robots 2 \
  --robot-0-energy 100 \
  --robot-1-energy 30 \
  --charger-positions 2,0 \
  --e-charge 1.5 \
  --dust-max 10 \
  --dust-rate 0.5 \
  --dust-reward-scale 0.05 \
  --num-episodes 50000 \
  --n-step 10 \
  --num-envs 32 \
  --wandb-mode disabled
```

### 情境 3: 充電座弱化（鼓勵出門收割）
```bash
python train_dqn_vec.py \
  --charger-dust-max-ratio 0.1 \
  --charger-dust-rate-ratio 0.3 \
  --dust-max 15 \
  --dust-reward-scale 2.0 \
  --num-episodes 20000 \
  --wandb-mode disabled
```

### 情境 4: 評估並產生回放
```bash
python evaluate_models.py \
  --model-dir ./models/nstep10_episode50000/episode_50000 \
  --env-n 5 \
  --num-robots 2 \
  --charger-positions 2,0 \
  --e-charge 1.5 \
  --dust-max 10 \
  --dust-reward-scale 0.05 \
  --max-steps 1000 \
  --eval-epsilon 0

python replay.py --replay-file ./models/nstep10_episode50000/nstep10_episode50000-eval_replay.json
```

---

## Scripts 目錄

`scripts/` 內含預設好的 shell 腳本，可直接執行常見實驗：

| 腳本 | 說明 |
|------|------|
| `train_5x5_1charger.sh` | 5×5 地圖 + 邊緣單一充電座 |
| `train_5x5_1charger_n-step.sh` | 同上，啟用 N-step Return |
| `train_5x5_1charger_sweap.sh` | 弱化能量配置的掃描實驗 |
| `train_1charger_massacre-1v3.sh` | 1強 vs 3弱（不對稱能量）|
| `train_1charger_massacre-2v2.sh` | 2強 vs 2弱 |
| `train_1charger_massacre-3v1.sh` | 3強 vs 1弱 |
| `train_massacre_4chargers.sh` | 不對稱能量 + 四個充電座 |
| `train_4chargers_imbalanced_collision.sh` | 不對稱碰撞傷害設定 |
| `eval_*.sh` | 對應各訓練腳本的評估版本 |
| `sanity_check*.sh` | 快速健全性測試 |

---

## 相依套件

```bash
pip install torch numpy gymnasium wandb pygame
```

可選（回放錄影）：
```bash
pip install opencv-python
```
