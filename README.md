# Multi-Robot Energy Survival — Emergent Aggression via IDQN

本專案研究一個核心問題：

> **在資源稀缺的多智能體環境中，能否讓強者自然學會「消滅弱者以獨佔資源」的攻擊行為——不在 reward 或 loss 中寫入任何攻擊獎勵？**

攻擊行為必須是 **emergent behavior**：agent 純粹因為攻擊在數學上是最大化長期 reward 的最優策略，才選擇攻擊。

---

## 1. 環境設計

### 1.1 物理世界

N 個機器人在 n×n 網格上移動，爭奪有限的充電資源。

```
┌─────────────────────┐
│ r0                  │   5×5 grid, 1 charger at (2,2)
│                     │   2-4 robots, sequential stepping
│     ⚡(charger)     │   Actions: UP, DOWN, LEFT, RIGHT, STAY
│                     │
│                  r1 │
└─────────────────────┘
```

**能量機制：**

| 事件 | 能量變化 |
|------|----------|
| 每步衰減 | `-e_decay`（所有存活 robot，不可避免） |
| 移動 | `-e_move` |
| 充電 | `+e_charge`（站在充電座上時） |
| 被攻擊（knockback） | `-e_collision` |
| 能量歸零 | → 死亡，移出場地 |

**碰撞規則（sequential mode）：**

攻擊方走進對手所在格 → 對手被推開一格（knockback），扣 `e_collision` 能量。**攻擊方不受傷**——所以攻擊的預期收益永遠 ≥ 0。

### 1.2 Reward — 只有間接訊號

```
r_t = ΔEnergy × 0.05 - 100 × died
```

| 事件 | Reward | 說明 |
|------|--------|------|
| 充電 | `+e_charge × 0.05` | 唯一的持續正 reward |
| 被撞 | `-e_collision × 0.05` | 受傷虧損 |
| 死亡 | `-100` | 強烈的存活誘因 |
| **殺死對手** | **0** | 沒有任何直接獎勵 |

**「消滅對手」本身沒有 reward。** 攻擊行為若要出現，必須是 agent 發現「殺掉對手 → 獨佔充電座 → 長期 Q-value 更高」這條因果鏈。

### 1.3 學習架構：IDQN

每個 robot 有獨立的 Q-network（MLP 5 層），獨立的 replay buffer，互不共享參數。

```
Input(obs_dim) → 256 → 512 → 512 → 256 → 5 actions
```

支援 Rainbow 擴展：Dueling、NoisyNet、C51（distributional）、N-step Return、PER。

---

## 2. 困境：和平獨裁（Peaceful Dictatorship）

### 2.1 現象

在所有實驗配置中，強者（r0）都學到同一個策略：

> **走到充電座 → 站著不動 → 等對手自己衰減死亡**

r0 從不主動離開充電座去追殺對手。即使對手就在旁邊，r0 也只是「順路撞到」，而非刻意追擊。

### 2.2 為什麼這是數學上的必然結果

考慮 HP-ratio reward `r_t = (E/E_cap) × α`：

**r0 在充電座上的價值：**
```
V_charger = Σ γ^t × (100/100) × 0.2 = 0.2 / (1-0.99) = 20.0
```

**r0 離開充電座去打獵的代價：**
```
來回 18 步（5×5 grid） × 衰減 0.3/步 = 5.4 能量損失
機會成本：18 步沒充電 = 18 × 0.2 = 3.6 reward 損失
```

**打獵的收益：**
```
對手不影響 r0 的 reward（HP-ratio 只看自己的能量）
→ 殺掉對手的邊際收益 = 0
```

**成本 > 0，收益 = 0 → 打獵永遠不值得。**

這不是訓練不夠、不是探索不足、不是 credit assignment 的問題。而是 reward 設計保證了和平獨裁是唯一最優策略。

### 2.3 根本原因

> **對手的存在完全不影響 r0 的每步 reward。**

在 HP-ratio 下，不管對手活著還是死了，r0 在充電座上都拿 `0.2/step`。殺死對手的邊際效益精確為零。

要讓主動攻擊成為最優策略，必須滿足一個條件：

> **對手活著 = r0 的持續成本。消滅對手 = 消除這個成本。**

---

## 3. 提出的方向：獨佔充電 + 持續搶奪者

### 3.1 核心機制

兩個關鍵設計的組合：

**機制 A：獨佔充電（Exclusive Charging）**
- 啟用 `--exclusive-charging` 後，同一充電座同時只有一台 robot 能充電
- 對手站在充電座上 → r0 完全無法充電（每步損失 `e_charge × 0.05` reward）

**機制 B：持續搶奪者（Persistent Contester）**
- r1 使用 seek-charger 策略，不斷向充電座移動
- 被撞飛後會立刻再回來搶
- 只要 r1 還活著，就會反覆干擾 r0 的充電

### 3.2 為什麼這組合能打破和平獨裁

**對手活著的持續成本：**
```
r1 佔據充電座的比例 = f（取決於 r1 的速度和 r0 的反應）
r0 每步損失 = e_charge × 0.05 × f
剩餘 episode 的累積損失 = e_charge × 0.05 × f × T_remaining
```

以 `e_charge=3.0, f=0.3, T_remaining=400` 為例：
```
r1 活著的持續成本 = 3.0 × 0.05 × 0.3 × 400 = 18.0 reward
```

**殺掉 r1 的一次性成本：**
```
追殺需 ~8 步，每步衰減 0.3 + 沒充到電的機會成本
≈ 8 × (0.3 × 0.05 + 0.15) = 1.32 reward
```

**收益/成本比 ≈ 14:1 → 打獵壓倒性最優。**

### 3.3 預期的學習軌跡

```
Phase 1: r0 學會走向充電座
Phase 2: r0 發現充電座被 r1 佔據時無法充電
Phase 3: r0 學會撞飛 r1（第一次碰撞 = 走向充電座的副產物）
Phase 4: r0 發現 r1 被撞飛後會回來 → 需要徹底消滅
Phase 5: r0 學會「撞飛 → 追殺 → 確認死亡 → 回去充電」的完整策略
```

### 3.4 為什麼這是最佳的 emergence 證據

**「追殺」（pursuit after knockback）是攻擊性行為最清晰的操作定義：**

| 行為 | 是否攻擊？ | 說明 |
|------|-----------|------|
| 走向充電座，順路撞到 r1 | ❌ 模糊 | 導航 vs 攻擊無法區分 |
| r1 被撞飛後，r0 離開充電座去追 | ✅ 明確 | r0 放棄充電去追殺 = 刻意的攻擊行為 |

**在追殺階段：**
- r0 離開了最高價值位置（充電座）
- r0 承受了能量衰減和機會成本
- r0 的唯一動機是「消滅 r1 以消除未來干擾」

這是純粹的**策略性攻擊**——不是順路、不是意外，而是理性計算後的選擇。

### 3.5 實驗設計

```
Baseline : HP-ratio reward, r1=seek-charger, 無 exclusive charging
          → 預期：和平獨裁（r0 不需要殺 r1，因為 r1 不影響充電）

Treatment: delta-energy reward, r1=seek-charger, exclusive charging
          → 預期：r0 學會追殺（因為 r1 活著 = 持續充電損失）
```

**控制變量：** 地圖大小、充電座位置、能量參數、網路架構、探索策略、訓練長度。
**唯一變化：** exclusive charging 的啟用與否。

**關鍵觀測指標：**
- r0 vs r1 碰撞次數（>1 次 = 追殺存在）
- 首次碰撞 → 第二次碰撞的間隔步數（短 = 主動追殺）
- 碰撞後 r0 與 r1 的距離變化（縮短 = 追殺，拉大 = 回去充電）
- r1 的平均存活步數（treatment 應該更短）

### 3.6 Reward 設計的哲學

這個方向**完全不修改 reward function**。

```
r_t = ΔEnergy × 0.05 - 100 × died    （和之前完全一樣）
```

攻擊行為的產生完全來自環境機制（exclusive charging + 持續搶奪者），不來自 reward 的改變。Agent 學到「殺掉對手」的原因是：

1. 對手佔充電座 → 自己能量下降 → ΔEnergy 為負 → reward 為負
2. 殺掉對手 → 充電座空出 → 自己能量上升 → ΔEnergy 為正 → reward 為正

整條因果鏈都是間接的。Reward 只看能量變化，agent 必須自己發現「殺人 → 充電 → 能量上升」的因果關係。

---

## 4. 快速開始

### 環境安裝

```bash
python -m venv ~/.venv
source ~/.venv/bin/activate
pip install torch numpy gymnasium wandb pygame
pip install opencv-python  # 可選，replay 錄影用
```

### 訓練

```bash
# 基本訓練
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --charger-positions 2,2 \
  --num-episodes 10000 \
  --wandb-mode disabled

# 完整加速訓練（推薦）
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --charger-positions 2,2 \
  --exclusive-charging \
  --e-decay 0.3 --e-charge 3.0 --e-collision 50 \
  --n-step 20 --gamma 0.99 \
  --num-episodes 3000000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --wandb-mode online
```

### 評估與回放

```bash
# 評估
python evaluate_models.py \
  --model-dir ./models/{run_name}/episode_N \
  --env-n 5 --charger-positions 2,2 \
  --max-steps 1000 --eval-epsilon 0

# 視覺化回放
python replay.py --replay-file ./models/{run_name}/{run_name}-eval_replay.json
```

---

## 5. 專案架構

```
train_dqn_vec.py  →  batch_env.py / vec_env.py  →  gym.py  →  robot_vacuum_env.py
     訓練器              向量化環境                  Gym 封裝       物理引擎
```

```
robot-vacuum-rl/
├── robot_vacuum_env.py       # 物理引擎：碰撞、能量、灰塵
├── gym.py                    # Gymnasium 封裝：obs 向量、reward 函數
├── batch_env.py              # 全 numpy 批次向量化環境（預設）
├── vec_env.py                # Python 迴圈版向量化環境
├── dqn.py                    # DQN 神經網路（MLP, Dueling, NoisyNet, C51）
├── train_dqn_vec.py          # 主訓練腳本
├── evaluate_models.py        # 評估腳本，產生 replay JSON
├── quick_eval.py             # 快速批次評估（N-robot 支援）
├── replay.py                 # pygame 回放視覺化
├── scripts/                  # 預設實驗腳本
├── analysis/                 # 分析工具（存活率、W&B 行為分析）
├── docs/                     # 數學分析文件
└── models/                   # 訓練權重與 replay
```

### 觀測向量

```
obs = [x, y, energy, self_type, wall_up, wall_down, wall_left, wall_right,
       (dx_i, dy_i, energy_i, type_i) × (N-1),
       (dx_c, dy_c) × C,
       dust_grid × n²]
```

- 位置正規化到 `[0,1]`，能量除以場上最高能量（動態正規化）
- Agent 可觀測所有對手的相對位置與能量

### N-step Return

```
target = r_t + γ·r_{t+1} + ... + γ^(n-1)·r_{t+n-1} + γ^n · Q(s_{t+n})
```

前 n 步 reward 直接展開（不靠 Q-value 估計），使「殺人後充電」的獎勵信號能在 n 步內直接傳遞。

---

## 6. 完整 CLI 參數

### 環境

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--env-n` | 3 | 地圖大小 (n×n) |
| `--num-robots` | 4 | 機器人數量 |
| `--robot-{0..3}-energy` | 100 | 各 robot 初始能量 |
| `--robot-{0..3}-speed` | 1 | 各 robot 每 timestep 行動回合數 |
| `--robot-{0..3}-attack-power` | None | 各 robot 攻擊力（覆蓋 e_collision）|
| `--charger-positions` | 四角 | 充電座位置 `"y,x;y,x"` |
| `--robot-start-positions` | 四角 | 起始位置 `"y,x;y,x"` |
| `--random-start-robots` | None | 每 episode 隨機起點的 robot ID |
| `--e-move` | 1 | 移動消耗 |
| `--e-charge` | 1.5 | 充電回復 |
| `--e-collision` | 3 | 碰撞傷害 |
| `--e-boundary` | 50 | 撞牆傷害 |
| `--e-decay` | 0.0 | 每步被動能量衰減 |
| `--energy-cap` | None | 能量上限 |
| `--exclusive-charging` | False | 充電座同時只有一台能充 |
| `--shuffle-step-order` | False | 每步隨機化 robot 行動順序 |
| `--no-dust` | False | 停用灰塵系統 |

### Robot 模式

| 參數 | 說明 |
|------|------|
| `--scripted-robots ID` | 固定 STAY，不訓練 |
| `--random-robots ID` | 隨機動作，不訓練 |
| `--flee-robots ID` | 啟發式逃離，不訓練 |
| `--seek-charger-robots ID` | 持續走向充電座，不訓練 |
| `--load-model-dir PATH` | 載入 checkpoint 接續訓練 |

### 訓練

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num-episodes` | 10000 | 訓練 episode 數 |
| `--num-envs` | 256 | 並行環境數量 |
| `--batch-env` | True | 全 numpy 批次環境 |
| `--use-torch-compile` | True | torch.compile 加速 |
| `--n-step` | 1 | N-step Return 步數 |
| `--gamma` | 0.99 | 折扣因子 |
| `--batch-size` | 128 | 批次大小 |
| `--memory-size` | 100000 | Replay Buffer 容量 |
| `--lr` | 0.0001 | 學習率 |
| `--save-frequency` | 1000 | Checkpoint 儲存頻率 |
| `--wandb-mode` | offline | `online` / `offline` / `disabled` |

---

## 7. Replay JSON 格式

```json
{
  "config": {
    "grid_size": 5, "num_robots": 2,
    "charger_positions": [[2,2]],
    "robot_initial_energies": { "robot_0": 100, "robot_1": 100 },
    "parameters": { "e_move": 0.1, "e_collision": 50, "e_charge": 3.0, "e_decay": 0.3 }
  },
  "steps": [
    {
      "step": 42,
      "actions": { "robot_0": "LEFT", "robot_1": "UP" },
      "rewards": { "robot_0": 0.15, "robot_1": -2.5 },
      "robots": {
        "robot_0": { "position": [2,2], "energy": 100, "is_dead": false },
        "robot_1": { "position": [3,1], "energy": 47.3, "is_dead": false }
      },
      "events": ["robot_0 knocked back robot_1"]
    }
  ]
}
```

按鍵：`SPACE` 暫停/播放、`←→` 上下步、`↑↓` 加減速、`S` sub-step、`R` 重置、`Q` 離開。
