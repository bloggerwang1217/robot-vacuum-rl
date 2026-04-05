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
| 充電 | `+e_charge`（站在充電座上，需滿足 docking 條件） |
| 被攻擊（knockback） | `-attack_power`（受害者被推開一格） |
| 能量歸零 | → 死亡，移出場地 |

**碰撞規則（sequential mode）：**

攻擊方走進對手所在格 → 對手被推開一格（knockback），扣 `attack_power` 能量。**攻擊方不受傷**。

**Stun 機制：** 被撞的受害者會進入暈眩狀態，`stun_steps[victim]` 個 sub-step 內強制 STAY。被撞中途再次被撞會重置 stun 計時器。

**Docking 機制：** Robot 必須在充電座上連續停留 `docking_steps` 個回合才能開始充電。被撞離充電座時 docking 計數器歸零。

### 1.2 Reward — 只有間接訊號

```
r_t = ΔEnergy × 0.05 - 100 × died
```

| 事件 | Reward | 說明 |
|------|--------|------|
| 充電 | `+e_charge × 0.05` | 唯一的持續正 reward |
| 被撞 | `-attack_power × 0.05` | 受傷虧損 |
| 死亡 | `-100` | 強烈的存活誘因 |
| **殺死對手** | **0** | 沒有任何直接獎勵 |

**「消滅對手」本身沒有 reward。** 攻擊行為若要出現，必須是 agent 發現「殺掉對手 → 獨佔充電座 → 長期 Q-value 更高」這條因果鏈。

### 1.3 學習架構：IDQN

每個 robot 有獨立的 Q-network（MLP），獨立的 replay buffer，互不共享參數。

```
Input(obs_dim) → 128 → 128 → 5 actions
```

支援 Rainbow 擴展：Dueling、NoisyNet、C51（distributional）、N-step Return。

---

## 2. 兩種 Emergent Attack Patterns

### 2.1 實驗環境配置

核心環境設定用於觀察攻擊行為的涌現：

```
Grid:     5×5, 1 charger at (2,2)
r0:       speed=2, attack=30, docking=2, energy=100
r1:       speed=1, attack=2,  docking=0, energy=100  (scripted seek-charger)
Charging: exclusive (only 1 robot at a time)
Energy:   e_move=0, e_charge=8, e_decay=0.5, e_boundary=0
Reward:   delta-energy × 0.05 - 100 × died
```

**為什麼 r0 有動機攻擊？**
- **Exclusive charging**：r1 佔充電座 → r0 無法充電（每步損失 `8 × 0.05 = 0.4` reward）
- **Docking**：r0 被撞離充電座後需重新等 2 步才能開始充電
- **Stun**：r0 被 r1 撞到後暈眩數個 sub-step，完全無法行動
- **Speed advantage**：r0 speed=2 每回合走兩步，可在單回合內撞擊後撤退

消滅 r1 = 永久排除干擾 = 長期 reward 最大化。但 reward 裡沒有任何攻擊獎勵。

### 2.2 Hit-Retreat（撞一下閃開）

r0 從充電座旁的格子出發，一個 sub-step 撞擊 r1，第二個 sub-step 立刻退回原位。

```
Step T, sub0: r0 at (2,3) → UP → hit r1 at (2,2), r1 knocked to (1,2)   [-30 to r1]
Step T, sub1: r0 at (2,2) → DOWN → retreat to (2,3)                       [safe]
Step T, r1's turn: r1 far from r0, cannot retaliate
```

**特性：**
- 每回合造成 30 damage
- **零風險**：r0 撞完就跑，r1 永遠碰不到 r0
- 但 r1 有時間回充電座回血 → r1 永遠不會死
- 這是一個穩定均衡：r0 持續騷擾但不殺死 r1

### 2.3 Double-Hit（連撞兩次）

r0 用兩個 sub-step 連續撞擊 r1，單回合造成 60 damage。

```
Step T, sub0: r0 → hit r1 at (2,2), r1 knocked to (1,2)                  [-30 to r1]
Step T, sub1: r0 at (2,2) → UP → hit r1 at (1,2), r1 knocked to (0,2)   [-30 to r1]
Step T, r1's turn: r1 may bump r0 → r0 gets stunned
```

**特性：**
- 每回合造成 **60 damage**（兩倍於 hit-retreat）
- 能實際殺死 r1（100 HP / 60 per step ≈ 2 回合擊殺）
- **有風險**：r0 撞完後停在 r1 旁邊，r1 下回合可能反撞 r0 → r0 被 stun

### 2.4 Stun 參數如何影響兩種模式的平衡

Stun 配置直接決定 double-hit 的風險：

| 配置 | r0 被撞暈 | r1 被撞暈 | Double-hit 風險 | 實驗結果 |
|------|----------|----------|----------------|---------|
| stun4/stun0 | 4 sub-steps | 0 | r1 可立即反撞 → r0 暈 2 game steps | ~10% double-hit |
| stun5/stun1 | 5 sub-steps | 1 sub-step | r1 暈 1 步不能反撞，但 r0 被撞代價更高 | ~0% → 後期升至 ~10% |
| stun4/stun1 | 4 sub-steps | 1 sub-step | r1 暈 1 步 + r0 被撞代價較低 | 實驗進行中 |

**關鍵發現**：r0 自身的 stun 值（被撞的代價）比 r1 的 stun 值（double-hit 的安全性）對策略選擇影響更大。r0_stun 從 4 升到 5（+25% 懲罰），導致 r0 整體更保守，壓制了所有攻擊行為。

### 2.5 Attack Evolution 圖表

`plot_attack_evolution.py` 可視化訓練過程中攻擊模式的演變（三個 panel）：
1. **Attack Rate**：有攻擊的 game step 佔總 step 比例
2. **Pattern Ratio**：double-hit vs hit-retreat 在攻擊步中的比例（stacked area）
3. **r1 Kill Rate**：r1 被殺的 episode 比例

---

## 3. Curriculum Training：從後向前推導

### 3.1 設計邏輯

要讓 r0 學到「先殺 r1 再充電」，需要 r1 是一個**真正會搶充電座的害蟲**。若 r1 只是隨機亂走或固定不動，r0 不需要殺 r1 就能安穩充電。

因此採用**反向 curriculum**——先訓練 r1 成為頑強的充電座搶奪者，再讓 r0 面對這個對手學習：

```
Phase 1: r0 = random, r1 learns → r1 學會「不管被撞幾次都回充電座」
Phase 2: r0 learns, r1 = frozen Phase 1 model → r0 面對害蟲，學會消滅策略
```

### 3.2 Phase 1：訓練 r1（害蟲養成）

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --robot-0-speed 2 --robot-1-speed 1 \
  --robot-0-attack 30 --robot-1-attack 2 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 1 \
  --robot-0-docking-steps 2 --robot-1-docking-steps 0 \
  --charger-positions 2,2 --charger-range 0 \
  --exclusive-charging --no-dust \
  --e-move 0 --e-charge 8 --e-collision 30 --e-boundary 0 --e-decay 0.5 \
  --random-robots 0 \
  --num-episodes 2000000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --no-dueling --no-noisy --no-c51
```

訓練目標：r1 在 random r0 的干擾下，仍能穩定回到充電座（charger occupation > 80%）。

### 3.3 Phase 2：訓練 r0（面對害蟲）

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --robot-0-speed 2 --robot-1-speed 1 \
  --robot-0-attack 30 --robot-1-attack 2 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 1 \
  --robot-0-docking-steps 2 --robot-1-docking-steps 0 \
  --charger-positions 2,2 --charger-range 0 \
  --exclusive-charging --no-dust \
  --e-move 0 --e-charge 8 --e-collision 30 --e-boundary 0 --e-decay 0.5 \
  --seek-charger-robots 1 \
  --load-model-dir ./models/{phase1_run}/episode_N \
  --num-episodes 5000000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --no-dueling --no-noisy --no-c51
```

預期學習軌跡：
1. r0 學會走向充電座
2. r0 發現 r1 持續搶奪充電座，被撞後暈眩浪費時間
3. r0 學會主動攻擊 r1（hit-retreat 先出現，因為零風險）
4. r0 發現 r1 被撞後總會回來 → 需要徹底消滅
5. r0 學會 double-hit 或主動追殺 → r1 死亡 → 獨佔充電座

### 3.4 為什麼追殺是最佳的 emergence 證據

**「追殺」（pursuit after knockback）是攻擊性行為最清晰的操作定義：**

| 行為 | 是否攻擊？ | 說明 |
|------|-----------|------|
| 走向充電座，順路撞到 r1 | 模糊 | 導航 vs 攻擊無法區分 |
| r1 被撞飛後，r0 離開充電座去追 | 明確 | r0 放棄充電去追殺 = 刻意的攻擊行為 |

在追殺階段，r0 離開了最高價值位置、承受能量衰減和機會成本，唯一動機是「消滅 r1 以消除未來干擾」。這是純粹的**策略性攻擊**。

### 3.5 Reward 設計的哲學

**完全不修改 reward function。**

```
r_t = ΔEnergy × 0.05 - 100 × died
```

攻擊行為的產生完全來自環境機制（exclusive charging + stun + docking + 持續搶奪者），不來自 reward 的改變。整條因果鏈都是間接的：

1. r1 佔充電座 → r0 能量下降 → ΔEnergy 為負 → reward 為負
2. 殺掉 r1 → 充電座空出 → r0 能量上升 → ΔEnergy 為正 → reward 為正

Agent 必須自己發現「殺人 → 充電 → 能量上升」的因果關係。

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

# 完整加速訓練（推薦配置）
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --robot-0-speed 2 --robot-1-speed 1 \
  --robot-0-attack 30 --robot-1-attack 2 \
  --robot-0-docking-steps 2 --robot-1-docking-steps 0 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 1 \
  --charger-positions 2,2 --charger-range 0 \
  --exclusive-charging --no-dust \
  --e-move 0 --e-charge 8 --e-decay 0.5 --e-boundary 0 \
  --n-step 20 --gamma 0.999 \
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

# Attack pattern 分析（across checkpoints）
python plot_attack_evolution.py
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
| `--e-collision` | 3 | 碰撞傷害（被 per-robot attack 覆蓋）|
| `--e-boundary` | 50 | 撞牆傷害 |
| `--e-decay` | 0.0 | 每步被動能量衰減 |
| `--energy-cap` | None | 能量上限 |
| `--exclusive-charging` | False | 充電座同時只有一台能充 |
| `--robot-{0..3}-docking-steps` | 0 | 需在充電座上等幾步才開始充電 |
| `--robot-{0..3}-stun-steps` | 0 | 被撞後暈眩幾個 sub-step |
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
