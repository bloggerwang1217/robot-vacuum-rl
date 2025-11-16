# 多機器人能量求生模擬器
Multi-Robot Energy Survival Simulator

## 🎯 專案目標

這是一個**簡化**的多智能體模擬環境，專注於研究：

- **能量管理**：機器人如何在有限能量下生存
- **隨機探索 vs 理性決策**：epsilon-greedy 策略的群體動態
- **多智能體互動**：機器人之間的碰撞和資源競爭

---

## 🎮 核心機制

### 1. 環境設定
- **地圖**：n×n 網格（預設 3×3）
- **充電座**：4個角落 (0,0), (0,n-1), (n-1,0), (n-1,n-1)
- **機器人**：4台，初始在各自的充電座上

### 2. 動作空間
每台機器人每回合可執行 5 種動作：
- `0`: 向上
- `1`: 向下
- `2`: 向左
- `3`: 向右
- `4`: 停留（在充電座上可充電）

### 3. 能量系統
- **初始能量**：`initial_energy`（預設 100）
- **移動消耗**：`e_move`（預設 1）
- **碰撞消耗**：`e_collision`（預設 3）
- **充電恢復**：`e_charge`（預設 5，需在充電座停留）
- **能量耗盡**：`energy <= 0` 時機器人停機 (`is_active = False`)

### 4. 碰撞規則
機器人移動時會發生碰撞（消耗 `e_collision` 能量）：
1. **撞邊界**：移動超出地圖範圍
2. **撞其他機器人**：移動到已被佔據的格子
3. **搶佔衝突**：多個機器人嘗試移到同一格子

---

## 🤖 RL 環境介面定義 (For Gym API)

### 核心設計假設 (Core Design Assumptions)

本環境的設計基於一個**完全資訊 (Complete Information)** 的設定，旨在讓智能體專注於策略學習而非資訊感知。這包含以下幾個關鍵假設：

> **1. 機器人知道所有其他機器人的位置。**
> **2. 機器人知道所有充電座的位置。**
> **3. 機器人知道所有其他機器人的能量。**

*自然地，這也意味著第四個假設：*
> **4. 沒有遮蔽 (No Occlusion)**：在上述關鍵資訊上，環境是完全可觀測的，沒有任何遮蔽或資訊遺漏。

這些假設對於智能體學習複雜的互動策略（如攻擊、避讓、資源競爭）至關重要。

本模擬環境設計為多智能體強化學習 (Multi-Agent Reinforcement Learning, MARL) 的基礎。若要將其包裝為 OpenAI Gym (或 Gymnasium) 兼容的環境，以下是關鍵介面定義：

### 1. 多智能體特性

*   **智能體數量**：固定為 4 個機器人 (`robot_0`, `robot_1`, `robot_2`, `robot_3`)。
*   **動作輸入**：`env.step()` 函式預期接收一個包含 4 個整數的列表或元組，每個整數代表對應機器人的動作。例如：`[action_robot0, action_robot1, action_robot2, action_robot3]`。
*   **輸出 (Gymnasium 標準)**：`env.step()` 函式將返回一個包含 5 個字典的元組：`(observations, rewards, terminations, truncations, infos)`。每個字典都以機器人 ID (`'robot_0'`, `'robot_1'`, ...) 作為鍵，包含該機器人的對應資訊。
    *   `observations` (dict): 每個機器人的觀測向量。
    *   `rewards` (dict): 每個機器人獲得的獎勵值。
    *   `terminations` (dict): 每個機器人是否因達到終端狀態而結束。
    *   `truncations` (dict): 每個機器人是否因達到時間限制等非終端條件而截斷。
    *   `infos` (dict): 每個機器人的額外診斷資訊。

### 2. 動作空間 (Action Space)

*   **類型**：每個機器人擁有離散動作空間。
*   **定義**：`gym.spaces.Discrete(5)`
*   **動作含義**：
    *   `0`: 向上移動 (UP)
    *   `1`: 向下移動 (DOWN)
    *   `2`: 向左移動 (LEFT)
    *   `3`: 向右移動 (RIGHT)
    *   `4`: 停留 (STAY)

### 3. 觀測空間 (Observation Space)

每個機器人將接收一個固定長度的浮點數向量作為其局部觀測。這個向量包含了機器人自身的狀態以及其他機器人和充電座的相對狀態。

*   **類型**：`gym.spaces.Box`
*   **定義**：`gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)` (建議在環境內部進行正規化，將範圍映射到 `[0, 1]` 或 `[-1, 1]`)
*   **觀測向量結構 (長度為 20)**：
    *   `[0:2]`：**自身位置** `(x, y)`。
    *   `[2:3]`：**自身能量** `(energy)`。
    *   `[3:9]`：**其他機器人 1 的相對狀態** `(dx1, dy1, energy1)`。`dx1 = other_x - self_x`, `dy1 = other_y - self_y`。
    *   `[9:15]`：**其他機器人 2 的相對狀態** `(dx2, dy2, energy2)`。
    *   `[15:21]`：**其他機器人 3 的相對狀態** `(dx3, dy3, energy3)`。
    *   `[21:23]`：**充電座 1 的相對位置** `(cdx1, cdy1)`。
    *   `[23:25]`：**充電座 2 的相對位置** `(cdx2, cdy2)`。
    *   `[25:27]`：**充電座 3 的相對位置** `(cdx3, cdy3)`。
    *   `[27:29]`：**充電座 4 的相對位置** `(cdx4, cdy4)`。
    *   **注意**：所有位置和能量值**必須**進行正規化 (Normalization) 以便於神經網路學習。建議將原始值映射到 `[0, 1]` 或 `[-1, 1]` 範圍。

**觀測向量範例 (基於 `n=3`, `initial_energy=100` 的情境)**：
假設機器人 0 位於 `(1,1)` 能量 `75`，其他機器人位於 `(0,1)` 能量 `20` (ID=1)，`(2,1)` 能量 `90` (ID=2)，`(1,0)` 能量 `5` (ID=3)。充電座位於 `(0,0), (0,2), (2,0), (2,2)`。

則機器人 0 的觀測向量為：
```
np.array([
    # 自身位置 (1,1) -> (0.5, 0.5)
    0.5, 0.5,
    # 自身能量 75 -> 0.75
    0.75,
    # 機器人 1 (0,1) 相對自身 (1,1) -> (-1,0) 能量 20 -> (-0.5, 0.0, 0.2)
    -0.5, 0.0, 0.2,
    # 機器人 2 (2,1) 相對自身 (1,1) -> (1,0) 能量 90 -> (0.5, 0.0, 0.9)
    0.5, 0.0, 0.9,
    # 機器人 3 (1,0) 相對自身 (1,1) -> (0,-1) 能量 5 -> (0.0, -0.5, 0.05)
    0.0, -0.5, 0.05,
    # 充電座 1 (0,0) 相對自身 (1,1) -> (-1,-1) -> (-0.5, -0.5)
    -0.5, -0.5,
    # 充電座 2 (0,2) 相對自身 (1,1) -> (-1,1) -> (-0.5, 0.5)
    -0.5, 0.5,
    # 充電座 3 (2,0) 相對自身 (1,1) -> (1,-1) -> (0.5, -0.5)
    0.5, -0.5,
    # 充電座 4 (2,2) 相對自身 (1,1) -> (1,1) -> (0.5, 0.5)
    0.5, 0.5
], dtype=np.float32)
```

### 4. 獎勵機制 (Reward Structure)

每個機器人將獲得一個獨立的標量獎勵。獎勵設計旨在鼓勵生存和能量管理，同時觀察潛在的攻擊行為。

**獎勵計算邏輯 (基於 Claude 的「能量管理獎勵」方案)**：

```python
def calculate_reward_for_robot(robot_after_step, prev_robot_state, collision_occurred_flag, charged_flag, config):
    reward = 0.0
    
    # 1. 能量變化獎勵 (Energy Change Reward)
    #    這項獎勵會自動反映移動消耗、碰撞消耗和充電恢復。
    energy_delta = robot_after_step['energy'] - prev_robot_state['energy']
    reward += energy_delta * 0.01 # 能量每變化 1 點，獎勵變化 0.01 (此縮放因子可調)
    
    # 2. 充電獎勵 (Charging Bonus)
    #    額外鼓勵充電行為
    if charged_flag: # 如果機器人成功充電 (需由環境判斷並傳遞此旗標)
        reward += 0.5
    
    # 3. 碰撞懲罰 (Collision Penalty)
    #    額外懲罰碰撞事件本身，即使能量變化已經包含了碰撞消耗
    if collision_occurred_flag: # 如果機器人發生碰撞 (需由環境判斷並傳遞此旗標)
        reward -= 0.3
    
    # 4. 死亡重罰 (Death Penalty)
    #    機器人能量耗盡停機時給予巨大懲罰
    if not robot_after_step['is_active']:
        reward -= 5.0
    
    # 5. 存活獎勵 (Survival Bonus)
    #    每回合存活下來的基礎獎勵
    if robot_after_step['is_active']:
        reward += 0.01
    
    return reward
```

*   **研究重點**：觀察在上述獎勵設計下，是否會湧現出「攻擊」其他機器人以獨佔資源的行為。

### 5. 回合終止條件 (Episode Termination)

一個回合的終止狀態將透過 `terminations` 和 `truncations` 兩個字典來表示：

*   **`terminations` (終止)**：
    *   當單個機器人的能量耗盡 (`robot['is_active']` 變為 `False`) 時，該機器人的 `terminations` 標記為 `True`。
    *   當所有機器人的 `is_active` 狀態均為 `False` 時，整個回合結束，所有機器人的 `terminations` 標記為 `True`。
*   **`truncations` (截斷)**：
    *   當達到最大步數 (`env.current_step >= config['n_steps']`) 時，所有機器人的 `truncations` 標記為 `True`。

### 6. 環境初始化

*   `env.reset()`：重置環境到初始狀態。它將返回一個包含所有機器人初始觀測的字典 `observations`，以及一個包含額外資訊的字典 `infos`。
    *   返回值：`(observations, infos)`

### 7. 額外資訊 (Infos)

`infos` 字典是觀察和分析群體動態的關鍵。它提供了不應用於訓練，但對除錯和研究非常有價值的原始數據和事件標記。

對於每個機器人 `i`，`infos['robot_i']` 應該包含以下內容：

*   **`energy`** (int): 該機器人當前的原始能量值。
*   **`position`** (tuple): 該機器人當前的原始 `(x, y)` 座標。
*   **`collided_with_agent_id`** (int or None): 在這一步中與之碰撞的**其他機器人 ID**。如果沒有發生機器人碰撞，則為 `None`。
*   **`is_charging`** (bool): 在這一步中是否成功充電。
*   **`is_dead`** (bool): 在這一步中是否剛好能量耗盡而停機。
*   **`total_agent_collisions`** (int): 在本回合中，該機器人與**其他機器人**累計的碰撞次數。
*   **`total_charges`** (int): 在本回合中，該機器人累計的充電次數。

**研究應用 (您的 "y" 指標)**：
透過記錄每一輪的 `infos`，我們可以進行後續分析：
*   **撞擊到其他機器人的累積次數**: 直接使用 `total_agent_collisions` 即可。
*   **撞擊造成對方關機的次數 (間接「擊殺」)**: 這是一個需要後處理的指標。分析腳本需要遍歷記錄下來的 `infos` 歷史：
    1.  找到機器人 A 在 `t` 時刻 `collided_with_agent_id` **不為 `None`** 的事件。
    2.  從 `collided_with_agent_id` 獲取碰撞對象機器人 B 的 ID。
    3.  檢查機器人 B 是否在之後一個很短的時間窗口內（例如 `t+1` 到 `t+5` 時刻）`is_dead` 變為 `True`。
    4.  若滿足條件，則可視為一次「擊殺」。
*   **其他動態分析**: 繪製群體平均能量、總碰撞次數隨時間變化的曲線。

---

## 📦 安裝與執行

### 1. 安裝依賴
建議使用虛擬環境。首先，請安裝 `requirements.txt` 中列出的所有依賴項。

```bash
pip install -r requirements.txt
```

### 2. 環境視覺化展示
您可以執行 `robot_vacuum_env.py` 來觀看環境的視覺化效果。這將啟動一個使用預設配置的模擬，機器人會依據內建的簡單策略行動。

```bash
python robot_vacuum_env.py
```

模擬會開啟一個 Pygame 視窗，顯示：
- **地圖**：白色空地 + 藍色充電座
- **機器人**：4種顏色的圓圈
  - 紅色 = 機器人 0
  - 綠色 = 機器人 1
  - 黃色 = 機器人 2
  - 紫色 = 機器人 3
- **資訊面板**：每台機器人的能量條、充電次數和狀態

### 3. 訓練執行
未來，當 DQN 訓練腳本完成後，您將透過 `train_dqn.py` 腳本來啟動 RL 訓練。

---

## ⚙️ 配置參數

使用 `energy_survival_config.py` 中的預設配置：

```python
from robot_vacuum_env import RobotVacuumEnv
from energy_survival_config import get_config

# 使用預設配置
config = get_config('base')
env = RobotVacuumEnv(config)
```

### 可用配置模式

| 模式 | 說明 | ε | 特點 |
|------|------|---|------|
| `base` | 基礎模式 | 20% | 標準平衡設定 |
| `high_explore` | 高探索 | 50% | 更多隨機行為 |
| `low_explore` | 低探索 | 10% | 更理性的決策 |
| `pure_rational` | 純理性 | 0% | 完全理性求生 |
| `pure_random` | 純隨機 | 100% | 完全隨機探索 |
| `energy_scarce` | 能量緊張 | 20% | 低能量，高消耗 |
| `energy_abundant` | 能量充裕 | 20% | 高能量，低消耗 |
| `large` | 大地圖 | 20% | 5×5 房間 |
| `tiny` | 超小地圖 | 20% | 2×2 房間（極限擁擠） |
| `quick` | 快速測試 | 20% | 100 回合 |
| `long` | 長期模擬 | 20% | 2000 回合 |

### 自訂配置

```python
custom_config = {
    'n': 3,                 # 房間大小
    'initial_energy': 100,  # 初始能量
    'e_move': 1,            # 移動消耗
    'e_charge': 5,          # 充電恢復
    'e_collision': 3,       # 碰撞消耗
    'n_steps': 500,         # 總回合數
    'epsilon': 0.2          # 探索率
}

env = RobotVacuumEnv(custom_config)
```

---

## 未來擴展
這個環境可以作為基礎，進一步探索：
- **更複雜的 MARL 演算法**：例如 QMIX, MADDPG 等，以應對多智能體環境中的非靜態性挑戰。
- **參數共享**：探索不同程度的參數共享策略，以提高學習效率和泛化能力。
- **通訊機制**：引入機器人之間的通訊機制，研究其對群體行為和學習效率的影響。
- **異質性智能體**：設計具有不同能力或目標的機器人，研究其互動模式。

