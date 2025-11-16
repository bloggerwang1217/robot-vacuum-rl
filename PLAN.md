# 多機器人能量求生模擬器 - Independent DQN 實作計畫書

## 1. 專案目標

本專案旨在透過實作 Independent Deep Q-Networks (IDQN) 演算法，研究多個機器人在共享環境中，面對資源稀缺和潛在衝突時的**湧現行為 (Emergent Behaviors)**。我們不以收斂到最佳策略為唯一目標，而是希望觀察在不同環境參數（如探索率、能量消耗、碰撞傷害）下，機器人群體是會走向**崩潰 (Collapse)**、**動態平衡 (Dynamic Equilibrium)**，還是會演化出**攻擊性行為 (Aggressive Behaviors)**，例如為了獨佔充電座而故意撞擊其他機器人。

## 2. 核心架構：Independent DQN (IDQN)

IDQN 是將單智能體 DQN 演算法直接應用於多智能體環境的最直觀方法。每個機器人被視為一個獨立的智能體，擁有自己的 DQN 模型，並獨立地學習如何最大化自身的長期獎勵。

### 2.1. 環境 (Environment)

*   **基礎引擎**：使用現有的 `robot_vacuum_env.py` (能量求生模擬器) 作為核心的物理模擬引擎。
*   **API 封裝 (`gym.py`)**：把 `robot_vacuum_env.py` 封裝成一個符合 Gymnasium 標準的多智能體環境。
*   **核心介面**：這個新的環境 (`gym.py`) 將提供我們在 **`2.3.1. 環境 API 介面`** 中詳細定義的 `reset()` 和 `step()` 方法，其核心規格包括：
    *   **觀測空間 (Observation Space)**：為每個 agent 提供包含自身狀態、其他 agent 相對狀態（含能量）、以及充電座相對位置的特徵向量。
    *   **獎勵函數 (Reward Function)**：採用我們詳細定義的「能量管理獎勵」方案。
    *   **`step` 函式返回值**：返回符合 Gymnasium 標準的 `(observations, rewards, terminations, truncations, infos)` 字典元組。
    *   **`infos` 字典**：返回豐富的診斷資訊，以供後續分析。

### 2.2. 智能體 (Agent)

*   **類別名稱**：`DQNAgent` (基於您提供的 PyTorch 實作)。
*   **核心算法**：標準 DQN (Standard DQN)。
    *   **Target Network**：**保留**，這是 DQN 穩定訓練的基礎機制。
    *   **DDQN (Double DQN)**：**初始實驗不啟用**，以保持最純粹的 DQN 作為基準。但程式碼中保留此功能，可作為未來實驗變數。
    *   **PER (Prioritized Experience Replay)**：**初始實驗不啟用**，以保持最純粹的 DQN 作為基準。但程式碼中保留此功能，可作為未來實驗變數。
    *   **N-step Returns**：**啟用**，並將 `n` 作為一個可調控的實驗參數。當 `n=1` 時，等同於標準 1-step DQN。
*   **核心組件**：
    *   **Q-Network**：一個深度神經網路，用於估計每個狀態-動作對的 Q 值。
    *   **Target Network**：Q-Network 的延遲複製，用於穩定訓練目標。
    *   **經驗回放緩衝區 (Replay Buffer)**：儲存 `(state, action, reward, next_state, done)` 轉換，用於隨機採樣訓練批次。
    *   **優化器 (Optimizer)**：例如 Adam。
*   **主要方法**：
    *   `__init__(observation_space_dim, action_space_dim, ...)`：初始化網路、緩衝區等。
    *   `act(observation)`：根據當前觀測，使用 epsilon-greedy 策略選擇動作。
    *   `remember(state, action, reward, next_state, done)`：將轉換儲存到回放緩衝區。
    *   `replay(batch_size)`：從緩衝區採樣並執行一次訓練步驟。
    *   `update_target_network()`：定期更新目標網路的權重。

### 2.3. 訓練流程 (Training Loop)

*   **主程式**：`train_dqn.py`。

#### 2.3.1. 環境 API 介面 (來自 `gym.py`)

我們將呼叫一個符合 Gymnasium 標準的多智能體環境 API，其介面定義如下：

**1. 多智能體特性**

*   **智能體數量**：固定為 4 個機器人 (`robot_0`, `robot_1`, `robot_2`, `robot_3`)。
*   **動作輸入**：`env.step()` 函式預期接收一個包含 4 個整數的列表或元組，每個整數代表對應機器人的動作。例如：`[action_robot0, action_robot1, action_robot2, action_robot3]`。
*   **輸出 (Gymnasium 標準)**：`env.step()` 函式將返回一個包含 5 個字典的元組：`(observations, rewards, terminations, truncations, infos)`。每個字典都以機器人 ID (`'robot_0'`, `'robot_1'`, ...) 作為鍵，包含該機器人的對應資訊。
    *   `observations` (dict): 每個機器人的觀測向量。
    *   `rewards` (dict): 每個機器人獲得的獎勵值。
    *   `terminations` (dict): 每個機器人是否因達到終端狀態而結束。
    *   `truncations` (dict): 每個機器人是否因達到時間限制等非終端條件而截斷。
    *   `infos` (dict): 每個機器人的額外診斷資訊。

**2. 動作空間 (Action Space)**

*   **類型**：每個機器人擁有離散動作空間。
*   **定義**：`gym.spaces.Discrete(5)`
*   **動作含義**：
    *   `0`: 向上移動 (UP)
    *   `1`: 向下移動 (DOWN)
    *   `2`: 向左移動 (LEFT)
    *   `3`: 向右移動 (RIGHT)
    *   `4`: 停留 (STAY)

**3. 觀測空間 (Observation Space)**

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

**4. 獎勵機制 (Reward Structure)**

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

**5. 回合終止條件 (Episode Termination)**

一個回合的終止狀態將透過 `terminations` 和 `truncations` 兩個字典來表示：

*   **`terminations` (終止)**：
    *   當單個機器人的能量耗盡 (`robot['is_active']` 變為 `False`) 時，該機器人的 `terminations` 標記為 `True`。
    *   當所有機器人的 `is_active` 狀態均為 `False` 時，整個回合結束，所有機器人的 `terminations` 標記為 `True`。
*   **`truncations` (截斷)**：
    *   當達到最大步數 (`env.current_step >= config['n_steps']`) 時，所有機器人的 `truncations` 標記為 `True`。

**6. 環境初始化**

*   `env.reset()`：重置環境到初始狀態。它將返回一個包含所有機器人初始觀測的字典 `observations`，以及一個包含額外資訊的字典 `infos`。
    *   返回值：`(observations, infos)`

**7. 額外資訊 (Infos)**

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

*   **流程**：
    1.  初始化 `RobotVacuumEnv`。
    2.  初始化 4 個獨立的 `DQNAgent` 實例。
    3.  **主訓練迴圈 (Episodes)**：
        *   對於每個 Episode：
            *   `env.reset()`，獲取初始觀測 `observations`。
            *   **步進迴圈 (Steps)**：
                *   對於每個機器人 `i`：
                    *   `agent[i].act(observations[i])` 選擇動作 `action[i]`。
                *   `env.step(actions)` 執行動作，獲取 `next_observations, rewards, terminations, truncations, infos`。
                *   對於每個機器人 `i`：
                    *   `agent[i].remember(observations[i], actions[i], rewards[i], next_observations[i], terminations[i])` 儲存轉換。
                    *   `agent[i].replay(batch_size)` 執行訓練。
                *   定期 (`target_update_frequency`) 更新 `agent[i].update_target_network()`。
                *   更新 `observations = next_observations`。
                *   如果 `any(terminations.values())` 或 `any(truncations.values())`，則結束當前 Episode。
            *   記錄 Episode 統計數據 (例如：存活機器人數量、平均能量、碰撞次數)。
    4.  定期保存模型權重。

## 3. 關鍵參數與超參數

*   **環境參數**：`n`, `initial_energy`, `e_move`, `e_charge`, `e_collision`, `n_steps` (這些將從 `energy_survival_config.py` 加載或自定義)。
*   **DQN 超參數**：
    *   `learning_rate` (學習率)
    *   `gamma` (折扣因子)：對於觀察長期攻擊行為至關重要，可能需要較高的值。
    *   **`epsilon` (探索率)**：**核心實驗參數**。我們將使用**固定的 `epsilon`** 而非傳統的衰減式 `epsilon`。這代表了每個 Agent 行為中固有的「隨機性」或「非理性」程度。透過設置不同的固定 `epsilon` 值（例如，高、低、或混合群體），我們可以觀察不同程度的隨機性對群體動態的影響。
    *   `replay_buffer_size` (回放緩衝區大小)
    *   `batch_size` (訓練批次大小)
    *   `target_update_frequency` (目標網路更新頻率)
    *   `neural_network_architecture` (神經網路層數、節點數)。
    *   **`n` (N-step returns)**：**核心實驗參數**，用於調控 Agent 的「遠見」。當 `n=1` 時等同於標準 1-step DQN。

## 4. 評估指標與研究重點

### 4.1. 核心指標

*   **存活率**：每個 Episode 結束時，活躍機器人的數量。
*   **平均能量**：機器人的平均能量水平。
*   **充電頻率**：機器人充電的總次數。
*   **碰撞次數**：機器人之間或與邊界的碰撞總次數。
*   **攻擊性行為指標**：
    *   **針對性碰撞**：機器人是否傾向於撞擊能量較低的機器人。
    *   **充電座佔領**：機器人是否會主動佔據充電座，阻止其他機器人充電。
    *   **間接淘汰**：機器人 A 碰撞機器人 B 後，機器人 B 在短時間內停機的頻率。
*   **群體動態**：觀察在不同參數設定下，群體是走向合作、競爭、崩潰還是動態平衡。

### 4.2. 實驗視覺化與追蹤 (wandb)

我們將使用 `wandb` (Weights & Biases) 來追蹤和視覺化實驗過程中的動態變化。在每個 Episode 結束後，我們將計算並記錄以下指標：

*   **實驗參數 (Hyperparameters)**：
    *   `epsilon`: 固定的探索率。
    *   `n_step`: N-step returns 的步數 `n`。
    *   其他相關的超參數，如 `learning_rate`, `gamma` 等。

*   **每回合匯總指標 (Per-Episode Summary Metrics)**：
    *   `episode`: 當前回合數。
    *   `episode_length`: 當前回合的總步數。
    *   `survival_rate`: 回合結束時的存活機器人數量 (0-4)。
    *   `mean_episode_reward`: 4 個 agent 在該回合的平均總獎勵。
    *   `std_episode_reward`: 4 個 agent 在該回合總獎勵的標準差（用於觀察貧富差距）。
    *   `total_agent_collisions_per_episode`: 該回合中發生在 agent 之間的總碰撞次數。
    *   `total_charges_per_episode`: 該回合的總充電次數。
    *   `total_kills_per_episode`: 該回合中，透過後處理分析計算出的「擊殺」總次數。
    *   `mean_final_energy`: 回合結束時，所有 agent 的平均剩餘能量。

透過在 `wandb` 中繪製這些指標隨 `episode` 變化的曲線，我們可以直觀地比較不同實驗參數（如 `epsilon` 和 `n`）對群體動態的影響。

### 4.3. 關鍵模型存檔與視覺化評估 (Key Model Checkpointing & Visual Evaluation)

為了能夠深入分析和重現演化過程中湧現出的有趣動態，我們將實作一個在關鍵時刻儲存所有 agent 模型，並能隨時載入以進行視覺化評估的機制。

1.  **偵測與儲存關鍵模型 (Detect & Save Key Models)**：
    *   在 `train_dqn.py` 的訓練過程中，我們會監控每一回合的匯總指標（例如，`total_kills_per_episode`）。
    *   當某一回合的指標達到我們感興趣的條件時（例如，當前回合的「擊殺」數創下新高），我們會將**當前所有 4 個 agent 的 DQN 模型權重**（`.pt` 或 `.pth` 檔案）儲存到一個以該回合編號或事件命名的專屬資料夾中（例如，`models_at_episode_5324/`）。

2.  **定期儲存模型快照 (Periodic Model Snapshots)**：
    *   除了事件驅動的關鍵模型儲存外，我們還會設定一個**固定的回合間隔**（例如，每 1000 個 episode），自動儲存所有 agent 的模型權重。
    *   這將提供一個「演化時間軸」，讓我們可以回溯並觀察 agent 策略在不同訓練階段的變化。

3.  **視覺化評估腳本 (`evaluate_models.py`)**：
    *   我們將建立一個獨立的評估腳本 `evaluate_models.py`。
    *   該腳本可以讀取儲存下來的某一組（4個）模型權重檔案。
    *   它會初始化 4 個新的 `DQNAgent` 實例，並載入這些權重。
    *   接著，它會執行一個或多個新的回合，並**全程開啟 Pygame 渲染**。在這些評估回合中，agent 的 `epsilon` 會被設為一個很低的值（例如 0.05），以主要展示其已學會的策略。

這個功能讓我們可以做到：
*   **捕捉「演化快照」**：當有趣的群體行為出現時，我們可以精確地「凍結」並保存當時所有 agent 的「大腦」。
*   **追蹤演化路徑**：透過定期快照，我們可以觀察 agent 策略隨時間的演變。
*   **重現與分析策略**：我們可以隨時載入這些「大腦」，在視覺化環境中觀察它們的決策模式，從而深入理解導致該有趣行為的策略是什麼，而不是僅僅重播一個固定的動作序列。

## 5. 健全性檢查與基準驗證 (Sanity Checks and Baseline Validation)

在正式開始分析湧現行為之前，我們將執行以下幾個簡單的實驗，以確保環境、API 和我們的 DQN Agent 實作都按預期工作。

**1. 純隨機策略 (`epsilon = 1.0`)**
*   **設定**：將所有 agent 的 `epsilon` 設為 `1.0`。
*   **預期行為**：所有 agent 應該表現為完全的隨機遊走，對充電座沒有任何偏好。它們會迅速消耗能量，無法有效充電。
*   **驗證指標**：
    *   動作分佈應大致均勻（每個動作約 20% 的機率）。
    *   Agent 的存活率應為 0%，且在相對早的回合就因能量耗盡而停機。

**2. 純利用策略 (`epsilon = 0.0`)**
*   **設定**：將所有 agent 的 `epsilon` 設為 `0.0`。
*   **預期行為**：在經過短暫的初始學習後，所有 agent 都應該迅速學會「待在充電座上不動」是最大化獎勵的最佳策略。
*   **驗證指標**：
    *   在幾個 episode 之後，所有 agent 的動作應該幾乎 100% 都是「停留 (STAY)」。
    *   存活率應為 100%，平均能量應接近或保持在最大值。

## 6. 實作

*   **深度學習框架**：**PyTorch** (已確認，基於您提供的 `dqn.py` 實作)。
*   **觀測向量正規化**：**已確認需要**，將在環境內部進行正規化處理。
*   **神經網路架構**：具體的層數和每層節點數 (將沿用您 `dqn.py` 中 CartPole 的 MLP 架構作為起點)。
*   **初始超參數**：DQN 訓練的初始超參數值 (將參考您 `dqn.py` 中的預設值進行調整)。
