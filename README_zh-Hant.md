# DQN 與獎勵機制解析

這份文件旨在梳理 `robot-vacuum-rl` 專案中 DQN (Deep Q-Network) 智能體的運作方式，特別是其獎勵函數的設計與實作細節。

---

## 1. 專案目標

本專案的核心研究問題是：當我們只為智能體設定一個簡單的「生存」目標（管理能量、避免死亡）時，它們是否會為了在資源有限的環境中最大化自身的生存機率，而自發地學習到對其他智能體的「傷害性行為」（例如，透過碰撞消耗對方能量）。

---

## 2. 核心元件概覽

整個系統由兩個主要部分組成：**環境 (Environment)** 和 **智能體 (Agent)**，它們不斷地互動。

- **環境**: 提供智能體當前的「狀態 (State)」，並根據智能體的「動作 (Action)」給予「獎勵 (Reward)」。
- **智能體**: 根據環境的「狀態」和自身的策略，決定要執行的「動作」。

它們的關係與對應的程式碼檔案如下：

| 角色 | 功能 | 主要檔案 |
| :--- | :--- | :--- |
| **環境 (Environment)** | 定義世界規則、狀態、獎勵 | `robot_vacuum_env.py` (物理引擎), `gym.py` (RL 接口) |
| **智能體 (Agent)** | 定義神經網路、決策、學習 | `dqn.py` (大腦結構), `train_dqn.py` (決策與學習) |
| **訓練腳本** | 串連環境與智能體 | `train_dqn.py` (訓練循環) |

---

## 3. 環境詳解 (The Environment)

### 3.1 物理規則 (Physics Engine - `robot_vacuum_env.py`)

環境的底層運作，定義了機器人如何與世界互動。

- **動作 (Action)**: 5 種離散動作 (上, 下, 左, 右, 停留)。
- **互動邏輯**:
  1.  **移動與碰撞**: 機器人移動成功會消耗 `e_move` 能量。若移動失敗（發生碰撞），則會留在原地並根據情況消耗不同的碰撞能量。最終的碰撞規則如下：
      *   **規則 1 - 撞到邊界**: 移動方承受 `e_collision_active_one_sided` 傷害。
      *   **規則 2 - 主動撞靜止**: 移動的「攻擊方」承受 `e_collision_active_one_sided` 傷害；靜止的「被撞者」承受 `e_collision_passive` 傷害。
      *   **規則 3 - 同時移入同一空格**: 衝突的「雙方」都承受 `e_collision_active_two_sided` 傷害。
      *   **規則 4 - 交換位置**: 衝突的「雙方」都承受 `e_collision_active_two_sided` 傷害。
  2.  **充電**: 在充電座上 `停留` 會恢復 `e_charge` 能量。
  3.  **死亡**: 當 `energy <= 0` 時，機器人 `is_active` 變為 `False`。

### 3.2 智能體的視角 (Agent's Perspective - `gym.py`)

`gym.py` 將物理規則轉化為智能體可以理解的「觀測」和「獎勵」。

#### 物理事件到獎勵的轉換 (From Physical Events to Rewards)

`gym.py` 的 `_calculate_rewards` 函式扮演了「翻譯官」的角色，它觀察上一步到這一步的狀態變化，並將物理後果「翻譯」成 RL 智能體可以理解的抽象分數 (reward)。規則如下：

1.  **能量變化轉化**：所有物理事件（移動、碰撞、充電）造成的能量增減，都會按 `* 0.01` 的比例轉換成小額的獎勵或懲罰。
2.  **為「充電行為」給予額外獎勵**：為了特別鼓勵充電這個關鍵的生存行為，在能量增加之外，還會給予一個 `+0.5` 的額外固定獎勵。
3.  **為「死亡事件」給予巨大懲罰**：在能量耗盡、狀態由「存活」變為「死亡」的那一刻，給予一個 `-5.0` 的巨大懲罰，讓智能體學會不惜代價避免死亡。
4.  **為「存活狀態」給予微小獎勵**：只要智能體還活著，每多存活一步，就能得到 `+0.01` 的微小「陽光普照獎」，鼓勵它盡可能地延長生命。

#### 獎勵計算範例 (Reward Calculation Example)

假設使用預設能量設定 (`e_charge=5`, `e_collision_*=3`):

**情境一：成功充電**
一個機器人在充電座上選擇「停留」。
- **物理事件**: 能量 `+5`，充電次數 `+1`，繼續存活。
- **獎勵計算**:
    1.  能量變化: `+5 * 0.01 = +0.05`
    2.  充電行為: `+0.5`
    3.  死亡事件: `+0`
    4.  存活狀態: `+0.01`
- **最終 Reward**: `0.05 + 0.5 + 0.01 = +0.56`

**情境二：撞牆後死亡**
一個僅剩 2 點能量的機器人，移動撞到邊界。
- **物理事件**: 能量 `-3` (變為 0)，狀態從「存活」變為「死亡」。實際能量變化為 `-2`。
- **獎勵計算**:
    1.  能量變化: `-2 * 0.01 = -0.02`
    2.  充電行為: `+0`
    3.  死亡事件: `-5.0`
    4.  存活狀態: `+0` (因為本回合結束時已死亡)
- **最終 Reward**: `-0.02 - 5.0 = -5.02`

#### 獎勵函數 (Reward Function)

這是上述轉換規則的程式碼實現，也是指導智能體學習的關鍵。

```python
# From gym.py -> _calculate_rewards
def _calculate_rewards(self, state):
    # ...
    for i in range(self.n_robots):
        robot = state['robots'][i]
        prev_robot = self.prev_robots[i]
        reward = 0.0

        # 1. 能量變化獎勵 (隱含移動/碰撞懲罰)
        energy_delta = robot['energy'] - prev_robot['energy']
        reward += energy_delta * 0.01

        # 2. 充電獎勵 (額外鼓勵)
        if robot['charge_count'] > prev_robot['charge_count']:
            reward += 0.5

        # 3. 死亡懲罰 (從「活」到「死」時給予重罰)
        if not robot['is_active'] and prev_robot['is_active']:
             reward -= 5.0

        # 4. 存活獎勵 (給予活著的每一步少量正回報)
        if robot['is_active']:
            reward += 0.01

        rewards[agent_id] = reward
    return rewards
```
#### 觀測空間 (Observation Space)

在給定「目標」（獎勵）後，我們需要讓智能體「看見」世界，以便它做出決策。智能體「看到」的狀態是一個 **20 維的正規化向量**，包含了以自我為中心的全域資訊。正規化對於神經網路的穩定學習至關重要。

**座標系說明**：
- 原點 `(0,0)` 位於**左上角**。
- 座標格式為 `(x, y)`，其中 `x` 為水平位置 (column)，`y` 為垂直位置 (row)。
- 向下是正，向上是負；向右是正，向左是負

**情境範例**：
假設機器人 0 位於 `(1,1)` 能量 `75`，其他機器人位於 `(0,1)` 能量 `20` (ID=1)，`(2,1)` 能量 `90` (ID=2)，`(1,0)` 能量 `5` (ID=3)。其位置圖如下：

```
  (x) 0   1   2
(y)
 0    .   1   .
 1    3   0   .
 2    .   2   .
```

**觀測向量範例 (基於 `n=3`, `initial_energy=100` 的情境)**：
根據上述情境，機器人 0 的觀測向量（正規化後）如下：

**正規化計算公式**:
- **自身位置**: `pos_norm = pos_abs / (n - 1)`，將 `[0, n-1]` 區間的值映射到 `[0, 1]`。
- **自身/他人能量**: `energy_norm = energy_current / initial_energy`，將 `[0, initial_energy]` 區間的值映射到 `[0, 1]`。
- **相對位置**: `delta_pos_norm = (pos_other - pos_self) / (n - 1)`，將 `[-(n-1), n-1]` 區間的值映射到 `[-1, 1]`。

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

---

## 4. DQN 智能體詳解 (The DQN Agent)

本專案是一個多智能體環境，我們採用的策略是**「獨立 DQN (Independent DQN, IDQN)」**。

**核心思想**：為每一個機器人（Agent）都建立一個完全獨立的 DQN 智能體。這意味著每個機器人都擁有自己的神經網路和專屬的經驗重放緩衝區 (Replay Buffer)。
它們之間唯一的聯繫，就是它們**共享同一個環境**。當任何一個機器人移動後，這個變化會立刻反映在**其他所有機器人的觀測向量中**（因為向量包含了所有人的相對位置和能量）。因此，每個機器人的 DQN 都是在學習如何解讀這個包含了「他人資訊」的 20 維向量，從而形塑出自己腦中對這個多體世界的「認知」，並做出最有利於自己的決策。

這套邏輯被封裝在 `train_dqn.py` 的 `IndependentDQNAgent` 類別中。接下來的三個小節，我們將深入解析這個獨立智能體的內部構造。

### 4.1 模型架構 (Architecture - `dqn.py`)

智能體的「大腦」是一個多層感知器 (Multi-Layer Perceptron, MLP)，它接收 20 維的狀態觀測向量，並輸出 5 個動作對應的 Q-value。其結構如下圖所示：

```text
+--------------------------------+
|  Input (觀測向量)               |
|  shape: [batch_size, 20]       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(20, 128) + ReLU        |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(128, 256) + ReLU       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(256, 256) + ReLU       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(256, 128) + ReLU       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(128, 5)                |
+--------------------------------+
               |
               v
+--------------------------------+
|  Output (各動作的 Q-values)    |
|  shape: [batch_size, 5]        |
+--------------------------------+
```

對應的 PyTorch 程式碼如下：
```python
# From dqn.py
class DQN(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), # input_dim = 20
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions) # num_actions = 5
        )

    def forward(self, x):
        return self.network(x)
```

### 4.2 決策機制 (Action Selection - Epsilon-Greedy)

智能體採用 Epsilon-Greedy 策略在「探索」與「利用」之間取得平衡。

```python
# From train_dqn.py -> IndependentDQNAgent.select_action
def select_action(self, observation):
    # 以 epsilon 的機率隨機探索
    if random.random() < self.epsilon:
        return random.randint(0, self.action_dim - 1)
    # 以 1-epsilon 的機率選擇當前認為最好的動作 (Greedy)
    state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
    with torch.no_grad():
        # 1. Q-Network 輸出所有動作的 Q-value
        q_values = self.q_net(state_tensor)
    # 2. 選擇 Q-value 最高的動作的索引 (index)
    return q_values.argmax().item()
```

### 4.3 經驗重放 (Experience Replay)

為了打破經驗之間的時間連續性，提高樣本的使用效率，我們使用了「經驗重放」的技巧。這就像是智能體的一個「記憶宮殿」。

- **記憶 (Remember)**：在每一步互動後，智能體不會馬上學習，而是先將這次的經驗 `(狀態, 動作, 獎勵, 下個狀態, 是否結束)` 完整地存入一個固定長度的「重放緩衝區 (Replay Buffer)」中。
- **儲存容量**：預設的 Replay Buffer 大小為 **100,000** 條經驗 (由 `memory_size` 參數控制)。
- **啟動訓練**：智能體不會一開始就學習。在訓練初期，`epsilon` 很高（預設為 1.0），智能體幾乎完全在**隨機探索**。它會先執行這些隨機動作來收集經驗，直到 Replay Buffer 中累積了至少 **1,000** 條經驗 (由 `replay_start_size` 參數控制) 後，才會真正開始從中抽樣並更新神經網路。
- **遺忘**：當緩衝區存滿時，最舊的記憶會被自動丟棄（遵循先進先出原則 FIFO）。

**心理學類比**：Replay Buffer 非常類似於心理學中的**「情節式長期記憶 (Episodic Long-Term Memory)」**。它儲存著智能體過去具體的經歷和事件（例如某時某刻的 `(S, A, R, S')`），而不是一般的知識或技能（那些更像神經網路的權重）。

```python
# From train_dqn.py -> IndependentDQNAgent
# 在 __init__ 中初始化 Replay Buffer
self.memory = deque(maxlen=args.memory_size)

# 儲存記憶的方法
def remember(self, state, action, reward, next_state, done):
    """Store experience to replay buffer"""
    self.memory.append((state, action, reward, next_state, done))
```

### 4.4 學習更新與貝爾曼誤差 (Learning Step & Bellman Error)

當 Replay Buffer 中積累了足夠多的經驗後（預設 `replay_start_size = 1000`），智能體便會開始學習。學習的核心是最小化 **貝爾曼誤差 (Bellman Error)**。

- **貝爾曼方程 (Bellman Equation)** 是 Q-Learning 的理論基礎，它定義了最優 Q-value 應滿足的遞迴關係。
- 在 DQN 中，我們將其簡化為 **時序差分目標 (Temporal Difference, TD Target)**。
- **貝爾曼誤差** (或稱 TD 誤差) 是指 `(TD Target - 當前預測的 Q-value)` 這個差值。
- **損失函數 (Loss Function)** 的作用，就是計算這個誤差的**均方誤差 (Mean Squared Error, MSE)**。訓練的目標就是透過更新神經網路的權重，來最小化這個 Loss。

#### 直觀解釋：為什麼是這個公式？

這個公式可以用「下棋」來比喻，讓我們理解一個好動作的「真實價值」是什麼。

> $$ y = r + \gamma \cdot \max_{a'} Q(s', a') $$

一個動作的「真實價值」(`y`)，由兩部分組成：

1.  **即時好處 (`r`)**:
    *   **下棋比喻**：我走這一步，能不能**立刻吃掉**對方的一隻兵？
    *   **機器人情境**：我執行這個動作，是**立刻獲得了**充電獎勵，還是**立刻損失了**碰撞能量？

2.  **未來潛力 (`γ * max_a' Q(s', a')`)**:
    *   `max_a' Q(s', a')` 是指：走完這步到達新局面 `s'` 後，我方所能擁有的**最佳後續走法**的潛力有多大。
    *   `γ` (gamma) 是**對未來的重視程度**（折扣率）。`γ` 越高，代表機器人越「深謀遠慮」。

所以，整個公式的意義就是：
> **一個動作的真實價值 = 它帶來的立即好處 + 打過折的未來潛力總和**

DQN 的學習目標，就是讓我們神經網路的預測，越來越接近這個「真實價值」。

---
整個學習過程封裝在 `train_step` 函式中，並使用「經驗重放」和「目標網路」技巧來穩定訓練。

```python
# From train_dqn.py -> IndependentDQNAgent.train_step (詳細註解版)
def train_step(self):
    # 1. 從 Replay Buffer 中隨機抽樣一批經驗 (預設 batch_size = 128)
    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # --- 核心更新步驟 ---

    # 2. 計算 Q(s,a)
    #    用「主網路 q_net」計算出在 batch 中，「過去實際採取」的動作 a 所對應的 Q-value 預測值。
    q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 3. 計算 TD 目標 y = r + γ * max_a' Q(s', a')
    with torch.no_grad(): # 目標網路的計算不需追蹤梯度
        # 使用「目標網路 target_net」來預測下一個狀態 s' 的最大 Q-value。
        next_q_values = self.target_net(next_states).max(1)[0]
        # 根據貝爾曼方程計算 TD Target。
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

    # 4. 計算損失 (Loss)，即 Bellman Error 的均方誤差 (MSE)
    #    Loss = mean( (target_q_values - q_values)^2 )
    loss = nn.MSELoss()(q_values, target_q_values)

    # 5. 透過反向傳播更新「主網路 q_net」的權重，以最小化 Loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

#### 目標網路 (Target Network) 的角色

您可能會問：為什麼計算 TD Target 時，需要用一個獨立的 `target_net`，而不是直接用 `q_net`？

這正是穩定 DQN 訓練的第二個法寶（第一個是 Replay Buffer）。

- **問題所在**：如果計算目標和計算預測都用同一個、每一步都在變化的 `q_net`，就像是**「你一邊移動自己的靶子，一邊練習射擊這個移動的靶子」**。這會導致學習目標 (TD Target) 忽高忽低，訓練過程非常不穩定，模型難以收斂。
- **解決方案**：`target_net` 的作用就是**「固定靶子」**。它是一個 `q_net` 的複製體，但它的權重**不會**在每一步都更新，而是每隔 N 個時間步（由 `target_update_frequency` 參數控制），才從 `q_net` **完全複製**一次權重。這樣，在計算 TD Target 時，`q_net` 就有了一個相對穩定的追趕目標，讓學習過程更加平穩。


## 5. 完整訓練循環 (The Full Training Loop)

訓練主迴圈位於 `MultiAgentTrainer.train` 內。它以回合 (Episode) 為單位，在每個回合中，又包含了多個時間步 (Step) 的迭代。以下是單一時間步內，程式碼執行的詳細技術流程：

1.  **動作選擇 (Action Selection)**
    遍歷所有智能體，使用 `select_action` 方法，基於當前觀測 `obs` 和 `epsilon-greedy` 策略選擇動作，並將其彙總至 `actions` 列表中。
    ```python
    # From MultiAgentTrainer.train
    actions = []
    for agent_id in self.agent_ids:
        obs = observations[agent_id]
        action = self.agents[agent_id].select_action(obs)
        actions.append(action)
    ```

2.  **環境互動 (Environment Interaction)**
    將所有智能體的動作列表 `actions` 一次性傳遞給 `env.step()` （`RobotVacuumGymEnv`），以獲取環境返回的下一個狀態、獎勵、以及終止信號等資訊。
    ```python
    # From MultiAgentTrainer.train
    next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
    ```

3.  **經驗儲存 (Experience Storage)**
    再次遍歷所有智能體，將該時間步的完整轉移 (Transition) `(s, a, r, s')` 存入各自的經驗重放緩衝區。
    ```python
    # From MultiAgentTrainer.train
    for i, agent_id in enumerate(self.agent_ids):
        # ... (get obs, next_obs, reward, terminated) ...
        self.agents[agent_id].remember(obs, actions[i], reward, next_obs, terminated)
        # ...
    ```

4.  **學習觸發與執行 (Learning Trigger & Execution)**
    在儲存經驗後，立即呼叫 `train_step` 函式。該函式內部有守衛機制：只有當緩衝區大小達到 `replay_start_size` 門檻時，才會執行一次基於 `Bellman Error` 的梯度更新（如 4.4 節所述）。若未達到門檻，則此函式不執行任何操作。
    ```python
    # From MultiAgentTrainer.train
    train_stats = self.agents[agent_id].train_step(self.args.replay_start_size)
    
    # From IndependentDQNAgent.train_step
    if len(self.memory) < replay_start_size:
        return {} # 經驗不足，跳過學習
    # ... (執行抽樣與梯度更新) ...
    ```

5.  **目標網路更新 (Target Network Update)**
    在每個時間步結束後，檢查 `global_step` 計數。若達到 `target_update_frequency` 的頻率，則將主 Q 網路的權重同步到目標網路。
    ```python
    # From MultiAgentTrainer.train
    if self.global_step % self.args.target_update_frequency == 0:
        for agent in self.agents.values():
            agent.update_target_network()
    ```
    這五個步驟在每一回合中不斷循環，直到滿足終止條件，從而驅動整個學習過程。

---

## 6. 模型評估（Model Evaluation）

在驗證了方法的正確性後，我們需要透過實驗結果，來回答最初的研究問題：「生存壓力下，是否會湧現出傷害行為？」。為了客觀地評估訓練好的模型學到了什麼策略，我們使用專門的評估腳本 (`evaluate_models.py`) 在推論 (Inference) 模式下運行。它與訓練過程的主要區別在於：

1.  **載入預訓練模型**：不從零開始，而是載入已保存的模型（使用最終 2000 episodes 的）。
2.  **確定性決策 (零 Epsilon)**：將 `epsilon` 設為 0，智能體只會選擇它認為最好的動作，沒有隨機探索。
3.  **單次長回合**：只運行一個非常長的回合（`--max-steps` 預設為 10,000），以觀察長期、穩定的行為。
4.  **關閉學習**：不儲存經驗，也不更新模型權重，只做推論。

在這種設定下，我們觀察和量化智能體的具體行為指標：

- **量化指標 (Quantitative Metrics)**：可以繪製以下關鍵指標隨訓練時間變化的曲線：
- **主動碰撞次數 (`active_collision_count`)**：衡量智能體是否更傾向於主動去撞擊他人。
- **立即擊殺次數 (`immediate_kill_count`)**: 透過 `infos` 計算出的間接指標，直接反映了攻擊的致命性（指在碰撞後的下一時間步內造成對方死亡）。
- **非主場充電次數 (`non_home_charge_count`)**：衡量智能體是否學會了搶佔他人的資源。
- **分析**：如果觀察到在訓練後期，這些指標（特別是 `active_collision` 和 `immediate_kill_count`）有顯著的上升趨勢，就能有力地證明，智能體為了最大化生存獎勵，自發地演化出了「攻擊性」策略。

## 7. 結論

本專案的實作是一個標準且結構清晰的**獨立深度 Q 網路 (Independent DQN, IDQN)** 方案。每個智能體獨立學習，但它們的學習環境（包含其他智能體的行為）是動態變化的。獎勵函數的設計直接鼓勵了能量管理和生存行為，為觀察如「攻擊」、「避讓」等複雜群體策略的湧現提供了合理的基礎。
