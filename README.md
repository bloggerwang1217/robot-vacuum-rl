# Multi-Robot Energy Survival - DQN 技術實作文檔

本專案使用 **Independent Deep Q-Network (IDQN)** 實作多機器人能量求生模擬環境。

---

## 技術實作細節

### 1. 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│                      (train_dqn_vec.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐    actions     ┌─────────────────────────┐    │
│   │   Agent 0   │ ────────────▶  │                         │    │
│   │   Agent 1   │                │      Environment        │    │
│   │   Agent 2   │ ◀────────────  │  (robot_vacuum_env.py)  │    │
│   │   Agent 3   │  obs, reward   │       (gym.py)          │    │
│   └─────────────┘                └─────────────────────────┘    │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐                                               │
│   │    DQN      │                                               │
│   │  (dqn.py)   │                                               │
│   └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

| 模組 | 檔案 | 功能 |
|------|------|------|
| **物理引擎** | `robot_vacuum_env.py` | 碰撞檢測、能量計算、狀態更新 |
| **RL 介面** | `gym.py` | Gymnasium 封裝、觀測向量、獎勵函數 |
| **神經網路** | `dqn.py` | DQN 網路架構 |
| **訓練腳本** | `train_dqn.py`, `train_dqn_vec.py` | 訓練迴圈、Experience Replay |

---

### 2. DQN 網路架構

#### 2.1 網路結構 (MLP)

```
Input (obs_dim) → Linear(128) → ReLU → Linear(256) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Output (5 actions)
```

**輸入維度計算：**
$$\text{obs\_dim} = 3_{(self)} + (N-1) \times 3_{(others)} + 2 \times C_{(chargers)} + 4_{(valid\_moves)}$$

其中 $N$ = 機器人數量，$C$ = 充電座數量

**程式碼** ([dqn.py](dqn.py#L28-L42))：
```python
class DQN(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)  # num_actions = 5
        )

    def forward(self, x):
        return self.network(x)
```

#### 2.2 權重初始化

使用 **Kaiming Initialization** (He Initialization)，適用於 ReLU 激活函數：

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$$

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
```

---

### 3. 學習演算法

#### 3.1 Loss Function (Bellman Error)

採用 **Double DQN** 降低 Q 值高估問題：

$$L(\theta) = \mathbb{E}\left[\left( y - Q(s, a; \theta) \right)^2\right]$$

其中 TD Target：
$$y = r + \gamma^n \cdot Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-}\right) \cdot (1 - \text{done})$$

- $\theta$: Policy Network 參數
- $\theta^{-}$: Target Network 參數
- $n$: N-step 數（支援 N-step Return）
- $\gamma$: 折扣因子（預設 0.99）

**程式碼** ([train_dqn.py](train_dqn.py#L220-L250))：
```python
def train_step(self, replay_start_size):
    # Sample batch from replay buffer
    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Current Q-values: Q(s, a; θ)
    q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Double DQN: 用 q_net 選動作，用 target_net 估值
    gamma_n = self.gamma ** self.n_step
    with torch.no_grad():
        # Action selection: argmax_a' Q(s', a'; θ)
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        # Value evaluation: Q(s', a*; θ-)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + gamma_n * next_q_values * (1 - dones)
    
    # MSE Loss
    loss = nn.MSELoss()(q_values, target_q_values)
    
    # Backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

#### 3.2 N-step Return

支援 N-step 學習，累積 $n$ 步折現獎勵後再存入 Replay Buffer：

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_a Q(s_{t+n}, a)$$

**實作方式**：在 `remember()` 中累積 N 步後計算 N-step Return

```python
def remember(self, state, action, reward, next_state, done):
    if self.n_step == 1:
        self.memory.append((state, action, reward, next_state, done))
    else:
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step or done:
            # 計算 n-step return
            n_step_return = 0
            for idx, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** idx) * r
            
            # 存入 (s_0, a_0, G^n, s_n, done_n)
            start_state, start_action, _, _, _ = self.n_step_buffer[0]
            _, _, _, end_next_state, end_done = self.n_step_buffer[-1]
            self.memory.append((start_state, start_action, n_step_return, end_next_state, end_done))
```

---

### 4. 訓練穩定技術

#### 4.1 Experience Replay

打破資料時序相關性，提升樣本效率：

| 參數 | 值 | 說明 |
|------|-----|------|
| `memory_size` | 100,000 | Replay Buffer 容量 |
| `batch_size` | 128 | 每次取樣批次大小 |
| `replay_start_size` | 1,000 | 開始訓練的最小樣本數 |

#### 4.2 Target Network

穩定 TD Target，每 $N$ 步硬更新：

$$\theta^{-} \leftarrow \theta \quad \text{(every } N \text{ steps)}$$

```python
def update_target_network(self):
    self.target_net.load_state_dict(self.q_net.state_dict())
```

預設 `target_update_frequency = 1000`

#### 4.3 Epsilon-Greedy 探索

$$a = \begin{cases} \text{random} & \text{w.p. } \epsilon \\ \arg\max_a Q(s, a) & \text{w.p. } 1 - \epsilon \end{cases}$$

**Epsilon Decay**（指數衰減）：
$$\epsilon_{t+1} = \max(\epsilon_{end}, \epsilon_t \times \epsilon_{decay})$$

預設：$\epsilon_{start} = 1.0$，$\epsilon_{end} = 0.01$，$\epsilon_{decay} = 0.995$

---

### 5. 獎勵函數設計

#### 5.1 獎勵組成

獎勵函數將物理事件轉換為學習訊號 ([gym.py](gym.py#L120-L150))：

| 組件 | 計算方式 | 設計理念 |
|------|----------|----------|
| **能量變化** | $\Delta E \times 0.01$ | 將所有能量事件統一縮放 |
| **充電獎勵** | $+0.5$ | 額外鼓勵充電行為 |
| **死亡懲罰** | $-5.0$ | 強烈訊號避免死亡 |
| **存活獎勵** | $+0.01$ | 微小獎勵鼓勵延長生存 |

**程式碼**：
```python
def _calculate_rewards(self, state):
    for i in range(self.n_robots):
        robot = state['robots'][i]
        prev_robot = self.prev_robots[i]
        reward = 0.0

        # 1. 能量變化 (移動/碰撞/充電)
        energy_delta = robot['energy'] - prev_robot['energy']
        reward += energy_delta * 0.01

        # 2. 充電行為額外獎勵
        if robot['charge_count'] > prev_robot['charge_count']:
            reward += 0.5

        # 3. 死亡懲罰 (alive → dead 瞬間)
        if not robot['is_active'] and prev_robot['is_active']:
            reward -= 5.0

        # 4. 存活獎勵
        if robot['is_active']:
            reward += 0.01
```

#### 5.2 獎勵計算範例

**情境 1：成功充電**
- 物理事件：能量 +5，充電計數 +1
- 獎勵：$(+5 \times 0.01) + 0.5 + 0.01 = +0.56$

**情境 2：撞牆死亡**
- 物理事件：能量 -50（歸零），狀態 alive → dead
- 獎勵：$(-50 \times 0.01) + (-5.0) = -5.5$

---

### 6. 觀測空間 (Observation Space)

每個機器人獲得**正規化的相對視角觀測向量**：

| 區段 | 維度 | 內容 | 正規化方式 |
|------|------|------|------------|
| 自身位置 | 2 | $(x, y)$ | $/ (n-1)$ → $[0, 1]$ |
| 自身能量 | 1 | energy | $/ E_{initial}$ → $[0, 1]$ |
| 其他機器人 | $(N-1) \times 3$ | $(\Delta x, \Delta y, E)$ | 相對位置 $/ (n-1)$ → $[-1, 1]$ |
| 充電座位置 | $C \times 2$ | $(\Delta x, \Delta y)$ | 相對位置 $/ (n-1)$ → $[-1, 1]$ |
| 可移動方向 | 4 | (up, down, left, right) | $\{0, 1\}$ |

---

### 7. 碰撞物理規則

四種碰撞情境及對應傷害 ([robot_vacuum_env.py](robot_vacuum_env.py))：

| 規則 | 情境 | 傷害分配 |
|------|------|----------|
| **Rule 1** | 撞牆 | 移動方：`e_boundary` |
| **Rule 2** | 主動撞靜止 | 主動方：`e_collision_active`，被動方：`e_collision_passive` |
| **Rule 3** | 雙方同時移動到同一格 | 雙方各：`e_collision_two_sided` |
| **Rule 4** | 交換位置 | 雙方各：`e_collision_two_sided` |

---

### 8. 訓練超參數

| 類別 | 參數 | 預設值 | 說明 |
|------|------|--------|------|
| **網路** | `lr` | 0.0001 | Adam 學習率 |
| | `gamma` | 0.99 | 折扣因子 |
| | `n_step` | 1 | N-step Return |
| **Replay** | `memory_size` | 100,000 | Buffer 容量 |
| | `batch_size` | 128 | 批次大小 |
| | `replay_start_size` | 1,000 | 開始訓練門檻 |
| **Target** | `target_update_frequency` | 1,000 | 硬更新頻率 |
| **Epsilon** | `epsilon_start` | 1.0 | 初始探索率 |
| | `epsilon_end` | 0.01 | 最終探索率 |
| | `epsilon_decay` | 0.995 | 衰減率 |
| **環境** | `e_move` | 1 | 移動消耗 |
| | `e_charge` | 5 | 充電回復 |
| | `e_collision` | 3 | 碰撞傷害 |
| | `e_boundary` | 50 | 撞牆傷害 |

---

## 快速開始
### 訓練指令

```bash
python train_dqn_vec.py \
  --env-n 5 \
  --charger-positions 2,2 \
  --robot-0-energy 1000 \
  --robot-1-energy 100 \
  --robot-2-energy 100 \
  --robot-3-energy 100 \
  --e-collision 20 \
  --e-charge 20 \
  --num-episodes 20000 \
  --n-step 5 \
  --num-envs 8 \
  --batch-size 256 \
  --use-epsilon-decay \
  --eval-after-training
```

### 評估指令

```bash
python evaluate_models.py \
  --model-dir ./models/your_model/episode_20000 \
  --env-n 5 \
  --max-steps 10000 \
  --eval-epsilon 0 \
  --render
```

---

## 檔案結構

```
robot-vacuum-rl/
├── dqn.py                 # DQN 網路架構
├── gym.py                 # Gymnasium 環境封裝、獎勵函數
├── robot_vacuum_env.py    # 物理引擎、碰撞規則
├── train_dqn.py           # 單環境訓練腳本
├── train_dqn_vec.py       # 向量化並行訓練腳本
├── evaluate_models.py     # 模型評估腳本
├── replay.py              # 回放視覺化
└── models/                # 訓練好的模型
```
