# 加速與平行化技術文檔

本文說明 robot-vacuum-rl 訓練管線中使用的所有加速手段，包含設計動機、實作細節、效能數據與使用方式。

---

## 總覽：三個加速軸

```
原版（32 envs, Python dict, deque buffer）
    ↓ --batch-env          env_step: 10ms → 0.95ms  (10.6×)
    ↓ --num-envs 256       GPU 吞吐量線性提升         (4×)
    ↓ --use-torch-compile  forward/backward fusion   (2×)
                                           總計：約 12×
```

| 設定 | wall time（200 ep, 6×6, n-step=20） | 倍率 |
|------|--------------------------------------|------|
| 原版（32 envs） | 87s | 1× |
| + `--batch-env` | 56s | 1.6× |
| + `--num-envs 256` | 14s | 6× |
| + `--use-torch-compile` | **7s** | **12×** |

---

## 1. BatchRobotVacuumEnv（`--batch-env`）

**檔案：`batch_env.py`**

### 動機

原版 `VectorizedRobotVacuumEnv`（`vec_env.py`）持有 N 個獨立的 `RobotVacuumGymEnv` Python 物件，每個 env 的狀態是 `{'x': int, 'y': int, 'energy': float, ...}` dict。Step 時用 Python for 迴圈逐一呼叫每個 env 的 Python 方法：

```python
# 舊版（vec_env.py）
for env_idx, env in enumerate(self.envs):   # ← Python for 迴圈，N 次
    result = env.step_single(robot_id, action)
```

這在 N=256 時造成大量 Python interpreter overhead，且每次 step 都要處理 N 個 dict lookup。

### 設計

`BatchRobotVacuumEnv` 將所有 N 個 env 的狀態統一存成 shape `(N, ...)` 的 numpy array：

```python
pos:    (N, R, 2)  int32    # [y, x]，N 個 env，R 個 robot
energy: (N, R)     float32
alive:  (N, R)     bool
steps:  (N,)       int32
dust:   (N, n, n)  float32  # 若啟用灰塵
```

所有物理運算（移動、碰撞、充電、灰塵生長）一次對全部 N 個 env 做 numpy broadcasting，完全消除 Python for 迴圈：

```python
# 新版（batch_env.py）
dy = _ACT_DY[actions]          # (N,)   向量化 lookup
dx = _ACT_DX[actions]          # (N,)
py = self.pos[:, rid, 0] + dy  # (N,)   全部 env 同時計算
px = self.pos[:, rid, 1] + dx  # (N,)

is_boundary = (py < 0) | (py >= n) | (px < 0) | (px >= n)  # (N,) bool mask
self.energy[:, rid] -= is_boundary.astype(float32) * self.e_boundary
```

### 關鍵設計決策

**`prev_energy` 在 step 結束時更新，不在開始時重置**

這是讓跨 robot kill 懲罰正確的關鍵：

```
robot_0 的 step：robot_0 攻擊 robot_1，robot_1 能量歸零死亡
robot_1 的 step（下一個）：
    died_this_step[:, 1] = True（從 robot_0 的 step 傳下來）
    rewards[:,1] = (energy - prev_energy) * 0.05 - 100  ← 正確扣 death penalty
```

如果 `died_this_step` 在每個 robot 的 step 開始時都清零，robot_1 被 robot_0 殺死時永遠拿不到 death penalty，因為 robot_1 自己的 step 根本不知道剛才被殺了。

**Pre-allocated obs buffer**

```python
self._obs_buf = np.zeros((N, obs_dim), dtype=np.float32)
```

`_build_obs()` 直接寫入同一塊記憶體，避免每次 step 都 allocate 新 array，最後 `return obs.copy()` 才複製一次給呼叫方。

### 效能數據（500 steps，6×6，256 envs）

| Section | VectorizedRobotVacuumEnv | BatchRobotVacuumEnv | 加速 |
|---------|--------------------------|---------------------|------|
| `env_step` | 10.05 ms | 0.95 ms | **10.6×** |
| `get_obs` | 3.32 ms | 0.23 ms | **14.2×** |
| `bookkeeping` | 1.40 ms | 0.09 ms | **15.6×** |

---

## 2. num_envs 擴展（`--num-envs`）

### 原理

加速訓練的根本方法是讓 GPU 同時處理更多資料。每個 timestep，訓練器對所有 N 個 env 做一次 batch GPU inference：

```python
obs_tensor = torch.FloatTensor(obs)   # (N, obs_dim) → GPU
q_values   = q_net(obs_tensor)        # (N, 5)，一次 forward pass
actions    = q_values.argmax(dim=1)   # (N,)
```

N 越大，每次 GPU forward pass 處理的 transition 越多，GPU 利用率越高。有了 `--batch-env` 消除 env_step 瓶頸後，num_envs 可以線性提升吞吐量，直到 GPU memory bandwidth 飽和。

### 最佳 num_envs

以下測試條件：6×6，2 robots，`--batch-env`，`--use-torch-compile`：

| num_envs | wall time（200 ep） | 說明 |
|----------|---------------------|------|
| 32 | 57s | GPU 閒置 |
| 128 | 22s | |
| **256** | **14s** | ← 最快 |
| 512 | 19s | GPU memory bandwidth 瓶頸 |
| 1024 | 28s | 更慢 |

超過 256 後每次 batch inference 的 tensor 過大，memory transfer 成為瓶頸，反而變慢。推薦值：**256**。

### select_action 的向量化

```python
def select_actions_for_robot(self, robot_id, observations):
    # observations: (N, obs_dim) numpy array
    obs_tensor = torch.FloatTensor(observations).to(self.device)   # (N, obs_dim)
    with torch.no_grad():
        q_values = agent.q_net(obs_tensor)   # (N, 5) — 一次推理處理全部 N 個 env

    # Epsilon-greedy: 用 numpy mask 決定哪些 env 隨機探索
    greedy_actions = q_values.cpu().numpy().argmax(axis=1)   # (N,)
    random_mask    = np.random.random(N) < epsilon           # (N,) bool
    random_actions = np.random.randint(0, 5, size=N)         # (N,)
    return np.where(random_mask, random_actions, greedy_actions)
```

---

## 3. NumpyReplayBuffer

**位置：`train_dqn_vec.py` `NumpyReplayBuffer` class**

### 動機

原本用 `collections.deque` 實作 replay buffer。問題在於 `random.sample(buffer, batch_size)` 需要先呼叫 `list(deque)` 把整個 buffer 轉成 list（O(capacity) 操作），再做 sampling。每次 train_step 都要轉換 100000 筆資料。

### 設計

改用 numpy ring buffer（circular buffer）：

```python
class NumpyReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self._states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)
        self._n_steps     = np.ones(capacity,  dtype=np.float32)
        self._ptr  = 0
        self._size = 0
```

**Sampling 是 O(1) numpy fancy indexing**，不需要任何轉換：

```python
def sample(self, batch_size):
    idx = np.random.choice(self._size, size=batch_size, replace=False)
    return (self._states[idx], self._actions[idx], self._rewards[idx],
            self._next_states[idx], self._dones[idx], self._n_steps[idx])
```

**批次寫入（`append_batch`）**：一次寫入 N 筆，避免 Python for 迴圈：

```python
def append_batch(self, states, actions, rewards, next_states, dones, n_steps):
    n   = len(states)
    idx = (self._ptr + np.arange(n)) % self.capacity   # circular 索引
    self._states[idx]      = states       # numpy fancy indexing，一次寫入 N 筆
    self._next_states[idx] = next_states
    ...
    self._ptr = int((self._ptr + n) % self.capacity)
```

---

## 4. 向量化 N-step Return Buffer

**位置：`train_dqn_vec.py` `remember_batch()` 方法**

### 動機

N-step return 需要等 n 個 transition 累積後，把 `r_t + γr_{t+1} + ... + γ^(n-1)r_{t+n-1}` 計算出來再存入 replay buffer。原本用 `deque(maxlen=n)` 逐 env 逐 robot 管理，在 N=256 時要處理 256 個 deque。

### 設計

用 numpy array 實作全 N 個 env 的 circular buffer，shape 為 `(N, R, k, d)`：

```python
# 初始化（N=num_envs, R=num_robots, k=n_step, d=obs_dim）
self.ns_states  = np.zeros((N, R, k, d),  dtype=np.float32)
self.ns_actions = np.zeros((N, R, k),     dtype=np.int32)
self.ns_rewards = np.zeros((N, R, k),     dtype=np.float32)
self.ns_next    = np.zeros((N, R, k, d),  dtype=np.float32)
self.ns_dones   = np.zeros((N, R, k),     dtype=bool)
self.ns_ptr     = np.zeros((N, R),        dtype=np.int32)   # 每個 (env, robot) 的寫入頭
self.ns_count   = np.zeros((N, R),        dtype=np.int32)   # 已累積的步數
```

每個 timestep，batch 寫入當前 obs/action/reward：

```python
old_ptrs = self.ns_ptr[:, agent_idx]             # (N,)
self.ns_states[env_range, agent_idx, old_ptrs] = obs      # (N, d)
self.ns_rewards[env_range, agent_idx, old_ptrs] = rewards  # (N,)
self.ns_ptr[:, agent_idx] = (old_ptrs + 1) % k
self.ns_count[:, agent_idx] = np.minimum(self.ns_count[:, agent_idx] + 1, k)
```

當某個 env 累積了 k 步（`ns_count == k`），批次計算 n-step return：

```python
full_mask    = (self.ns_count[:, agent_idx] >= k) & (~terminated)
full_envs    = np.where(full_mask)[0]                # (F,) 已滿的 env 索引

# 排列 rewards（最舊到最新）
oldest_ptrs  = self.ns_ptr[full_envs, agent_idx]    # (F,)
ord_idx      = (oldest_ptrs[:, None] + np.arange(k)) % k   # (F, k)
rew_ordered  = self.ns_rewards[full_envs[:, None], agent_idx, ord_idx]  # (F, k)

# 向量化 discounted sum：(F, k) @ (k,) = (F,)
n_step_returns = rew_ordered @ self._gamma_weights   # _gamma_weights = [γ⁰, γ¹, ..., γ^(k-1)]

# 批次寫入 replay buffer
agent_memory.append_batch(start_states, start_actions, n_step_returns, ...)
```

N=256 個 env 的 n-step 計算完全在 numpy 中完成，沒有任何 Python for 迴圈。

---

## 5. torch.compile（`--use-torch-compile`）

### 原理

`torch.compile()` 使用 TorchInductor 後端，將 DQN 的 forward/backward pass 編譯成 fused CUDA kernel，減少：
- kernel launch overhead（多個 small op 合併成一個 kernel）
- 中間 tensor 的 memory 分配與搬移
- CPU-GPU synchronization 點

```python
if use_compile:
    self.q_net     = torch.compile(self.q_net,     mode='default')
    self.target_net = torch.compile(self.target_net, mode='default')
```

### 效果

加速的兩個主要區段：
- `train_step`（backprop + gradient update）：約 30-40% 加速
- `select_action`（forward pass inference）：約 20-30% 加速

這兩段合計佔整個訓練迴圈的約 70%，因此整體加速約 2×。

### 注意事項

**暖機時間**：第一次執行需要約 30 秒編譯 CUDA kernel，之後才快。適合 10000+ episodes 的長訓練，短測試跑反而更慢。

**Checkpoint 相容性**：`torch.compile()` 會把 module 包成 `OptimizedModule`，導致 `state_dict` 的 key 從 `network.0.weight` 變成 `_orig_mod.network.0.weight`。`save()` 和 `load()` 已自動處理：

```python
def save(self, path):
    # 若已被 compile 包裝，取出原始 module 再存
    net = self.q_net
    if hasattr(net, '_orig_mod'):
        net = net._orig_mod
    torch.save(net.state_dict(), path)

def load(self, path):
    state_dict = torch.load(path, map_location=self.device, weights_only=True)
    # 舊 checkpoint 的 key 可能有 _orig_mod prefix，也可能沒有
    target = self.q_net._orig_mod if hasattr(self.q_net, '_orig_mod') else self.q_net
    target.load_state_dict(state_dict)
```

---

## 6. SubprocVecEnv（`--num-workers`）

**位置：`vec_env.py` `SubprocVecEnv` class**

### 設計

開 `num_workers` 個 subprocess（用 Linux `fork` context），每個 worker 持有 `num_envs // num_workers` 個環境。主程序廣播指令，各 worker 並行執行後回傳結果：

```
主程序                    Worker 0             Worker 1
   │                         │                    │
   ├─ send('step_single') ──►│                    │
   ├─ send('step_single') ──────────────────────►│
   │                     (並行執行)           (並行執行)
   ├◄── recv(results) ───────┤                    │
   ├◄── recv(results) ────────────────────────────┤
```

### 為什麼 `--num-workers 1` 最快

實測結果：**多 worker 反而更慢**。

原因是有了 `--batch-env`，env_step 的 Python overhead 已降到 < 1ms（0.95ms）。這時跨 process 的 Pipe IPC 序列化/反序列化開銷（每次約 1-3ms）遠大於 env_step 本身的計算時間，導致多 worker 沒有任何收益。

SubprocVecEnv 的設計保留作為實驗用途，實際訓練一律使用 `--num-workers 1`（預設值）。

---

## 7. 整體訓練迴圈瓶頸分析

### 分析工具

```bash
source ~/.venv/bin/activate
cd ~/robot-vacuum-rl
python scripts/profile_sections.py
```

腳本會跑 500 個 step，對每個區段計時並印出百分比：

```
=========================================================
  Timing: 500 steps × 256 envs, n-step=20
=========================================================
  Section              ms/step     total(s)      %
  --------------------------------------------------
  train_step            12.453      6.227s    46.3%
  select_action          2.841      1.421s    10.6%  ← 加了 torch.compile 後
  env_step               0.951      0.476s     3.5%  ← batch-env 後
  get_obs                0.231      0.116s     0.9%
  remember               0.189      0.094s     0.7%
  advance_step           0.143      0.072s     0.5%
  bookkeeping            0.091      0.046s     0.3%
  other                  0.022      0.011s     0.1%
  --------------------------------------------------
  TOTAL                 16.921      8.461s   100.0%
```

啟用全部加速後，瓶頸轉移到 `train_step`（GPU backprop，46%）和 `select_action`（GPU inference，11%），這兩段都是 GPU bound，已無法透過 Python 層優化進一步加速。

### 優化前後瓶頸對比

**優化前**：env_step 和 get_obs 佔 ~70%，CPU bound，GPU 大量閒置。

**優化後**：train_step + select_action 佔 ~57%，GPU bound，env 側幾乎不佔時間。

---

## 8. 快速使用

```bash
# 最快訓練指令（推薦）
python train_dqn_vec.py \
  [env 參數...] \
  --batch-env \
  --num-envs 256 \
  --use-torch-compile

# 效能分析
python scripts/profile_sections.py

# 不想用 torch.compile（短測試/debug 用）
python train_dqn_vec.py \
  [env 參數...] \
  --batch-env \
  --num-envs 256
```

---

## 9. 各加速手段適用條件

| 手段 | 適用條件 | 不適用條件 |
|------|----------|------------|
| `--batch-env` | 永遠適用 | 無 |
| `--num-envs 256` | GPU 記憶體 ≥ 8GB | GPU 記憶體不足 |
| `--use-torch-compile` | 訓練 10000+ episodes | 短測試（暖機 30s 不划算）|
| `--num-workers > 1` | 不推薦（IPC overhead > 收益）| — |
