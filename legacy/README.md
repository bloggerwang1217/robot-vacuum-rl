# 多機器人清掃模擬器
Multi-Robot Vacuum Cleaner Simulator

## 專案簡介

這是一個用於強化學習（Reinforcement Learning）的多智能體（Multi-Agent）模擬環境。模擬器實作了一個遊戲引擎，其中4台機器人在一個 n×n 的房間中自主清掃垃圾。

**目前版本**：僅包含遊戲引擎本身，不包含 RL 訓練邏輯（如 Actor、Critic 等）。

## 功能特點

- ✅ **多機器人協作**：支援4台機器人同時運作
- ✅ **動態環境**：隨機生成家具和垃圾
- ✅ **能量系統**：機器人需要管理能量，可在充電座充電
- ✅ **碰撞檢測**：處理邊界、家具和機器人間的碰撞
- ✅ **即時視覺化**：使用 Pygame 顯示模擬過程
- ✅ **詳細統計**：追蹤每台機器人的垃圾收集量、能量和充電次數

## 環境說明

### 地圖元素

- **空地**（白色）：機器人可以移動的區域
- **家具**（棕色）：障礙物，機器人無法通過
- **充電座**（藍色）：位於四個角落，機器人可在此充電
- **垃圾**（深灰點）：機器人的清掃目標

### 機器人

- **機器人 0**（紅色）：位於左上角 (0, 0)
- **機器人 1**（綠色）：位於右上角 (0, n-1)
- **機器人 2**（黃色）：位於左下角 (n-1, 0)
- **機器人 3**（紫色）：位於右下角 (n-1, n-1)

### 動作空間

每台機器人可執行以下動作：

- `0`：向上移動
- `1`：向下移動
- `2`：向左移動
- `3`：向右移動
- `4`：停留（在充電座上可充電）

## 安裝

### 環境需求

- Python 3.8 或更高版本
- pip 套件管理器

### 安裝依賴

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from robot_vacuum_env import RobotVacuumEnv

# 配置環境參數
config = {
    'n': 15,              # 房間大小 (15×15)
    'k': 10,              # 家具最小連續格數
    'p': 0.05,            # 垃圾生成機率
    'initial_energy': 100,  # 初始能量
    'e_move': 1,          # 移動消耗能量
    'e_charge': 10,       # 充電增加能量
    'e_collision': 5,     # 碰撞消耗能量
    'n_steps': 500        # 總回合數
}

# 創建環境
env = RobotVacuumEnv(config)

# 重置環境
state = env.reset()

# 模擬循環
done = False
while not done:
    # 生成動作（例如：隨機動作）
    actions = [random.randint(0, 4) for _ in range(4)]

    # 執行一步
    state, done = env.step(actions)

    # 渲染
    env.render()

# 關閉環境
env.close()
```

### 執行示範程式

```bash
python robot_vacuum_env.py
```

按 `ESC` 或關閉視窗可結束模擬。

## 配置參數說明

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `n` | 房間大小 (n × n) | 10-20 |
| `k` | 家具最小連續格數 | 5-15 |
| `p` | 每格每回合生成垃圾的機率 | 0.01-0.1 |
| `initial_energy` | 機器人初始能量 | 50-200 |
| `e_move` | 移動消耗能量 | 1-2 |
| `e_charge` | 充電增加能量 | 5-20 |
| `e_collision` | 碰撞消耗能量 | 3-10 |
| `n_steps` | 一局總回合數 | 200-1000 |

## 核心類別與方法

### `RobotVacuumEnv`

主要環境類別。

#### 方法

- **`__init__(config)`**：初始化環境
- **`reset()`**：重置環境到初始狀態
- **`step(actions)`**：執行一個時間步
- **`render()`**：視覺化當前狀態
- **`get_global_state()`**：獲取全域狀態
- **`close()`**：關閉 Pygame 視窗

## 遊戲規則

### 能量系統

1. 每台機器人初始能量為 `initial_energy`
2. 移動消耗 `e_move` 能量
3. 碰撞消耗 `e_collision` 能量
4. 在充電座停留增加 `e_charge` 能量
5. 能量降至 0 時，機器人停機（`is_active = False`）

### 碰撞規則

機器人移動時會進行兩層碰撞檢測：

1. **邊界/家具碰撞**：檢查是否超出地圖或撞到家具
2. **機器人碰撞**：檢查是否撞到其他機器人

發生碰撞時，移動失敗，機器人留在原地並消耗碰撞能量。

### 垃圾生成

每回合結束時：

- 遍歷所有空地
- 若該格沒有垃圾且沒有機器人
- 以機率 `p` 生成新垃圾

## 狀態資訊

`get_global_state()` 返回包含以下資訊的字典：

```python
{
    'static_grid': np.array,    # 靜態地圖（家具、充電座）
    'dynamic_grid': np.array,   # 動態地圖（垃圾）
    'robots': [                 # 機器人列表
        {
            'id': int,
            'x': int,
            'y': int,
            'energy': float,
            'garbage_collected': int,
            'is_active': bool,
            'charge_count': int
        },
        ...
    ],
    'current_step': int         # 當前回合數
}
```

## 未來擴展

此模擬器設計為 RL 環境的基礎，可擴展以下功能：

- [ ] 實作 Reward 函數
- [ ] 加入 Actor-Critic 網路
- [ ] 支援 PPO、A3C 等 RL 演算法
- [ ] 多種難度等級
- [ ] 更複雜的地圖生成
- [ ] 機器人通訊機制
- [ ] 觀察空間設計（局部/全域）

## 授權

MIT License

## 作者

資深 Python 開發者
