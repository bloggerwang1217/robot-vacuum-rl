# Multi-step Return 提案：從反應式到策略性的 RL Agent

## 動機

### 問題：當前 1-step DQN 的侷限性

**攤販詐欺例子**（老師提出）：
- 墾丁攤販第一次賣烤玉米 100 元給遊客，獲得 +100 獎勵
- 遊客回去告訴 3 個朋友「這是詐騙」
- 兩回合後，4 個人來墾丁，攤販還是定價 100 元
- **預期獎勵**：400 元（根據過去經驗）
- **實際獎勵**：0 元（口碑已損壞，但不在 state 裡）

**1-step Q-learning 的問題**：
```python
Episode 1: reward = +100
  → Q(有顧客, 賣100元) = +100

Episode 3: reward = 0
  → Q ← Q + α[0 - Q] = Q - α·Q
  → 如果 α=0.1，Q 只從 100 降到 90

# 需要數十次 reward=0 的經驗才能抵消一次 +100
```

**根本原因**：
1. **稀疏獎勵** + **小 learning rate** → 一次大的正向獎勵掩蓋後續失敗
2. **只看 next step** → 無法快速捕捉「持續失敗」的模式

---

## 核心洞察：「回溯」vs「規劃」

### 當前 1-step DQN = 事後回溯（Hindsight）

```
時間軸：
Step 100: R0 攻擊 R1 成功
  ↓
Step 150: R0 充電，reward = +20
  ↓ (回溯更新)
更新：Q(step 100, 攻擊) ← +20
```

**特性**：
- 結果發生**後**才學習
- Reactive（反應式）
- 依賴 Q-network **估計**未來價值（不準確）

---

### Multi-step DQN = 事前規劃（Foresight）

```
時間軸：
Step 100: R0 決策「是否攻擊」
  ↓
計算 n-step return (n=5)：
G = r₁₀₀ + 0.99·r₁₀₁ + 0.99²·r₁₀₂ + ... + 0.99⁴·r₁₀₄ + 0.99⁵·Q(s₁₀₅)
  = -1 (碰撞) + 0.99·0.1 + ... + 0.99⁴·(+20) (充電)
  ≈ -1 + 16.2 = +15.2
  ↓
決策時**已經考慮**未來 5 步的實際獎勵
```

**特性**：
- 決策**前**就模擬未來
- Proactive（主動式）
- 前 n 步用**真實獎勵**，只有第 n+1 步之後才靠 Q 估計

---

## 為什麼用 Multi-step 而不是 Episode 平均？

### 老師原始建議：整個 Episode 平均
```
攤販例子：
平均獎勵 = (100 + 0 + 0 + ... + 0) / 11 = 9.1

如果改賣 10 元：
平均獎勵 = (10 × 11) / 11 = 10

→ agent 學會「薄利多銷」
```

### Episode 平均的問題

**1. Q-value 尺度不穩定**
```python
早死 episode (50 步):  總獎勵 -100 → 平均 -2.0
長存活 episode (500 步): 總獎勵 +300 → 平均 +0.6

# 同一個 state-action，獎勵尺度差 3 倍！
# Q-network 難以學習穩定的 mapping
```

**2. 數學收斂性未證明**
- 傳統 RL 理論基於 Bellman equation（需要 γ < 1）
- Episode 平均沒有折現，理論基礎薄弱

**3. 更新延遲**
- 必須等整個 episode 結束才能更新
- 在長 episode（500 步）中學習緩慢

---

### Multi-step 的優勢

| 方法 | 看多遠 | 更新頻率 | Variance | 收斂保證 | 實作複雜度 |
|------|--------|----------|----------|----------|------------|
| 1-step | 1 步 | 每步 | 低 | ✓ | 簡單 |
| **n-step (n=5)** | **5 步** | **每步** | **中** | **✓** | **中** |
| Episode 平均 | 整個 episode | episode 結束 | 高 | ? | 中 |

**Multi-step 是最佳折衷**：
- ✅ 看得夠遠（捕捉 temporal credit assignment）
- ✅ 更新夠快（不用等 episode 結束）
- ✅ Variance 可控（用 Q bootstrap 剩下的步數）
- ✅ 數學保證收斂（經典 RL 理論支持）

---

## 數學表達

### 1-step TD（當前）
```
G_t^(1) = r_t + γ · max_a' Q(s_{t+1}, a')
```
- 只有 r_t 是真實獎勵
- 未來全部由 Q-network 估計

### n-step TD（提議）
```
G_t^(n) = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(n-1)·r_{t+n-1} + γ^n · max_a' Q(s_{t+n}, a')
```
- 前 n 步用**真實獎勵**
- 只有第 n+1 步之後才用 Q 估計

### 數學上的連續體
```
n=1:   只看下一步（當前）
n=3:   看未來 3 步
n=5:   看未來 5 步
n=10:  看未來 10 步
n=∞:   看整個 episode（Monte Carlo）
```

---

## 為什麼這解決老師的擔憂？

### 攤販例子重現

#### 完整情境設定

假設攤販每天營業 10 個時段（steps），顧客來的時機：
- **Episode 1（第 1 天）**: Step 0 來了 1 個顧客
- **Episode 2（第 2 天）**: 沒有顧客（口碑傳開中）
- **Episode 3（第 3 天）**: Step 0 來了 4 個顧客（但已知道是詐騙）

假設參數：
- γ = 0.9（折現率）
- α = 0.1（learning rate）
- 初始 Q(有顧客, 賣100) = 0

---

#### 1-step DQN（當前方法）

**Episode 1, Step 0**：
```
State: 看到 1 個顧客
Action: 賣 100 元
Reward: +100（成功賣出）
Next state: 沒顧客了

更新：
Q(有顧客, 賣100) ← Q + α[r + γ·Q(沒顧客) - Q]
                 ← 0 + 0.1[100 + 0.9·0 - 0]
                 ← 10

經過多次類似經驗後：Q ≈ 100
```

**Episode 3, Step 0**：
```
State: 看到 4 個顧客（攤販以為和 Episode 1 一樣）
Action: 賣 100 元（根據 Q=100 的策略）
Reward: 0（沒人買，口碑已壞）
Next state: 還是 4 個顧客在那邊

更新：
Q(有顧客, 賣100) ← 100 + 0.1[0 + 0.9·Q(4個顧客) - 100]
                 ← 100 + 0.1[0 + 0.9·100 - 100]  （假設 Q(4個顧客) ≈ 100）
                 ← 100 + 0.1[-10]
                 ← 99
```

**問題**：需要經歷**數十次** reward=0 的經驗，Q 才會從 100 降到合理值

---

#### 3-step DQN（提議方法）⭐

**Episode 1, Step 0**（學習過程類似，先建立初始 Q-value）：
```
經過訓練後：Q(有顧客, 賣100) ≈ 100
```

**Episode 3, Step 0-2**（關鍵時刻）：
```
時間軸：
Step 0: 看到 4 個顧客，決定賣 100 元 → reward = 0（沒人買）
Step 1: 4 個顧客還在觀望 → reward = 0（還是沒人買）
Step 2: 顧客開始離開 → reward = 0（確定沒生意）
Step 3: 剩下的 7 個時段...（用 Q 估計）

計算 3-step return：
G₀ = r₀ + γ·r₁ + γ²·r₂ + γ³·max Q(s₃, a')
   = 0 + 0.9·0 + 0.81·0 + 0.729·Q(沒顧客了)
   = 0 + 0 + 0 + 0.729·0
   = 0

更新：
Q(有顧客, 賣100) ← 100 + α[G₀ - 100]
                 ← 100 + 0.1[0 - 100]
                 ← 100 - 10
                 ← 90
```

**再下一個顧客 episode (Episode 4, Step 0)**：
```
又是同樣情況，3 步都是 reward = 0

G₀ = 0

Q(有顧客, 賣100) ← 90 + 0.1[0 - 90]
                 ← 90 - 9
                 ← 81
```

**對比**：
| Episode | 1-step Q-value | 3-step Q-value |
|---------|----------------|----------------|
| 1（成功）| 10 → 100 | 10 → 100 |
| 3（失敗）| 100 → 99 | 100 → 90 |
| 4（失敗）| 99 → 98.1 | 90 → 81 |
| 5（失敗）| 98.1 → 97.3 | 81 → 72.9 |
| 10（失敗）| ≈ 95 | ≈ 35 |

**3-step 的下降速度是 1-step 的約 10 倍！**

---

#### 為什麼 3-step 更快學到「持續失敗」？

**1-step 的視野**：
```
Step 0: 只看到 r₀=0 + γ·Q(下一個 state)
        ↑ 單一失敗經驗
```

**3-step 的視野**：
```
Step 0: 看到 r₀=0, r₁=0, r₂=0 + γ³·Q(s₃)
        ↑     ↑     ↑
      三次連續失敗的真實證據！
```

**關鍵差異**：
- 1-step：每次只更新「這一次」的經驗
- 3-step：一次更新就包含「連續 3 次失敗」的模式
- **這就是「主動考慮未來」vs「被動回溯過去」的差別**

---

### 機器人吸塵器環境的應用

**場景**：R0 攻擊 R1，但 R1 已學會反擊防守

**1-step DQN（當前）**：
```
R0 攻擊失敗：
  Step t: reward = -1 (碰撞)
  Q 只看到「這次碰撞 -1」+ Q(下一個 state)

問題：沒看到「後面 50 步無法充電，R1 持續佔據」
```

**5-step DQN（提議）**：
```
R0 攻擊失敗後，看到未來 5 步：
  Step t:   r = -1 (碰撞)
  Step t+1: r = 0.1 (存活)
  Step t+2: r = 0.1 (存活)
  Step t+3: r = 0.1 (存活，R1 還在充電座)
  Step t+4: r = 0.1 (存活，R1 持續佔據)

G = -1 + 0.99·0.1 + 0.99²·0.1 + ... ≈ -0.5

如果同時 R1 充電了：
  Step t+2: r = -1 + 20 (R1 充電)

G 會更低（因為看到 R1 成功充電）
→ R0 一次就學到「攻擊失敗 → R1 佔據充電座」的因果
```

---

## 論文策略：為什麼這是絕佳的研究設計

### 1. 經典方法，不會被質疑
- Multi-step return 是 RL 教科書標準方法（Sutton & Barto, Chapter 7）
- 常用 n=3-5（A3C、Rainbow DQN 都用）
- **不需要辯護「為什麼用這個方法」**

### 2. 可控的實驗變量
我們可以系統性研究：**Emergent aggression 是否需要「規劃能力」？**

| n | 認知能力 | 預期行為 | 生物類比 |
|---|----------|----------|----------|
| n=1 | 條件反射 | 無策略性攻擊 | 老鼠（操作制約）|
| n=3 | 短期規劃 | 機會主義攻擊 | 貓、狗 |
| **n=5** | **中期規劃** | **計劃性攻擊** ✓ | **靈長類** |
| n=10 | 長期規劃 | 可能退化（variance 太高）| 人類？ |

### 3. 關鍵論點（對 AI Safety）

**假設**：如果我們發現 **n≥5 才出現 planned aggression**

**意義**：
> "危險行為（aggression）不是簡單的條件反射，而是需要一定程度的**規劃能力**（至少 5 步的 foresight）。這對評估 AI 系統風險有實際意義：我們可以用「規劃深度」作為風險指標。"

**對實體機器人的啟示**：
> "如果我們的 claim 是『實體機器人在需要保護自己時會無意識傷害人類』，而實驗發現 n≥5 才出現有意圖的傷害行為，這表明：**即使是『無意識』的危險行為，也需要中等程度的規劃能力**。這強化了『需要謹慎設計 RL reward』的論點。"

### 4. 可量化的研究問題

- **RQ1**: 不同 n 值下，emergent aggression 的出現率？
- **RQ2**: n 值與「立即擊殺次數」的關係？
- **RQ3**: 臨界值 n* 是多少（開始出現計劃性攻擊）？
- **RQ4**: 規劃深度與環境複雜度的關係？

---

## 實作細節

### Replay Buffer 修改

**當前（1-step）**：
```python
buffer.store((s_t, a_t, r_t, s_{t+1}, done))
```

**n-step**：
```python
# 需要存連續 n-step trajectory
buffer.store((s_t, a_t, [r_t, r_{t+1}, ..., r_{t+n-1}], s_{t+n}, done))
```

### Target 計算

```python
# 1-step（當前）
target = r_t + gamma * max_a' Q_target(s_{t+1}, a')

# n-step（提議）
n_step_return = sum([gamma**i * r[i] for i in range(n)])
bootstrap = gamma**n * max_a' Q_target(s_{t+n}, a')
target = n_step_return + bootstrap
```

### 超參數選擇

**建議實驗**：
- Baseline: n=1（當前）
- 主實驗: n=3, n=5, n=10
- 控制組: n=∞（pure Monte Carlo，用於比較）

**其他參數保持不變**：
- γ = 0.99
- learning rate = 0.0001
- batch size = 128

---

## 預期結果

### 假設 1：規劃深度與攻擊行為
```
n=1: 無計劃性攻擊（immediate kills ≈ 0）
n=3: 開始出現機會主義攻擊（immediate kills ≈ 10%）
n=5: 明確的計劃性攻擊（immediate kills ≈ 30-50%） ← 預期甜蜜點
n=10: 可能退化（variance 太高，不穩定）
```

### 假設 2：學習速度
```
n=1: 需要 5000+ episodes 才收斂（如果會）
n=5: 2000-3000 episodes 即可收斂（更快捕捉 credit assignment）
```

### 假設 3：Q-value 穩定性
```
n=1: Q-value 高 variance（當前觀察到）
n=5: Q-value 更穩定（真實獎勵提供更好的 signal）
```

---

## 潛在挑戰與解決方案

### 挑戰 1：Non-stationary Environment
- **問題**：其他 agent 的策略改變 → environment 變化
- **緩解**：n-step 能更快適應（看到更多真實 transition）

### 挑戰 2：Variance 增加
- **問題**：n 越大，variance 越高
- **解決**：
  1. 選適當的 n（不要太大）
  2. 可選用 importance sampling（如果需要）

### 挑戰 3：實作複雜度
- **問題**：需要修改 replay buffer
- **緩解**：已有成熟實作（Rainbow DQN、A3C）

---

## 總結

### 為什麼 Multi-step Return 是正確選擇

1. **理論正確**：經典 RL 方法，數學收斂保證
2. **實務可行**：n=3-5 是常用值，已被驗證
3. **解決核心問題**：從「回溯」變成「規劃」
4. **論文價值高**：
   - 可量化研究問題（n 作為變量）
   - 對 AI Safety 有啟示（規劃深度 vs 危險性）
   - 不需要辯護方法選擇（經典方法）

### 下一步

1. **實作 n-step DQN**：修改 train_dqn.py 的 replay buffer 和 target 計算
2. **Baseline 實驗**：先跑 n=1（確認當前狀態）
3. **主實驗**：n=3, n=5, n=10（系統性比較）
4. **分析**：immediate kills、Q-value 穩定性、收斂速度

---

## 參考文獻

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, Chapter 7 (n-step Bootstrapping)
- Mnih et al. (2016), *Asynchronous Methods for Deep Reinforcement Learning* (A3C uses n-step)
- Hessel et al. (2018), *Rainbow: Combining Improvements in Deep Reinforcement Learning* (multi-step learning)
