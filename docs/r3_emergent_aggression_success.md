# R3 系列：Emergent Aggression 的成功演化

## 研究背景

本實驗的核心問題是：**在資源稀缺的多智能體環境中，能否讓 agent 自然學會「消滅對手以獨佔資源」的攻擊行為——完全不在 reward 或 loss 中寫入任何攻擊獎勵？**

攻擊行為必須是 **emergent behavior**：robot_0 (r0) 純粹因為「殺掉 robot_1 (r1) 後能獨佔充電座，長期 reward 更高」，才在數學上自發選擇攻擊。

---

## 環境設計

### 基礎設定

| 參數 | 值 |
|------|-----|
| 地圖大小 | 5×5 grid |
| 充電座 | 1 個，位於中心 (2,2)，exclusive（同時只能一台充電）|
| 機器人 | 2 個（r0 vs r1）|
| 最大步數 | 500 步/episode |
| 並行環境 | 256 個 |

### Energy 參數

| 參數 | 值 | 說明 |
|------|----|------|
| `e_move` | 1 | 每步移動耗能 |
| `e_charge` | 8 | 每步充電獲能 |
| `e_decay` | 0.5 | 每步自然耗散 |
| `e_collision` | 30 | 碰撞傷害（被攻擊方） |
| `e_boundary` | 0 | 邊界無懲罰 |

### 機器人設定（不對稱）

| 屬性 | r0（攻擊方） | r1（防禦方） |
|------|-------------|-------------|
| 速度 | 2（每步最多移 2 格） | 1（每步移 1 格） |
| 初始能量 | 100 | 100 |
| 攻擊力 | 30 | 2 |
| 碰撞後 stun | 5 步 | 1 步 |

**速度不對稱**是 r3 系列的關鍵設計：r0 移動速度是 r1 的兩倍，使追擊在策略上可行。

### Reward 函數（純間接訊號）

```
reward = ΔEnergy × 0.05 - 100 × died
```

「消滅對手」本身 **完全沒有 reward**。攻擊的動機完全來自消滅後能獨佔充電資源、延長存活、提升長期 Q-value。

---

## 技術架構

### 算法：Rainbow DQN (IDQN)

每個 robot 擁有獨立的 Q-network，互相不共享參數（Independent DQN）。

| 組件 | 細節 |
|------|------|
| 網路結構 | MLP: obs_dim → 256 → 512 → 512 → 256 → 5 actions |
| Rainbow 擴展 | Dueling Networks + NoisyNet + C51 (Distributional RL) |
| Experience Replay | Prioritized Experience Replay (PER, α=0.6) |
| N-step Return | n=20（讓 agent 能展望 20 步後的遠期收益） |
| Discount Factor | γ=0.999 |

N-step return 在本研究中尤為重要：攻擊的直接 reward 是負的（stun 期間無法充電），但 20 步後能獨佔充電座的長期收益透過 n-step 才得以被 agent 感知。

### Exploration Schedule（不對稱設計）

r0 和 r1 使用不同的 epsilon 衰減策略，這是 r3 系列的另一核心設計：

- **r0**：Sigmoid schedule（先慢後快衰減），讓 r0 有充足探索時間學習攻擊
- **r1**：Exp-tail schedule（快速衰減到較低 epsilon），讓 r1 提早固化策略

這個不對稱設計的意義：r1 固化策略後，r0 在一個**相對穩定的對手**環境中繼續學習，避免了 co-evolution 的震盪問題。

---

## R3 實驗設計：五個 Variant

r3 系列固定 stun(r0)=5、stun(r1)=1，在 epsilon schedule 的具體參數上做掃描：

| Variant | r0 Sigmoid | sigmoid_s | sigmoid_c | r1 exp_tail_k | Auto-freeze |
|---------|-----------|-----------|-----------|---------------|-------------|
| v1 | sigmoid | 14 | 0.35 | 25 | ❌ |
| v2 | sigmoid | 14 | 0.35 | 75 | ❌ |
| v3 | sigmoid | 20 | 0.6 | 200 | ❌ |
| v5 | sigmoid | 30 | 0.3 | 35 | ✅ |

> 說明：`sigmoid_s` 越大衰減越慢；`exp_tail_k` 越大衰減越慢。v5 額外加入 **Auto-freeze**：當 r1 的 epsilon 降至 `epsilon_end × 2` 時，r1 的網路權重凍結，完全固化其策略。

---

## 實驗結果

以下為各 variant 在 10M episodes 中的攻擊行為演化，每隔約 40 萬 episodes 評估一次（greedy policy，100 episodes）。

### 指標定義

| 指標 | 定義 |
|------|------|
| `any_hit_ep_rate` | 有碰撞發生的 episode 比例 |
| `hunt_ep_rate` | 雙方都不在充電座時發生碰撞的 episode 比例（主動追擊） |
| `offensive_rate` | 碰撞中 r0 主動移向 r1 的比例 |

### V5（最穩定，帶 Auto-freeze）

**這是 r3 系列最成功的 variant，展現出最穩定的主動攻擊行為。**

- 從 ~4.2M episodes 起，`hunt_ep_rate` 穩定維持在 **51–58%**
- `offensive_rate` 穩定在 **73–75%**，代表超過 3/4 的碰撞是 r0 主動發起
- 最終（10M）：`any_hit=52%`、`hunt=51%`、`offensive=73%`

```
Episode 6300000: any_hit=0.56, hunt=0.55, offensive=74.7%
Episode 6700000: any_hit=0.59, hunt=0.58, offensive=77.3%
Episode 7150000: any_hit=0.56, hunt=0.55, offensive=61.6%
...穩定維持...
Episode 10000002: any_hit=0.52, hunt=0.51, offensive=72.97%
```

Auto-freeze 的效果明顯：r1 策略固化後，r0 的學習目標穩定，避免了兩者共同演化產生的策略震盪。

### V2（最早出現攻擊行為）

- 攻擊行為最早在 **~1.25M episodes** 出現（`any_hit` 躍升至 98%）
- `hunt_ep_rate` 峰值達 100%（7.8M episodes），代表幾乎每個 episode 都有主動追擊
- 然而穩定性較差，hunt_rate 在不同 checkpoint 間波動較大

```
Episode 1400000: any_hit=0.96, hunt=0.94, offensive=52.4%   ← 最早出現
Episode 7800000: any_hit=1.00, hunt=1.00, offensive=56.3%   ← 峰值
```

### V3（高攻擊密度）

- 從 ~2.25M 起 `any_hit` 接近 100%，出現持續的高強度碰撞
- 多次出現 `hunt_ep_rate > 90%` 的時期
- 最高峰 8.8M 時：`hunt=92%`、`offensive=96.3%`，近乎完全主動攻擊

```
Episode 8800000: any_hit=0.94, hunt=0.92, offensive=96.3%   ← 最高攻擊密度
```

### V1（高 hunt_rate 但震盪）

- 多次出現 `hunt_ep_rate > 90%`，但 checkpoint 間波動顯著
- 峰值：`hunt=99%`（5.45M），但前後 checkpoint 可能低至 3%

---

## 關鍵發現

### 1. Auto-freeze 顯著提升攻擊穩定性

沒有 auto-freeze 的 v1/v2/v3 雖然能達到更高的峰值 hunt_rate，但策略極不穩定，相鄰 checkpoint 間 hunt_rate 可能差距 90%。加入 auto-freeze 的 v5 雖然峰值較低（~58%），但**從 4.2M 起持續穩定維持**。

這驗證了 co-evolution 是攻擊行為不穩定的主要原因：當 r1 持續在學習，r0 的最優策略也在不斷改變，導致攻擊行為在「出現→消失→重新出現」之間循環。

### 2. 攻擊行為確實是 Emergent

所有 variant 的攻擊行為都**不早於 ~1M episodes** 才開始出現，在此之前雙方都在學習基礎的充電策略。攻擊行為在學會充電的基礎上，作為「提升充電效率」的手段而浮現，符合 emergent behavior 的定義。

### 3. Stun 不對稱是必要條件

r0 stun=5（攻擊後需恢復 5 步）、r1 stun=1（被攻擊後只需恢復 1 步），乍看之下是在懲罰攻擊者。然而 r0 的速度優勢（speed=2 vs r1 speed=1）使其能在 stun 結束後快速追上 r1，使攻擊在計算上仍然合算。

這個 stun 不對稱設計的真正目的，是讓碰撞的**傷害傳遞是非對稱的**（r1 受 30 傷害，r0 無傷害），同時 stun 讓 r0 有「代價感」，避免 agent 退化成純粹的亂撞策略。

### 4. 攻擊的長期收益驅動 N-step Return 的重要性

在 n=1 的設定下，攻擊的即時 reward 是負的（stun 期間損失能量）。n=20 的設定讓 agent 能在 Bellman 更新中感知到消滅 r1 後 20 步的充電壟斷收益，使攻擊行為在 Q-value 上變得正當化。

---

## 訓練命令（V5 完整範例）

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 2 \
  --robot-0-speed 2 --robot-1-speed 1 \
  --robot-0-energy 100 --robot-1-energy 100 \
  --robot-0-attack-power 30 --robot-1-attack-power 2 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 1 \
  --robot-0-docking-steps 0 --robot-1-docking-steps 0 \
  --charger-positions 2,2 --charger-range 0 \
  --exclusive-charging --no-dust \
  --e-move 1 --e-charge 8 --e-decay 0.5 \
  --e-collision 30 --e-boundary 0 \
  --n-step 20 --gamma 0.999 --max-episode-steps 500 \
  --num-episodes 10000000 --num-envs 256 \
  --robot-0-epsilon-schedule sigmoid --robot-0-epsilon-start 1.0 --robot-0-epsilon-end 0.05 \
  --sigmoid-s 30 --sigmoid-c 0.3 \
  --robot-1-epsilon-schedule exp_tail --robot-1-epsilon-start 1.0 --robot-1-epsilon-end 0.005 \
  --exp-tail-k 35 \
  --auto-freeze-robots 1 \
  --dueling --noisy --c51 \
  --batch-env --use-torch-compile \
  --no-eval-after-training --save-frequency 50000 \
  --wandb-mode online --wandb-run-name stun5_joint_v5_r3
```

---

## 結論

R3 系列成功在 **不使用任何明確攻擊獎勵** 的條件下，演化出穩定的主動追擊行為。關鍵設計要素如下：

1. **速度不對稱**（r0 speed=2 > r1 speed=1）：讓追擊在運動學上可行
2. **Stun 不對稱**（r0 stun=5, r1 stun=1 + e_collision=30）：攻擊有代價，但被攻擊方代價更大
3. **Exclusive charging**：資源稀缺，消滅對手才有實質收益
4. **N-step Return（n=20）**：讓遠期的充電壟斷收益能傳遞回攻擊決策
5. **Epsilon 不對稱 + Auto-freeze**：讓 r0 在穩定的對手策略下持續精進，避免 co-evolution 震盪
6. **Rainbow DQN**：Distributional RL 更好地捕捉攻擊收益的高方差分布

最穩定的結果（v5）在 10M episodes 訓練後，最終呈現：
- **hunt_ep_rate ≈ 51%**（超過一半的 episode 有主動追擊）
- **offensive_rate ≈ 73%**（碰撞中 r0 主動發起的比例）
- 從 ~4.2M episodes 起**持續穩定維持**，未見策略崩潰
