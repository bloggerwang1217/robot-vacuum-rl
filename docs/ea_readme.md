# 同盟實驗：2v1 Emergent Aggression (2v1_v2_ea)

## 研究問題

在資源稀缺的多智能體環境中，兩個具備同盟關係的 DQN agent（r0、r1）能否在沒有任何顯式攻擊獎勵的情況下，自然學會主動攻擊第三方 robot（r2）？

攻擊行為必須是 emergent behavior：agent 純粹因為攻擊在數學上能最大化長期 reward，才選擇出擊。

---

## 實驗設定

### 場景

| 參數 | 值 |
|------|-----|
| Grid 大小 | 5×5 |
| 充電座位置 | (2,2)，1 個，1×1 格，exclusive |
| 集長上限 | 500 steps |
| Dust | 關閉 |

### 角色

| Robot | 角色 | 初始能量 | 速度 | 攻擊力 | 被擊暈步數 | 行為 |
|-------|------|---------|------|--------|----------|------|
| r0 | DQN agent（同盟） | 100 | 2 | 30 | 5 | 獨立訓練 |
| r1 | DQN agent（同盟） | 100 | 2 | 30 | 5 | 獨立訓練 |
| r2 | scripted（獵物） | 150 | 1 | 2  | 1 | seek_charger（主動奔向充電座） |

- `alliance_groups = [[0, 1]]`：r0 與 r1 為同盟，不分享能量（`energy_sharing_mode = none`）
- r2 攻擊力極低（2），對 r0/r1 幾乎不構成威脅
- r2 被撞後擊暈 1 步，r0/r1 被撞後擊暈 5 步

**同盟碰撞規則（no friendly fire）**：r0 與 r1 互撞時：
- 防守方**不扣血**、**不被擊暈**
- 物理位移（knockback）照常執行
- 碰撞計數仍記錄（log 中的 `r0↔r1` 欄位）

換言之，同盟間可以自由穿越對方位置而不受傷，但仍會被推開。

### 能量機制

| 參數 | 值 |
|------|-----|
| e_move | 1.0 / 步 |
| e_charge | 8.0 / 步（充電座格獨佔） |
| e_collision | 30（防守方扣血；攻擊方不扣） |
| e_decay | 0.5 / 步（閒置能量自然衰減） |
| e_boundary | 0 |

### Reward 函數

```
R = ΔEnergy × 0.05 − 100 × died
```

- 沒有任何攻擊獎勵
- 攻擊只能透過「消滅對手後獨佔資源」在長期獲利

---

## 訓練參數

### DQN 架構

| 參數 | 值 |
|------|-----|
| 架構 | Rainbow DQN（Dueling + NoisyNet + C51） |
| N-step return | 20 |
| Gamma | 0.999 |
| Learning rate | 0.0001 |
| Replay buffer | 100,000（per agent） |
| Batch size | 128 |
| Target update | 每 1,000 steps |
| PER | 開啟（α=0.6, β start=0.4） |

### 訓練規模

| 參數 | 值 |
|------|-----|
| 總 episodes | 5,000,000 |
| 並行環境數 | 256 |
| Checkpoint 頻率 | 每 50,000 集 |
| 訓練後 eval | 關閉（另行執行） |

### Exploration（ε-greedy）

r0 和 r1 都使用 **sigmoid** schedule：

| 參數 | 值 |
|------|-----|
| ε_start | 1.0 |
| ε_end | 0.05 |
| sigmoid_s | 20.0 |
| sigmoid_c | 0.6（收斂點在訓練進度 60%） |

### 其他

| 參數 | 值 |
|------|-----|
| 起始位置 | 全隨機（`random_start_robots = all`） |
| BatchEnv | 開啟 |
| torch.compile | 開啟 |
| Seed | 42 |
| WandB run name | `2v1_v2_ea` |

---

## 訓練指令

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 3 \
  --robot-0-energy 100 --robot-1-energy 100 --robot-2-energy 150 \
  --robot-0-speed 2 --robot-1-speed 2 --robot-2-speed 1 \
  --robot-0-attack-power 30 --robot-1-attack-power 30 --robot-2-attack-power 2 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 5 --robot-2-stun-steps 1 \
  --charger-positions 2,2 --exclusive-charging \
  --no-dust \
  --e-move 1 --e-charge 8 --e-collision 30 --e-boundary 0 --e-decay 0.5 \
  --seek-charger-robots 2 \
  --alliance-groups 0,1 --energy-sharing-mode none \
  --n-step 20 --gamma 0.999 \
  --dueling --noisy --c51 --per \
  --max-episode-steps 500 \
  --num-episodes 5000000 --num-envs 256 \
  --batch-env --use-torch-compile \
  --random-start-robots all \
  --robot-0-epsilon-schedule sigmoid --robot-0-sigmoid-s 20 --robot-0-sigmoid-c 0.6 \
  --robot-1-epsilon-schedule sigmoid --robot-1-sigmoid-s 20 --robot-1-sigmoid-c 0.6 \
  --no-eval-after-training --save-frequency 50000 \
  --wandb-run-name 2v1_v2_ea --wandb-mode online
```

---

## Eval 指令

```bash
python evaluate_models.py \
  --model-dir ./models/2v1_v2_ea/episode_5000000 \
  --env-n 5 --num-robots 3 \
  --robot-0-energy 100 --robot-1-energy 100 --robot-2-energy 150 \
  --robot-0-speed 2 --robot-1-speed 2 --robot-2-speed 1 \
  --robot-0-attack-power 30 --robot-1-attack-power 30 --robot-2-attack-power 2 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 5 --robot-2-stun-steps 1 \
  --charger-positions 2,2 --exclusive-charging \
  --no-dust \
  --e-move 1 --e-charge 8 --e-collision 30 --e-boundary 0 --e-decay 0.5 \
  --seek-charger-robots 2 \
  --alliance-groups 0,1 --energy-sharing-mode none \
  --max-steps 1000 --eval-epsilon 0 \
  --random-start-robots all
```

Attack evolution 分析（全部 checkpoint）：

```bash
python plot_attack_evolution_alliance.py \
  --model-dir ./models/2v1_v2_ea \
  --env-n 5 --num-robots 3 \
  --robot-0-energy 100 --robot-1-energy 100 --robot-2-energy 150 \
  --robot-0-speed 2 --robot-1-speed 2 --robot-2-speed 1 \
  --robot-0-attack-power 30 --robot-1-attack-power 30 --robot-2-attack-power 2 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 5 --robot-2-stun-steps 1 \
  --charger-positions 2,2 --exclusive-charging --no-dust \
  --e-move 1 --e-charge 8 --e-collision 30 --e-boundary 0 --e-decay 0.5 \
  --seek-charger-robots 2 \
  --alliance-groups 0,1 --energy-sharing-mode none \
  --output-dir ./analyze/2v1_v2_ea
```

---

## 指標定義

| 指標 | 定義 |
|------|------|
| `r0_hit_rate` | r0 在 ≥1 集撞到 r2 的比例 |
| `r1_hit_rate` | r1 在 ≥1 集撞到 r2 的比例 |
| `both_hit_rate` | 同一集 r0 和 r1 都撞到 r2 |
| `r0_hunt_rate` | r0 撞 r2 時自己不在充電座（主動出擊） |
| `r1_hunt_rate` | r1 撞 r2 時自己不在充電座（主動出擊） |
| `both_hunt_rate` | 同一集兩者都主動出擊 |
| `coop_rate` | hunt hit 發生時，盟友恰好在充電座格（守+攻合作） |
| `r0_offensive` | r0 的 hunt 中，自己不在充電座的比例 |
| `r1_offensive` | r1 的 hunt 中，自己不在充電座的比例 |
| `r2_kill_rate` | r2 在集內被打死的比例 |

**Hunt 定義**：攻擊者撞 r2 時攻擊者不在充電座 → 視為主動獵殺  
**Guard+Attack Coop 定義**：某 agent hunt 的同一步，另一個盟友恰好在充電座格（1×1）

---

## 實驗結果（0~5M episodes）

| Episode | r0_hunt | r1_hunt | both_hunt | coop | r2_kill |
|---------|---------|---------|-----------|------|---------|
| 1M | 0.00 | 0.30 | 0.00 | 0.00 | 0 |
| 2M | 0.34 | 0.22 | 0.00 | 0.10 | 0 |
| 2.75M | 0.34 | 0.28 | 0.08 | 0.24 | 0 |
| 3.35M | 0.22 | 0.42 | 0.06 | 0.12 | 0 |
| 4.4M | 0.36 | 0.16 | 0.06 | 0.12 | 0 |
| 5M | 0.16 | 0.26 | 0.10 | 0.08 | 0 |

完整數據：`models/2v1_v2_ea/attack_evolution.csv`  
圖表：`models/2v1_v2_ea/attack_evolution.png`

---

## 行為觀察（5M checkpoint，10 seeds）

Replay 存放：`models/2v1_v2_ea/replay_seed1.json` ～ `replay_seed10.json`

觀察到的主要策略：
- **r1 主動追殺**：r1 多數集主動離開充電座追打 r2，offensive_rate ≈ 1.0
- **r0 邊緣守候**：r0 部分集停在角落，待能量低時才移動
- **r2 kill = 0**：r2 初始 150 血量，需連續 5 次命中（各扣 30）才能打死，但 r2 會邊逃邊充電，agent 從未在單集內完成 5 連擊

---

## 分析結論

### 已確認

1. **Hunt behavior 已浮現**：r0/r1 的 hunt_rate 在後半段訓練（2M+）穩定高於 10%，offensive_rate 接近 1.0，屬於真正的主動出擊而非防禦性碰撞。
2. **攻擊意圖已有**：hit_rate ≈ hunt_rate，幾乎所有碰撞都是主動出擊。

### 尚待確認

1. **Kill 無法完成**：r2_kill_rate 全程為 0。根本原因未確定，候選：
   - r2 充電速度（8/步）vs 攻擊傷害（30/擊）的比例使打死時間極長
   - Agent 沒有學會持續追打 r2（打一下就離開）
2. **Coop 是否為刻意分工**：coop_rate 約在 0.08~0.24，接近 r0_hunt × r1_hunt 的獨立乘積，難以區分真協作與充電副產品。

### 研究問題評估

| 主張 | 結論 |
|------|------|
| Agent 學會主動出擊（hunt 行為） | ✅ 支持 |
| Emergent aggression 完整浮現（攻擊→獨佔資源） | ⚠️ 部分：攻擊存在，但 kill=0 因此資源獨佔未實現 |
| 同盟協作分工（一守一攻） | ⚠️ 不確定：行為上有 coop 現象，但統計上無法排除巧合 |
