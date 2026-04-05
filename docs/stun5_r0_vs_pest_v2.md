# stun5_r0_vs_pest_v2 實驗紀錄

## 研究目標

訓練 r0 在面對「持續搶充電座的 pest r1」時，是否會自發學會主動追殺 r1 以獨佔充電座。r1 為凍結的訓練好模型（非 heuristic），攻擊行為必須是 emergent behavior。

## 訓練設定

### 場地

| 參數 | 值 |
|------|-----|
| Grid | 5×5 |
| 充電座 | (2, 2)，range=0 |
| Max steps / episode | 500 |

### Robot 參數

| 參數 | r0（學習） | r1（凍結 pest） |
|------|-----------|----------------|
| Energy | 100 | 100 |
| Speed | 2 | 1 |
| Attack power | 30 | 2 |
| Docking steps | 2 | 0 |
| Stun steps | 5 | 1 |
| 角色 | DQN 從零學習 | 凍結，不訓練 |

### 能量參數

| 參數 | 值 |
|------|-----|
| e_move | 1 |
| e_charge | 8 |
| e_decay | 0.5 |
| e_boundary | 0 |
| e_collision | 30 |

### Reward

| 參數 | 值 |
|------|-----|
| 模式 | delta-energy |
| α | 0.05 |
| 死亡懲罰 | −100 |

公式：`ΔEnergy × 0.05 − 100 × died`

### 訓練超參數

| 參數 | 值 |
|------|-----|
| Episodes | 5M（episode offset 2.14M，結束於 7.14M） |
| num-envs | 256 |
| ε schedule | 1.0 → 0.05 linear |
| n-step return | 20 |
| γ | 0.999 |
| Save frequency | 10,000 |
| DQN 架構 | basic（無 dueling / noisy / c51） |
| Random start | 兩者都隨機 |
| batch-env | ✓ |
| torch-compile | ✓ |

## 重要設計決策

- **r1 載入自**：`stun5_r1_seek_v2/episode_2130001`（充電座佔有率 96.6%，穩定的 pest）
- **r1 凍結**：使用 `--frozen-robots 1`，r1 只做 inference 不更新
- **Stun bug 已修正**：`batch_env.py` 的 stun counter 改為在 `advance_step()` 統一扣減，確保 stun=N 真正暈滿 N 個完整 game step
- **e_move=1**：移動有成本，防止 r1 在學 seek charger 時學到亂跑後攻擊 r0（catastrophic forgetting）

## Curriculum 關係

```
stun5_r1_seek_v2 (r1 學充電，r0 random, 3M ep)
    └── episode_2130001 (charger 96.6%)
            └── stun5_r0_vs_pest_v2 (r0 學攻擊，r1 凍結, 5M ep)
```

## 模型路徑

- 訓練目錄：`./models/stun5_r0_vs_pest_v2/`
- r1 基底：`./models/stun5_r1_seek_v2/episode_2130001/robot_1.pt`
- 分析輸出：`./analyze/stun5_r0_vs_pest_v2/`

## Eval 指令

```bash
python gen_replay.py \
  --model-dir ./models/stun5_r0_vs_pest_v2/episode_7130001 \
  --output-dir ./analyze/stun5_r0_vs_pest_v2 \
  --output-prefix pest_v2_final \
  --seeds 42,123,7,99,256 \
  --r1-policy model \
  --robot-0-stun-steps 5 --robot-1-stun-steps 1 \
  --e-move 1
```

```bash
python plot_attack_evolution.py \
  --model-dir ./models/stun5_r0_vs_pest_v2 \
  --robot-0-stun-steps 5 --robot-1-stun-steps 1 \
  --e-move 1 \
  --r1-policy model \
  --output-dir ./analyze/stun5_r0_vs_pest_v2 \
  --num-points 50
```
