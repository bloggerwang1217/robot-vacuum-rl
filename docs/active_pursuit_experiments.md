# 主動追殺實驗總結

## 研究目標
讓 r0 學會**主動離開充電座去追殺 r1**，而不是被動等 r1 走過來。
Reward = `ΔEnergy × 0.05 - 100 × died`（無明確殺人獎勵）

## 不可變更的實驗約束

以下為固定規則，所有實驗設計必須遵守：

1. **`charger_range = 0`**：充電座無感應範圍，機器人必須站在正上方才能充電（貼近現實）
2. **雙方都是 trained agent**：不允許任何一方使用腳本（scripted / flee / seek-charger / random），兩邊都必須透過 DQN 學習
3. **無明確殺人獎勵**：reward 只包含 delta-energy 和死亡懲罰，攻擊行為必須是 emergent behavior

---

## 唯一成功案例：Round 4（charger_range=1）

| 參數 | 值 |
|------|-----|
| Grid | 5×5 |
| Charger | (2,2), **charger_range=1** |
| e_charge / e_decay | 8 / 5 |
| e_collision | 100（一撞必殺） |
| 雙方 HP | 100 |
| Speed | 1 / 1（同速） |
| 起始位置 | **隨機** |
| DQN 類型 | **Plain DQN**（無 C51/Dueling/Noisy） |
| 結果 | **83% kill rate, 80.9% far-start kill rate** |

### 為什麼成功
1. **charger_range=1** → 充電座影響範圍 3×3（9 格），兩人同時在範圍內 = 共享充電
2. **共享充電不可行**：`e_charge/2 - e_decay = 8/2 - 5 = -1/step`（淨虧損）
3. **獨佔充電可行**：`e_charge - e_decay = 8 - 5 = +3/step`（淨賺）
4. 殺人 = 唯一的長期生存策略
5. **隨機起始位置** → 有時雙方離充電座很遠，必須「先導航到對手附近 → 殺」

### 侷限
- ~30% seed 才出現 >50% kill rate（co-adaptation 導致不穩定）
- 250k 是 peak，之後下降（Red Queen cycling）

---

## 所有失敗的「主動追殺」嘗試

### A. Pest 系列（scripted r1 = seek-charger）
| 實驗 | r0 速度 | r1 行為 | 結果 | 為什麼失敗 |
|------|--------|---------|------|-----------|
| pest_spd1_v3 | 1 | seek-charger | 被動碰撞 | r0 蹲充電座，r1 自己走過來送死 |
| pest_spd2_v3 | 2 | seek-charger | 被動碰撞 | 同上，只是更快到充電座 |
| pest_spd3_v3 | 3 | seek-charger | 被動碰撞 | 同上 |

**結論**：seek-charger 腳本 = r1 永遠走向 r0 = 不需要主動追殺

### B. Mosquito 系列（r1 trained, speed 不對稱）
| 實驗 | r0 / r1 速度 | e_decay | 結果 | 為什麼失敗 |
|------|-------------|---------|------|-----------|
| M2-mosquito-v2 | 1 / 2 | 0.5 | r1 反過來騷擾 r0 | r1 太快，r0 追不到 |
| M2-hunt-r0fast | 2 / 1 | 0.5 | 待分析 | r0 太快，不用殺就能壟斷 |

**結論**：速度不對稱 → 快的一方不需要殺慢的，壟斷就夠了

### C. 投機充電腳本系列（r1 = opportunistic charger）
| 實驗 | e_decay | 結果 | 為什麼失敗 |
|------|---------|------|-----------|
| M4-opp-charger | 0.5 | 和平獨裁（0 碰撞）| r0 先到充電座，r1 永遠充不到電自然死 |
| M6-opp-high-decay | 2.0 | 和平獨裁（0 碰撞）| r1 30 步就 decay 死，r0 不用動手 |

**結論**：r0 speed=2 + charger_range=0 → r0 可以物理封鎖充電座，r1 無法充電

### D. 高 Decay 雙方學習
| 實驗 | e_decay | 結果 | 為什麼失敗 |
|------|---------|------|-----------|
| M5-high-decay | 2.0 | 偶發 on-charger 碰撞 | hunt:0/1, pursuit ratio=0.16，被動防守 |

### E. 對照組（預期結果，驗證成功）
| 實驗 | r1 行為 | 結果 | 意義 |
|------|---------|------|------|
| M3a-flee | flee 腳本 | 和平獨裁（0 碰撞）| ✅ r1 不來 → r0 不殺 |
| M3b-no-speed | 同速 both learn | 被動碰撞（1 次）| ✅ r1 走來送死 |

### F. 更早期的失敗（Round 1-3）
| Round | 設定 | 結果 | 原因 |
|-------|------|------|------|
| R1 | 3×3, range=0 | 和平獨裁 | 太小，充電座自然排他 |
| R2 | 5×5, decay=3 | 雙方快速死亡 | decay 太猛 |
| R3B | r1 random, 5×5 | 50% 偶發碰撞 | r0 蹲充電座不動 |
| R3D | both learn, shared viable | 完美和平 | 共享充電可行，不需要殺 |

---

## 核心問題分析

### charger_range=0 vs charger_range=1

| | charger_range=0 | charger_range=1 |
|--|----------------|----------------|
| 充電條件 | 必須站在充電座**正上方** | 在充電座**周圍 3×3** 即可 |
| 共享充電 | 不可能（同一格只能一人充）| 可能（兩人都在 3×3 內）|
| r1 干擾頻率 | 極低（1/25 = 4%）| 高（9/25 = 36%）|
| r0 可以封鎖？ | 是（站上去就行）| 難（3×3 區域太大）|
| 成功案例 | **無** | **Round 4: 83% kill rate** |

### 為什麼 charger_range=0 下主動追殺極難

1. **r0 站上充電座 = 完美防守**：charger_range=0 意味著只有精確站在上面才能充電。r0 站上去後，物理上封鎖了唯一資源。
2. **r1 的干擾成本太低**：r1 必須站到完全相同的格子才能干擾 r0，隨機走的機率只有 4%。
3. **殺人的機會成本 > 收益**：離開充電座追殺 = 放棄充電，而 r1 幾乎不構成威脅。

### 為什麼 charger_range=1 可以

1. **r0 無法單獨封鎖整個 3×3 區域**
2. **r1 隨機走入 3×3 區域的機率高（36%）**→ 持續干擾
3. **共享充電虧損**（-1/step）→ 殺掉 r1 才能正收益
4. **被動等待 = 持續虧損**，主動追殺 = 一次投資永久獲利

---

## 還沒嘗試但可能有效的方向

### 1. charger_range=0 + 修改充電機制
讓 r1 **活著**本身對 r0 造成持續損害：
- **吸能量**：r0 撞 r1 時吸取 r1 的能量（環境物理，非 reward hack）
- **充電干擾**：r1 存活 → 充電效率降低（如 charge_rate = base / n_alive）
- **能量掉落**：r1 死後原地掉落能量，r0 可撿取

### 2. charger_range=0 但多害蟲
- 3-4 隻隨機 r1 → 干擾機率從 4% 升到 ~16%
- r0 需要清除所有害蟲才能安心充電

### 3. 接受 charger_range=1 的結果
- Round 4 已經證明主動追殺可以出現
- charger_range=1 的物理解釋：充電座有感應範圍，多機器人靠近時干擾充電效率

### 4. 純 charger_range=0 + e_decay > e_charge/2 + 同速 + 隨機起始 + Plain DQN
- 完全複製 Round 4 的參數，只把 charger_range 從 1 改成 0
- 看 kill rate 是否仍然 >0%（可能不行，但值得確認）

---

## 追殺指標（已實作）

訓練 log 格式：
```
[Episode N] Steps:S | r0:X | r1:Y | Collisions(A;B) | r0_alive r1@Z hunt:3/5✓ pr:0.72 | ε:0.XXX
```

| 指標 | 定義 | 主動追殺 | 被動防守 |
|------|------|---------|---------|
| `hunt:X/Y` | X=非充電座碰撞 / Y=總碰撞 | X ≈ Y | X ≈ 0 |
| `✓` | 致命一擊在非充電座 | 有 | 無 |
| `pr:` | pursuit ratio（靠近 r1 的步數比）| >0.65 | <0.4 |

wandb: `pursuit/off_charger_hits`, `pursuit/off_charger_kill`, `pursuit/r0_pursuit_ratio`
