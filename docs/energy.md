可以，下面我直接幫你開成一份 **Stage 2 主實驗 spec**。
這份 spec 的原則是：

* **保留你原本的多人共享充電規則**
* **沿用你現在已經有的 type-control 管線**
* **只額外加入一個可開關的局部 type-based charging penalty**
* **不改 reward 公式，只改實際 charging dynamics**

你現在的 README 已經有 `--agent-types-mode observe`、`--triangle-agent-id` 這套 control 管線，也有 `--exclusive-charging` 與原本充電 / reward 設計，所以這份 spec 是接在那個基礎上最自然的下一步。 

---

# Spec：Heterotype Local Charging Penalty

## 1. 目的

檢驗在 4-agent、`3 circle + 1 triangle` 的設定下，
若**異類同時佔據同一個充電區會降低充電效率**，是否會自然誘發以下兩階段行為：

1. 三個 circle 先傾向排除 triangle
2. triangle 消失後，circle 之間再回到彼此競爭

這個設計不是直接給「霸凌 reward」，而是讓 **type 第一次在資源控制上具有策略意義**。

---

## 2. 設計原則

### 2.1 最小侵入

不改你原本的：

* reward 公式
* observation 主結構
* DQN / replay buffer / n-step 訓練流程
* collision / death / movement 規則

### 2.2 沿用既有 type 管線

沿用你現有的：

* `--agent-types-mode off|observe`
* `--triangle-agent-id`
* replay JSON 記錄 agent type 的做法 

### 2.3 只改 charging dynamics

你原本 reward 來自 `ΔEnergy`，所以只要改充電時實際補到的能量，reward 會自動反映，不需要再額外寫 type-specific reward。

---

## 3. 新增 CLI 參數

### 3.1 核心開關

`--heterotype-charge-mode`

可選值：

* `off`
* `local-penalty`

### 3.2 懲罰強度

`--heterotype-charge-factor`

* 型別：`float`
* 預設：`1.0`

語意：

* `1.0`：不折減
* `0.7`：mixed-type 同區時，充電效率變成原本的 70%
* `0.5`：mixed-type 同區時，充電效率變成原本的 50%

---

## 4. 啟用條件

本 spec 建議在以下情況使用：

* `num_robots = 4`
* `agent-types-mode = observe`
* `triangle-agent-id = k` 或每 episode 輪替一個 triangle
* `heterotype-charge-mode = local-penalty`

### 額外建議

這個實驗的第一版 **不要搭配 `--exclusive-charging`**，因為 `exclusive-charging` 會讓只要還有其他存活 robot，充電直接失效；那樣 type-based local penalty 幾乎發揮不了作用。你的 README 也明確定義了 `--exclusive-charging` 是全域型的充電封鎖。

---

## 5. 與現有多人共享充電規則的整合

你原本的多人共享規則是：

* 同一個 charger 區域內若有 `n` 台 robot，則每台分到 `e_charge / n`

這份 spec **保留這條規則不變**，只在 mixed-type occupancy 時額外乘上一個係數。

### 原本

[
charge_gain_i = \frac{e_charge}{n}
]

### 新版

[
charge_gain_i = \frac{e_charge}{n} \times f
]

其中：

* 若該 charger 內所有 occupants 都是同 type
  [
  f = 1.0
  ]

* 若該 charger 內同時出現 circle 與 triangle
  [
  f = \text{heterotype_charge_factor}
  ]

---

## 6. 精確規則

對每一個 charger，於每個 timestep 單獨計算：

### Step 1

找出該 charger 作用範圍內所有存活 robot，記為 occupancy set。

### Step 2

令：

* `n = occupancy 數量`
* `types_present = occupancy 中出現的 type 集合`

### Step 3

若 `n == 0`：

* 無人充電

若 `n > 0`：

* base charge = `e_charge / n`

### Step 4

若 `heterotype-charge-mode == off`：

* 每台得到 `base charge`

若 `heterotype-charge-mode == local-penalty`：

* 若 `len(types_present) == 1`

  * 每台得到 `base charge`
* 若 `len(types_present) >= 2`

  * 每台得到 `base charge * heterotype_charge_factor`

---

## 7. 重要語意

### 7.1 懲罰是對「混合佔據」本身，不是對 triangle

這條很重要。

也就是：

* ○○ 共享：正常
* △△ 共享：正常
* ○△ 共享：折減
* ○○△ 共享：折減

這樣你表達的是：
**異質群體在高價值資源區的共存效率較差**

而不是：
**triangle 天生該被懲罰**

### 7.2 懲罰是局部的，不是全域的

只看同一個 charger 的 occupancy composition。
別的 charger 不受影響。

---

## 8. 不改動項目

以下項目必須與 baseline / control 保持一致：

* reward 公式 `r_t = ΔE × 0.05 + ... - 100·died` 
* movement / collision / boundary / death 邏輯
* replay buffer
* n-step
* DQN 結構
* observation 中 type 欄位的編碼方式
* agent speed / initial energy / map size（除非你另做條件）

---

## 9. Observation

Observation 規格不需再改。

沿用你現在 control 組的 type observation：

* `agent-types-mode off`：padding 0
* `agent-types-mode observe`：填入實際 type 值（circle=0, triangle=1）

也就是說，這個 Stage 2 與 control 的差別不在 observation，
而在 **charging dynamics**。

---

## 10. Logging

建議新增以下 replay / debug 欄位，方便之後分析：

### Episode-level

* `agent_types`
* `heterotype_charge_mode`
* `heterotype_charge_factor`

### Step-level（可選）

對每個 charger 記錄：

* `occupants`
* `occupant_types`
* `occupancy_count`
* `is_mixed_type`
* `charge_factor_applied`

這樣之後你可以很清楚回放：

* triangle 是否真的常在混住時被排擠
* circle 是否會先把 triangle 從 charger 周圍驅離

---

## 11. 實驗假說

### H1

在 `local-penalty` 啟用後，triangle 的平均存活時間會下降。

### H2

在 triangle 存活期間，circle 對 triangle 的接近 / 碰撞 / 包圍行為會高於 circle 對 circle。

### H3

triangle 死亡後，circle 之間的攻擊 / 競爭行為會上升。

### H4

相較於 control 組，mixed-type charger occupancy 的持續時間會縮短，表示 agent 傾向避免異類共站資源區。

---

## 12. 最小分析指標

### 12.1 Triangle survival time

triangle 的平均存活步數。

### 12.2 Type-conditioned aggression

分開統計：

* circle → triangle 的碰撞率
* circle → circle 的碰撞率

### 12.3 Joint pressure frequency

triangle 存活時，是否常有 2 個以上 circle 同時靠近 triangle。

### 12.4 Mixed charger occupancy ratio

在 charger 範圍內，mixed-type occupancy 出現的比例與持續時間。

### 12.5 Post-triangle civil-war indicator

triangle 死後，circle 間碰撞 / 追逐是否上升。

---

## 13. 對照組

你至少要保留兩個對照：

### Control A

* `agent-types-mode observe`
* `heterotype-charge-mode off`

用途：

* 驗證只有 type observation 本身不足以穩定誘發霸凌

### Main

* `agent-types-mode observe`
* `heterotype-charge-mode local-penalty`
* `heterotype-charge-factor = 0.7`（先試）

用途：

* 檢驗 type 在資源區混住成本下是否變得有策略意義

---

## 14. 參數建議

第一輪我建議先試：

* `heterotype-charge-factor = 0.7`

理由：

* 夠溫和
* 不會一開始就把 mixed-type 共存打成完全不可行
* 比較像「摩擦」而不是「禁令」

若效果太弱，再試：

* `0.5`

先不要直接上：

* `0.2` 或 `0.0`

因為那會太像硬編規則。

---

## 15. CLI 範例

### Control 組

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 4 \
  --agent-types-mode observe \
  --triangle-agent-id 2 \
  --charger-positions 2,2 \
  --heterotype-charge-mode off \
  --wandb-run-name type_control
```

### Stage 2 主實驗

```bash
python train_dqn_vec.py \
  --env-n 5 --num-robots 4 \
  --agent-types-mode observe \
  --triangle-agent-id 2 \
  --charger-positions 2,2 \
  --heterotype-charge-mode local-penalty \
  --heterotype-charge-factor 0.7 \
  --wandb-run-name type_local_penalty
```

---

## 16. 工程實作建議

你大概只需要碰三塊：

### `train_dqn_vec.py`

* 加 CLI 參數
* 存進 env config

### `robot_vacuum_env.py`

* 在 charging 結算時判斷 occupancy types
* 套用 `heterotype_charge_factor`

### `README.md`

新增一節：

* `Heterotype Local Charging Penalty Experiment`

---

## 17. 一句話版定義

**保留原本的多人共享充電 `e_charge / n`，但若同一 charger 內出現 mixed types，則共享充電再乘上一個小於 1 的局部效率係數；除此之外，其餘 reward、obs、訓練與物理規則皆不變。**

如果你要，我下一步可以直接幫你把這份 spec 再往下寫成「程式修改清單」，例如每個檔案要新增哪些欄位、哪段 charging code 要怎麼改。
