以下是整理成可實作的**正式規格**版本，主軸是：

**所有結盟效果都作用在 energy / 血量變化層，reward function 本身不改。**
若不開啟此機制，系統行為必須與原版完全一致。這和你目前專案中「reward 由自身能量變化決定」的設計是一致的。

---

# Formal Specification: Optional Alliance Energy-Sharing Mechanism

## 1. Purpose

本規格定義一個**可選的 alliance energy-sharing 機制**，用於 3-agent bullying / alliance 實驗。

此機制的設計目標為：

1. 支援 3-agent 環境中，兩名 allied agents 對一名 outsider 的結盟互動。
2. 將 shared 機制集中在 **energy change** 層，而非直接共享 reward。
3. 保持既有 reward function 不變。
4. 確保與現有環境**完全向後相容**：

   * 若未啟用 alliance energy-sharing，行為必須與目前版本完全一致。
   * 舊有 2-agent 攻擊行為實驗不得因新增參數而報錯或改變結果。 

---

## 2. Design Principle

### 2.1 Reward function remains unchanged

Alliance 機制不得直接改寫 reward function。

每個 agent 的 reward 仍然只由**自己最終的 energy change** 決定。

### 2.2 Sharing occurs only at energy-transition level

所有共享效果只作用於特定事件造成的 energy gain / loss。

也就是：

* 先依原本環境規則計算各 agent 的 base energy changes
* 再依 alliance 規則做 energy-sharing
* 最後用每個 agent 自己的 final energy change 計算 reward

### 2.3 Optional and backward compatible

若未啟用此功能：

* 不應有任何行為差異
* 不應造成舊程式、舊實驗、舊腳本失效

---

## 3. Definitions

### 3.1 Base energy change

某 agent 在一個 step 中，依原始環境規則得到的 energy 變化，記為：

[
\Delta E_i^{base}
]

### 3.2 Event-specific energy change

將 base energy change 拆成不同事件來源：

[
\Delta E_i^{base}
=================

\Delta E_i^{charge}
+
\Delta E_i^{collision}
+
\Delta E_i^{move}
+
\Delta E_i^{decay}
+
\Delta E_i^{boundary}
+
\Delta E_i^{other}
]

其中：

* (\Delta E_i^{charge})：來自充電的 energy gain
* (\Delta E_i^{collision})：來自被撞 / knockback 的 energy loss
* (\Delta E_i^{move})：來自移動成本
* (\Delta E_i^{decay})：來自每步自然衰減
* (\Delta E_i^{boundary})：來自撞牆等邊界懲罰
* (\Delta E_i^{other})：其他未分類 energy 事件

### 3.3 Final energy change

套用 alliance energy-sharing 後，每個 agent 的最終 energy 變化，記為：

[
\Delta E_i^{final}
]

### 3.4 Alliance

Alliance 是一組 robot IDs，表示這些 agents 屬於同一隊，並會共享指定事件造成的 energy changes。

---

## 4. Reward Definition

Reward function 必須維持與原本一致：

[
r_i = c \cdot \Delta E_i^{final} - D \cdot \mathbf{1}[\text{agent } i \text{ died}]
]

其中：

* (c) 為 energy-to-reward scaling constant
* (D) 為死亡懲罰常數
* (\mathbf{1}[\cdot]) 為 indicator function

若沿用目前 README 中的設定，則為：

[
r_i = 0.05 \cdot \Delta E_i^{final} - 100 \cdot \mathbf{1}[\text{died}_i]
]

注意：

* alliance 機制只影響 (\Delta E_i^{final})
* reward function 公式本身不可因 alliance 而改寫

---

## 5. Energy-Sharing Mechanism

## 5.1 Supported scope in first version

第一版只需支援：

* 一組 alliance
* alliance 大小為 2
* outsider 可為第 3 個 agent
* 只共享指定事件的 energy changes

也就是典型的：

* robot_0, robot_1 為 allies
* robot_2 為 outsider

---

## 5.2 Shared events

第一版建議只支援以下兩種 event：

* `charge`
* `collision`

不共享：

* `move`
* `decay`
* `boundary`
* `other`

理由是：

* `charge` 與 `collision` 最具 alliance / bullying 意義
* 其他項目屬於日常個體成本，不應在第一版共享

---

## 5.3 Sharing weights

對於 alliance 中兩名 agent (i) 與 (j)，共享權重定義為：

* self weight: (w_{self})
* ally weight: (w_{ally})

第一版預設值：

[
w_{self} = \frac{2}{3}, \quad w_{ally} = \frac{1}{3}
]

即採用你要的 2:1 規則。

---

## 5.4 Event-only energy sharing rule

若某 agent (i) 與 (j) 為同盟，則共享事件池定義為：

[
S_i = \Delta E_i^{charge} + \Delta E_i^{collision}
]
[
S_j = \Delta E_j^{charge} + \Delta E_j^{collision}
]

共享後：

[
\tilde{S}*i = w*{self} S_i + w_{ally} S_j
]
[
\tilde{S}*j = w*{self} S_j + w_{ally} S_i
]

而各自最終 energy change 為：

[
\Delta E_i^{final}
==================

\tilde{S}_i
+
\Delta E_i^{move}
+
\Delta E_i^{decay}
+
\Delta E_i^{boundary}
+
\Delta E_i^{other}
]

[
\Delta E_j^{final}
==================

\tilde{S}_j
+
\Delta E_j^{move}
+
\Delta E_j^{decay}
+
\Delta E_j^{boundary}
+
\Delta E_j^{other}
]

outsider (k) 則不做共享：

[
\Delta E_k^{final} = \Delta E_k^{base}
]

---

## 6. Behavioral Interpretation

此機制的直觀意義如下：

### 6.1 Shared charging gain

若 allied teammate 充電獲得能量，另一位 ally 也獲得部分能量收益。

這表示：

* 隊友成功佔住 charger，對自己也有實質利益

### 6.2 Shared collision loss

若 allied teammate 因碰撞受損，另一位 ally 也承受部分能量損失。

這表示：

* 隊友被打不是個體私事，而是聯盟共同成本

### 6.3 Reward still remains individual

儘管 energy 有 sharing，每個 agent 最後仍只根據**自己的 final energy** 取得 reward。

因此 alliance 表現為：

* 共享生命利益與風險
* 但不直接共享 reward signal

---

## 7. CLI Parameters

## 7.1 Alliance definition

### `--alliance-groups`

定義同盟群組。

格式：

```bash
--alliance-groups "0,1"
```

第一版限制：

* 最多一組
* 每組恰有兩個 robot IDs

預設：

* `None`

語意：

* 若未提供，表示不存在 alliance

---

## 7.2 Sharing mode

### `--energy-sharing-mode`

可選值：

* `none`
* `event_only`

預設：

```bash
--energy-sharing-mode none
```

語意：

#### `none`

* 不啟用 alliance energy-sharing
* 系統必須與原版完全一致

#### `event_only`

* 僅對指定事件的 energy changes 做 sharing

---

## 7.3 Shared event types

### `--energy-sharing-events`

逗號分隔字串列表。

第一版可選值：

* `charge`
* `collision`
* `charge,collision`

預設：

```bash
--energy-sharing-events charge,collision
```

僅在 `energy-sharing-mode=event_only` 時生效。

---

## 7.4 Sharing weights

### `--energy-sharing-self-weight`

float
預設：

```bash
0.6666667
```

### `--energy-sharing-ally-weight`

float
預設：

```bash
0.3333333
```

僅在 `energy-sharing-mode != none` 時生效。

---

## 8. Backward Compatibility Requirements

以下情況必須與現有系統完全一致。

### Case A: no new parameters

```bash
python train_dqn_vec.py ...
```

### Case B: sharing mode explicitly disabled

```bash
python train_dqn_vec.py ... --energy-sharing-mode none
```

### Case C: disabled mode with extra sharing arguments

```bash
python train_dqn_vec.py \
  ... \
  --energy-sharing-mode none \
  --alliance-groups "0,1" \
  --energy-sharing-self-weight 0.6666667 \
  --energy-sharing-ally-weight 0.3333333
```

即使帶了 alliance / sharing 相關參數，只要 mode 是 `none`：

* 不應套用 sharing
* 不應報錯
* 輸出結果必須與原版一致

---

## 9. Error Handling

## 9.1 Non-strict default behavior

預設採用寬鬆容錯：

* 若 `energy-sharing-mode=none`，則所有 sharing 參數都可被安全忽略
* 若未指定 `alliance-groups`，則退化為 no sharing
* 不應因 sharing 參數缺失而 crash

## 9.2 Invalid alliance configuration

第一版若 `alliance-groups` 格式不合法，可採兩種策略擇一：

### Option A

直接 warning 並退化為 no sharing

### Option B

直接 raise error

若你要完全不影響舊實驗，我建議第一版採 **Option A**。

---

## 10. Implementation Requirements

## 10.1 No modification to existing physics rules

以下既有規則不得因 alliance sharing 而改變：

* movement
* collision / knockback
* stun
* docking
* charging eligibility
* exclusive charging
* death condition

Alliance sharing 只能是這些規則運算完成後的**energy post-process layer**。

---

## 10.2 Required internal decomposition

環境在 step 過程中，至少必須能取得每位 agent 的：

* `delta_e_charge[i]`
* `delta_e_collision[i]`
* `delta_e_move[i]`
* `delta_e_decay[i]`
* `delta_e_boundary[i]`
* `delta_e_other[i]`

若目前尚未顯式分解，需於 energy 計算過程中增加中間記錄。

---

## 10.3 Application order

每一步建議按以下順序執行：

1. 依原始環境規則更新位置、碰撞、充電、死亡等狀態
2. 計算每位 agent 的 base event-level energy changes
3. 若 `energy-sharing-mode=event_only` 且存在合法 alliance：

   * 對指定事件做 sharing
4. 合成每位 agent 的 final energy change
5. 用 final energy change 計算 reward
6. 回傳 observations, rewards, done flags, info

---

## 11. Example Calculation

假設：

* r0 與 r1 為 alliance
* 權重為 2:1
* 本 step 中：

[
\Delta E_0^{charge}=+8,\quad \Delta E_0^{collision}=0
]
[
\Delta E_1^{charge}=0,\quad \Delta E_1^{collision}=-30
]

則：

[
S_0 = +8
]
[
S_1 = -30
]

共享後：

[
\tilde{S}_0 = \frac{2}{3}(8) + \frac{1}{3}(-30) = \frac{16}{3} - 10 = -\frac{14}{3}
]

[
\tilde{S}_1 = \frac{2}{3}(-30) + \frac{1}{3}(8) = -20 + \frac{8}{3} = -\frac{52}{3}
]

再加回各自非共享項，得到 final energy changes。

此例表示：

* r0 雖然自己充電，但因隊友被重擊，仍承受聯盟連坐成本
* r1 雖然自己受重傷，但也部分分享隊友的充電收益

這種機制正是 alliance 共同命運的表現。

---

## 12. Logging Requirements

為了驗證 alliance 機制是否有效，建議至少記錄：

### Per-agent diagnostics

* `robot_i/delta_e_base`
* `robot_i/delta_e_final`
* `robot_i/delta_e_charge`
* `robot_i/delta_e_collision`
* `robot_i/delta_e_shared_component`

### Alliance diagnostics

* `ally_pair/shared_energy_given`
* `ally_pair/shared_energy_received`
* `ally_pair/ally_ally_collision_count`
* `ally_pair/ally_outsider_collision_count`

### Behavior diagnostics

* outsider 被攻擊率
* outsider 死亡率
* allies 互打率
* allies 對 outsider 的 joint pressure 指標

---

## 13. Recommended First Experimental Settings

## 13.1 Baseline

完全不開 sharing：

```bash
python train_dqn_vec.py \
  ... \
  --num-robots 3 \
  --energy-sharing-mode none
```

## 13.2 Alliance bullying experiment

```bash
python train_dqn_vec.py \
  ... \
  --num-robots 3 \
  --alliance-groups "0,1" \
  --energy-sharing-mode event_only \
  --energy-sharing-events charge,collision \
  --energy-sharing-self-weight 0.6666667 \
  --energy-sharing-ally-weight 0.3333333
```

---

## 14. Minimal Acceptance Tests

### Test 1: disabled mode equals original

條件：

* `energy-sharing-mode=none`

要求：

* `delta_e_final == delta_e_base`
* rewards 與舊版完全一致

### Test 2: charge sharing only

條件：

* r0 charge = +8
* r1 charge = 0
* sharing events = charge

要求：

* r0 最終 charge component = (2/3 \cdot 8)
* r1 最終 charge component = (1/3 \cdot 8)

### Test 3: collision sharing only

條件：

* r0 collision = -30
* r1 collision = 0
* sharing events = collision

要求：

* r0 最終 shared collision component = (-20)
* r1 最終 shared collision component = (-10)

### Test 4: no alliance groups

條件：

* mode = `event_only`
* 無 `alliance-groups`

要求：

* 不 crash
* 退化為 no sharing

### Test 5: outsider unaffected

條件：

* r0, r1 為 allies
* r2 為 outsider

要求：

* r2 的 final energy change 恆等於 base energy change

---

## 15. Out-of-Scope for First Version

以下內容不屬於第一版必要範圍：

* 直接 reward sharing
* energy full-step sharing
* 多組 alliance
* alliance size > 2
* teammate-to-teammate transfer action
* 多層級社會結構
* dynamic alliance formation / alliance breaking

---

## 16. Summary Statement

本規格的核心定義可總結為：

> Allied agents do not directly share rewards.
> Instead, they partially share energy gains and losses from selected events, while each agent’s reward is still computed solely from its own final energy change.

這樣的設計保留了你原本系統「reward 由自身能量變化驅動」的結構，只把 alliance 機制放在血量 / energy transition 層，因此比直接 shared reward 更乾淨，也更容易與原始攻擊行為實驗並存。

如果你要，我下一步可以把這份正式規格再壓成一版 **工程實作版 spec**，直接列成「要改哪些 argparse、哪些內部變數、step 裡哪個位置插入」。
