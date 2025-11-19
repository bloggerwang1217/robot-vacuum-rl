# 訓練分析工具

這個資料夾包含了用於分析機器人吸塵器強化學習訓練結果的工具。

## 檔案說明

### 1. `analyze_training.py`
分析訓練日誌檔案（training.log），產生訓練過程的視覺化圖表。

**功能：**
- 解析訓練日誌中的指標（獎勵、存活率、碰撞、擊殺等）
- 繪製時間序列圖表（含平滑曲線）
- 分析學習階段和相關性
- 比較不同訓練執行的結果

**使用方式：**
```bash
# 分析單一日誌檔案
python analyze_training.py --log-file path/to/training.log

# 分析整個目錄的所有日誌
python analyze_training.py --log-dir models/batch_training_xxx/

# 比較多個訓練執行
python analyze_training.py --log-dir models/ --compare
```

---

### 2. `analyze_kills_charges.py`
專注分析擊殺（kills）與非主場充電站使用（non-home charges）之間的關係。

**功能：**
- 散點圖：擊殺 vs 非主場充電
- 時間序列：訓練過程中兩者的演變
- 相關性分析
- 配置間的比較

**使用方式：**
```bash
# 使用預設設定
python analyze_kills_charges.py

# 自訂資料來源和輸出目錄
python analyze_kills_charges.py --wandb-dir wandb_data --output analysis_output
```

---

### 3. `analyze_nonzero_charges.py`
分析有使用充電站的回合（non-home charges > 0）與沒有使用的回合之間的差異。

**功能：**
- 比較有/無使用充電站的回合表現
- 顯示使用充電站最多的 Top N 回合
- 充電站使用的分布圖
- 時間軸上的充電事件追蹤

**使用方式：**
```bash
# 基本分析
python analyze_nonzero_charges.py

# 顯示 Top 30 的充電回合
python analyze_nonzero_charges.py --top-n 30
```

---

### 4. `analyze_per_config.py`
為每個訓練配置產生詳細的綜合分析圖表（12 個子圖）。

**功能：**
- 為每個配置產生一張大型綜合分析圖
- 包含分布、散點圖、時間序列、相關性矩陣等
- 統計摘要表格
- 有/無充電的分組比較

**使用方式：**
```bash
# 分析所有配置
python analyze_per_config.py

# 只分析特定配置
python analyze_per_config.py --config collision-50-energy-100-epsilon-decay
```

---

## 資料來源

所有分析工具預設使用以下資料來源：
- **WandB 資料：** `../wandb_data/` 目錄中的 CSV 檔案
- **訓練日誌：** `../models/` 目錄中的 training.log 檔案
- **輸出目錄：** `../analysis_output/` （預設）

## 輸出

所有視覺化圖表會儲存為高解析度 PNG 檔案（300 DPI），統計摘要會在終端機中顯示。

## 依賴套件

```bash
pandas
matplotlib
seaborn
numpy
scipy
```

確保已安裝所有依賴：
```bash
pip install pandas matplotlib seaborn numpy scipy
```

## 快速開始

```bash
# 1. 確保在專案根目錄
cd /Users/ihao/Desktop/lab/robot-vaccum-rl

# 2. 執行完整分析
python analysis/analyze_per_config.py
python analysis/analyze_kills_charges.py
python analysis/analyze_nonzero_charges.py

# 3. 如果有訓練日誌，也可以分析日誌
python analysis/analyze_training.py --log-dir models/
```

## 提示

- 使用 `--help` 查看每個工具的完整選項
- 可以自訂輸出目錄避免覆蓋先前的分析結果
- 建議先執行 `analyze_per_config.py` 獲得整體概覽
