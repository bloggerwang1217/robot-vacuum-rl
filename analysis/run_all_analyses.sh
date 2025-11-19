#!/bin/bash
# 一鍵執行所有分析腳本

set -e  # 遇到錯誤時停止

echo "=========================================="
echo "執行所有訓練分析工具"
echo "=========================================="

# 確保在正確的目錄
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${1:-analysis_output}"
WANDB_DIR="${2:-wandb_data}"

echo ""
echo "輸出目錄: $OUTPUT_DIR"
echo "WandB 資料目錄: $WANDB_DIR"
echo ""

# 建立輸出目錄
mkdir -p "$OUTPUT_DIR"

# 1. 執行配置分析
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1/4: 執行配置分析..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python analysis/analyze_per_config.py \
    --wandb-dir "$WANDB_DIR" \
    --output "$OUTPUT_DIR"

# 2. 執行擊殺與充電關係分析
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2/4: 執行擊殺與充電關係分析..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python analysis/analyze_kills_charges.py \
    --wandb-dir "$WANDB_DIR" \
    --output "$OUTPUT_DIR"

# 3. 執行非零充電分析
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3/4: 執行非零充電分析..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python analysis/analyze_nonzero_charges.py \
    --wandb-dir "$WANDB_DIR" \
    --output "$OUTPUT_DIR"

# 4. 執行訓練日誌分析（如果有日誌檔案）
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4/4: 執行訓練日誌分析..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "models" ]; then
    LOG_COUNT=$(find models -name "training.log" | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        echo "找到 $LOG_COUNT 個訓練日誌檔案"
        python analysis/analyze_training.py \
            --log-dir models \
            --output-dir "$OUTPUT_DIR" \
            --compare
    else
        echo "⚠️  未找到訓練日誌檔案，跳過此步驟"
    fi
else
    echo "⚠️  未找到 models 目錄，跳過此步驟"
fi

echo ""
echo "=========================================="
echo "✅ 所有分析完成！"
echo "=========================================="
echo "結果已儲存至: $OUTPUT_DIR"
echo ""
echo "您可以使用以下命令查看結果："
echo "  open $OUTPUT_DIR  # macOS"
echo "或直接瀏覽該目錄的圖片檔案"
echo ""
