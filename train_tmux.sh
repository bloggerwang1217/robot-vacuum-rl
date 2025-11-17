#!/bin/bash

# 使用 tmux 執行批次訓練腳本
# 可以安全地關閉電腦，訓練會在背景繼續執行

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SESSION_NAME="robot_vacuum_training"
SCRIPT_PATH="$SCRIPT_DIR/train_batch.sh"

# 檢查是否已安裝 tmux
if ! command -v tmux &> /dev/null; then
    echo "❌ 錯誤: 未安裝 tmux"
    echo ""
    echo "請先安裝 tmux："
    echo "  brew install tmux"
    exit 1
fi

# 檢查訓練腳本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 錯誤: 找不到 train_batch.sh"
    exit 1
fi

echo "=========================================="
echo "在 tmux 中啟動批次訓練"
echo "=========================================="
echo ""

# 檢查是否已有相同名稱的 session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  已存在名稱為 '$SESSION_NAME' 的 tmux session"
    echo ""
    read -p "是否要重新啟動？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "已終止舊的 session"
    else
        echo "已取消"
        exit 1
    fi
fi

# 建立新的 tmux session 並執行訓練腳本
echo "建立 tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# 在 session 中執行訓練腳本
tmux send-keys -t "$SESSION_NAME" "cd '$SCRIPT_DIR' && ./train_batch.sh" Enter

echo ""
echo "✓ 訓練已在 tmux 背景執行中"
echo ""
echo "=========================================="
echo "連接方式："
echo "=========================================="
echo ""
echo "方式1：立即連接（監看訓練過程）"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "方式2：稍後連接（例如下次開機）"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "方式3：在 tmux 中打開新視窗監看"
echo "  tmux new-window -t $SESSION_NAME"
echo ""
echo "=========================================="
echo "其他有用指令："
echo "=========================================="
echo ""
echo "查看所有 tmux sessions："
echo "  tmux list-sessions"
echo ""
echo "監看即時日誌："
echo "  tail -f '$SCRIPT_DIR/models/batch_training_*/*/training.log'"
echo ""
echo "強制結束訓練（如有需要）："
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "從 tmux 中脫離（按 Ctrl+b 再按 d）"
echo ""
