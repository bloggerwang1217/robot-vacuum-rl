#!/bin/bash

# 使用 tmux 執行 epsilon decay 批次訓練

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SESSION_NAME="epsilon_decay_training"
SCRIPT_PATH="$SCRIPT_DIR/train_epsilon_decay.sh"

# 檢查訓練腳本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 錯誤: 找不到 train_epsilon_decay.sh"
    exit 1
fi

echo "=========================================="
echo "在 tmux 中啟動 Epsilon Decay 訓練"
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
tmux send-keys -t "$SESSION_NAME" "cd '$SCRIPT_DIR/..' && ./scripts/train_epsilon_decay.sh" Enter

echo ""
echo "✓ 訓練已在 tmux 背景執行中"
echo ""
echo "=========================================="
echo "連接方式："
echo "=========================================="
echo ""
echo "監看訓練過程："
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "脫離 tmux（Ctrl+b 然後 d）"
echo ""
echo "強制結束訓練："
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "查看所有 sessions："
echo "  tmux list-sessions"
echo ""
