#!/bin/bash
set -e

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tag="A_SPP"
graph_path="z_ablation/results/MATH/round_5"
# 创建统一的实验文件夹
EXPERIMENT_DIR="${graph_path}/experiments/${tag}/${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

# 复制graph.py到实验目录
if [ -f "${graph_path}/graph.py" ]; then
    cp "${graph_path}/graph.py" "$EXPERIMENT_DIR/"
    cp "${graph_path}/prompt.py" "$EXPERIMENT_DIR/"
    echo "✅ 已复制 graph.py 和 prompt.py 到实验目录"
else
    echo "⚠️  警告: ${graph_path}/graph.py 不存在"
fi

# 日志文件路径
LOG_FILE="$EXPERIMENT_DIR/Run.log"

echo "🚀 开始实验，时间戳: $TIMESTAMP"
echo "📁 实验文件夹: $EXPERIMENT_DIR"
echo "📋 日志文件: $LOG_FILE"

# 确保日志文件可以创建
touch "$LOG_FILE" || {
    echo "❌ 无法创建日志文件: $LOG_FILE"
    exit 1
}

# 设置环境变量，让其他脚本知道当前实验文件夹
export EXPERIMENT_DIR="$EXPERIMENT_DIR"
export EXPERIMENT_TIMESTAMP="$TIMESTAMP"

echo "🔄 开始运行实验..."

# 使用更简单可靠的日志记录方式
{
  python3 -u run.py \
    --dataset MATH \
    --custom_data_path "/Users/codiplay/Documents/ustc_workspace/AFlow/data/datasets/math_test.jsonl" \
    --graph_path $graph_path \
    --batch_size 50 \
    "$@" 2>&1 | while IFS= read -r line; do
        echo "$line"
        # 移除ANSI颜色代码并写入日志文件
        echo "$line" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"
    done
}

# 检查Python脚本的退出状态
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "✅ 运行完成！"
echo "📁 所有实验文件已保存到: $EXPERIMENT_DIR"

# 显示生成的文件
echo "📊 生成的文件:"
if ls "$EXPERIMENT_DIR"/*.log >/dev/null 2>&1; then
    echo "   - 日志文件: $(ls "$EXPERIMENT_DIR"/*.log | xargs -n1 basename)"
fi
if ls "$EXPERIMENT_DIR"/*.csv >/dev/null 2>&1; then
    echo "   - CSV结果: $(ls "$EXPERIMENT_DIR"/*.csv | xargs -n1 basename)"
fi
if ls "$EXPERIMENT_DIR"/*.jsonl >/dev/null 2>&1; then
    echo "   - 失败样例: $(ls "$EXPERIMENT_DIR"/*.jsonl | xargs -n1 basename)"
fi
if [ -d "$EXPERIMENT_DIR/graph_backup" ]; then
    echo "   - Graph备份: graph_backup/"
fi

# 退出时使用Python脚本的退出代码
exit $PYTHON_EXIT_CODE 