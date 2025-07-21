#!/bin/bash
#SBATCH --job-name=seed_hunt_v3
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/seed_hunt_v3_%j.out
#SBATCH --error=results/logs/seed_hunt_v3_%j.err

echo "🎯 种子猎人计划V3启动 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

# 使用系统Python，安装必要包到用户目录
echo "🐍 使用系统Python: $(python --version)"

echo "📦 安装必要包到用户目录..."
pip install --user numpy torch torchvision scikit-learn h5py munkres scipy

# 检查安装结果
echo "📦 检查包安装:"
python -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>/dev/null || echo "  ❌ numpy failed"
python -c "import torch; print(f'  ✅ torch {torch.__version__}')" 2>/dev/null || echo "  ❌ torch failed"
python -c "import sklearn; print(f'  ✅ sklearn {sklearn.__version__}')" 2>/dev/null || echo "  ❌ sklearn failed"
python -c "import h5py; print(f'  ✅ h5py {h5py.__version__}')" 2>/dev/null || echo "  ❌ h5py failed"

# 确保必要目录存在
mkdir -p results/logs model graph

echo "📁 检查文件状态:"
echo "  数据: $(ls data/mtab.h5 2>/dev/null && echo '✅' || echo '❌')"
echo "  图: $(ls graph/mtab_processed_graph.txt 2>/dev/null && echo '✅' || echo '❌')"
echo "  模型: $(ls model/mtab.pkl 2>/dev/null && echo '✅' || echo '❌')"

# 如果图文件缺失，先构建图
if [ ! -f "graph/mtab_processed_graph.txt" ]; then
    echo "⚠️ 构建图文件..."
    python calcu_graph_mtab.py
fi

# 如果预训练模型缺失，先预训练 (快速版本)
if [ ! -f "model/mtab.pkl" ]; then
    echo "⚠️ 运行预训练..."
    python pretrain_mtab.py
fi

echo "🚀 开始种子搜索..."
python seed_experiment_enhanced.py

echo "✅ 种子猎人计划完成 - $(date)"
