#!/bin/bash
#SBATCH --job-name=seed_hunt_v2
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/seed_hunt_v2_%j.out
#SBATCH --error=results/logs/seed_hunt_v2_%j.err

echo "🎯 种子猎人计划V2启动 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

# 加载Python模块
echo "🔍 加载Python环境..."
module load python/3.11.7-55l7n7g

# 检查Python环境
echo "🐍 Python环境检查:"
python --version
which python

echo "📦 检查必要包:"
python -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>/dev/null || echo "  ❌ numpy missing"
python -c "import torch; print(f'  ✅ torch {torch.__version__}')" 2>/dev/null || echo "  ❌ torch missing"
python -c "import sklearn; print(f'  ✅ sklearn {sklearn.__version__}')" 2>/dev/null || echo "  ❌ sklearn missing"
python -c "import h5py; print(f'  ✅ h5py {h5py.__version__}')" 2>/dev/null || echo "  ❌ h5py missing"

# 如果缺少包，尝试pip安装到用户目录
if ! python -c "import numpy" 2>/dev/null; then
    echo "📦 安装numpy..."
    pip install --user numpy
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "📦 安装torch..."
    pip install --user torch torchvision
fi

if ! python -c "import sklearn" 2>/dev/null; then
    echo "📦 安装sklearn..."
    pip install --user scikit-learn
fi

if ! python -c "import h5py" 2>/dev/null; then
    echo "📦 安装h5py..."
    pip install --user h5py
fi

if ! python -c "import munkres" 2>/dev/null; then
    echo "📦 安装munkres..."
    pip install --user munkres
fi

# 确保必要目录和文件存在
mkdir -p results/logs model graph data

echo "📁 检查必要文件:"
ls -la data/ | head -5
ls -la model/ | head -5
ls -la graph/ | head -5

# 如果数据文件缺失，先运行预处理
if [ ! -f "data/mtab.h5" ]; then
    echo "⚠️ 数据文件缺失，先运行预处理..."
    if [ -f "preprocess.py" ]; then
        python preprocess.py
    else
        echo "❌ 预处理脚本不存在，无法继续"
        exit 1
    fi
fi

# 如果图文件缺失，先构建图
if [ ! -f "graph/mtab_processed_graph.txt" ]; then
    echo "⚠️ 图文件缺失，先构建图..."
    if [ -f "calcu_graph_mtab.py" ]; then
        python calcu_graph_mtab.py
    else
        echo "❌ 图构建脚本不存在，无法继续"
        exit 1
    fi
fi

# 如果预训练模型缺失，先预训练
if [ ! -f "model/mtab.pkl" ]; then
    echo "⚠️ 预训练模型缺失，先预训练..."
    if [ -f "pretrain_mtab.py" ]; then
        python pretrain_mtab.py
    else
        echo "❌ 预训练脚本不存在，无法继续"
        exit 1
    fi
fi

echo "🚀 开始智能种子搜索..."
python seed_experiment_enhanced.py

echo "✅ 种子猎人计划完成 - $(date)"
