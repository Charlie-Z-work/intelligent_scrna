#!/bin/bash
#SBATCH --job-name=seed_hunt_final
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/seed_hunt_final_%j.out
#SBATCH --error=results/logs/seed_hunt_final_%j.err

echo "🎯 种子猎人最终版启动 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

# 正确加载Python模块
echo "🔧 加载编译器和Python..."
module load gcc/11.2.0
module load python/3.11.7-55l7n7g

echo "🐍 Python环境:"
python --version
which python

# 检查或安装包
echo "📦 检查Python包..."
python -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>/dev/null || {
    echo "  📦 安装numpy..."
    python -m pip install --user numpy
}

python -c "import torch; print(f'  ✅ torch {torch.__version__}')" 2>/dev/null || {
    echo "  📦 安装torch..."
    python -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

python -c "import sklearn; print(f'  ✅ sklearn {sklearn.__version__}')" 2>/dev/null || {
    echo "  📦 安装sklearn..."
    python -m pip install --user scikit-learn
}

python -c "import h5py; print(f'  ✅ h5py {h5py.__version__}')" 2>/dev/null || {
    echo "  📦 安装h5py..."
    python -m pip install --user h5py
}

python -c "import munkres; print(f'  ✅ munkres OK')" 2>/dev/null || {
    echo "  📦 安装munkres..."
    python -m pip install --user munkres
}

echo "📁 文件检查:"
echo "  ✅ 数据文件: $(ls -lh data/mtab.h5 | awk '{print $5}')"
echo "  ✅ 图文件: $(ls -lh graph/mtab_processed_graph.txt | awk '{print $5}')"
echo "  ✅ 模型文件: $(ls -lh model/mtab.pkl | awk '{print $5}')"

echo "🚀 开始种子搜索 - 目标ARI=0.7764..."
python seed_experiment_enhanced.py

echo "✅ 种子猎人完成 - $(date)"
