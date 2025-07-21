#!/bin/bash
#SBATCH --job-name=seed_hunt_conda
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/seed_hunt_conda_%j.out
#SBATCH --error=results/logs/seed_hunt_conda_%j.err

echo "🎯 种子猎人Conda版启动 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

# 加载conda环境
echo "🐍 加载Conda环境..."
module load conda

# 初始化conda (如果需要)
source $(conda info --base)/etc/profile.d/conda.sh

# 创建或激活环境
ENV_NAME="scdsc_env"
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "✅ 激活现有环境: $ENV_NAME"
    conda activate $ENV_NAME
else
    echo "🔧 创建新环境: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME
fi

# 安装必要包
echo "📦 安装科学计算包..."
conda install -y numpy pandas scikit-learn h5py scipy
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install munkres

echo "🔍 验证环境:"
python --version
python -c "import numpy; print(f'✅ numpy {numpy.__version__}')"
python -c "import torch; print(f'✅ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import sklearn; print(f'✅ sklearn {sklearn.__version__}')"
python -c "import h5py; print(f'✅ h5py {h5py.__version__}')"

echo "📁 文件检查:"
echo "  ✅ 数据: $(ls -lh data/mtab.h5 | awk '{print $5}')"
echo "  ✅ 图: $(ls -lh graph/mtab_processed_graph.txt | awk '{print $5}')"  
echo "  ✅ 模型: $(ls -lh model/mtab.pkl | awk '{print $5}')"

echo "🚀 开始种子搜索 - 目标ARI=0.7764..."
python seed_experiment_enhanced.py

echo "✅ 种子猎人完成 - $(date)"
