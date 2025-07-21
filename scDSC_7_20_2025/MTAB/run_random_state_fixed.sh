#!/bin/bash
#SBATCH --job-name=random_state_fixed
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/random_state_fixed_%j.out
#SBATCH --error=results/logs/random_state_fixed_%j.err

echo "🎯 随机状态实验修复版 - $(date)"
echo "🖥️  节点: $(hostname)"

# 加载conda并检查环境
module load conda
echo "🐍 Conda版本: $(conda --version)"
echo "📁 Conda环境列表:"
conda env list

# 检查scdsc环境是否存在
if conda env list | grep -q "^scdsc"; then
    echo "✅ 找到scdsc环境，激活中..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate scdsc
else
    echo "❌ scdsc环境不存在，创建中..."
    conda create -n scdsc python=3.10 numpy torch scikit-learn h5py -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate scdsc
    pip install munkres
fi

echo "🔍 验证Python环境:"
python --version
python -c "import numpy, torch, sklearn, h5py; print('✅ 所有包OK')"

echo "🧪 运行3次随机状态实验..."
python random_state_experiment.py --num_experiments 3

echo "✅ 实验完成 - $(date)"
