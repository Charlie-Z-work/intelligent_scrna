#!/bin/bash
#SBATCH --job-name=random_state_final
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/random_state_final_%j.out
#SBATCH --error=results/logs/random_state_final_%j.err

echo "🎯 随机状态实验最终版 - $(date)"
echo "🖥️  节点: $(hostname)"

# 加载conda
module load conda
source $(conda info --base)/etc/profile.d/conda.sh

# 激活正确的环境名
conda activate scdsc_env

echo "🔍 验证环境:"
python --version
python -c "import numpy, torch, sklearn, h5py; print('✅ 所有包OK')"

echo "🧪 开始3次随机状态实验..."
python random_state_experiment.py --num_experiments 3

echo "✅ 实验完成 - $(date)"
