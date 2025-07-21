#!/bin/bash
#SBATCH --job-name=exp_50_clean
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/exp_50_clean_%j.out
#SBATCH --error=results/logs/exp_50_clean_%j.err

echo "🎯 50次清洁版随机状态实验 - $(date)"
echo "🖥️  节点: $(hostname)"

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scdsc_env

echo "🧪 开始50次实验 (无warnings)..."
python random_state_experiment_fixed.py --num_experiments 50

echo "✅ 实验完成 - $(date)"
