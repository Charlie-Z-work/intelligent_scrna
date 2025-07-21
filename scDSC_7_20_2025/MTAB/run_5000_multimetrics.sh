#!/bin/bash
#SBATCH --job-name=exp_5000_multi
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=results/logs/exp_5000_multi_%j.out
#SBATCH --error=results/logs/exp_5000_multi_%j.err

echo "🎯 5000次多指标种子发现实验 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "📊 集成8项聚类评估指标"

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scdsc_env

echo "🚀 开始增强版5000次实验..."
python random_state_experiment_enhanced.py --num_experiments 5000

echo "🏆 多指标实验完成 - $(date)"
