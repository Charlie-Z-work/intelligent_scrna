#!/bin/bash
#SBATCH --job-name=exp_5000_opt
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=results/logs/exp_5000_opt_%j.out
#SBATCH --error=results/logs/exp_5000_opt_%j.err

echo "🎯 5000次优化版实验 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "💾 内存: 16G"
echo "🎲 固定种子池 + Embedding分离"

echo "📊 内存快照："
free -h

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scdsc_env

echo "🚀 开始优化版实验..."
python random_state_experiment_optimized.py --num_experiments 5000 --batch_size 100

echo "🏆 优化版实验完成 - $(date)"
