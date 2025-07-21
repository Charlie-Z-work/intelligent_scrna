#!/bin/bash
#SBATCH --job-name=random_state_test
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/random_state_test_%j.out
#SBATCH --error=results/logs/random_state_test_%j.err

echo "🎯 随机状态发现实验 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

# 加载环境
module load conda
conda activate scdsc

echo "🧪 运行5次随机状态实验..."
python random_state_experiment.py --num_experiments 5

echo "✅ 随机状态实验完成 - $(date)"
