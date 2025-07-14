#!/bin/bash
#SBATCH --job-name=test_250_epochs
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=results/logs/test_250_%j.out
#SBATCH --error=results/logs/test_250_%j.err

echo "🧪 测试250轮训练时间 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

module load python/3.11.11/pytorch/2025.04.01

python test_250_epochs.py

echo "✅ 测试完成 - $(date)"
