#!/bin/bash
#SBATCH --job-name=analyze_best
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=results/logs/analysis_%j.out
#SBATCH --error=results/logs/analysis_%j.err

echo "🔍 分析最优配置训练过程 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

module load python/3.11.11/pytorch/2025.04.01
cd /users/tzheng/scDSC/scDSC_7_13_2025/MTAB

python analyze_best_config.py

echo "✅ 分析完成 - $(date)"
