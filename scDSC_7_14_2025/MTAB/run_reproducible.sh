#!/bin/bash
#SBATCH --job-name=reproducible
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/reproducible_%j.out
#SBATCH --error=results/logs/reproducible_%j.err

echo "🔍 寻找可复现的最佳配置 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

module load python/3.11.11/pytorch/2025.04.01
cd /users/tzheng/scDSC/scDSC_7_13_2025/MTAB

python reproducible_best_config.py --find_best

echo "✅ 可复现性分析完成 - $(date)"
