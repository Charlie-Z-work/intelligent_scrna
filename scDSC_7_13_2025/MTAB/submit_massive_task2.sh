#!/bin/bash
#SBATCH --job-name=massive_t2_va002
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --nodelist=va002
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=3:30:00
#SBATCH --output=results/logs/massive_task2_%j.out
#SBATCH --error=results/logs/massive_task2_%j.err

echo "🚀 大规模任务2 va002 (配置3751-7500) - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

module load python/3.11.11/pytorch/2025.04.01

export TASK_ID=2
python base_grid_search.py

echo "✅ 任务2完成 - $(date)"
