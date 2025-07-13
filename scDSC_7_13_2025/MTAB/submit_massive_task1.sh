#!/bin/bash
#SBATCH --job-name=massive_t1_va001
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --nodelist=va001
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=3:30:00
#SBATCH --output=results/logs/massive_task1_%j.out
#SBATCH --error=results/logs/massive_task1_%j.err

echo "🚀 大规模任务1 va001 (配置1-3750) - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

module load python/3.11.11/pytorch/2025.04.01

export TASK_ID=1
python base_grid_search.py

echo "✅ 任务1完成 - $(date)"
