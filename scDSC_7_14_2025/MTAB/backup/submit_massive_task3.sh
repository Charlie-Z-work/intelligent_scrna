#!/bin/bash
#SBATCH --job-name=massive_t3_va003
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --nodelist=va003
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=3:30:00
#SBATCH --output=results/logs/massive_task3_%j.out
#SBATCH --error=results/logs/massive_task3_%j.err

echo "🚀 大规模任务3 va003 (配置7501-11250) - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

module load python/3.11.11/pytorch/2025.04.01

export TASK_ID=3
python base_grid_search.py

echo "✅ 任务3完成 - $(date)"
