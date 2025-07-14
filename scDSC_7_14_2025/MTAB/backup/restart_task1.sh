#!/bin/bash
#SBATCH --job-name=restart_t1_va001
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --nodelist=va001
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/restart_task1_%j.out
#SBATCH --error=results/logs/restart_task1_%j.err

echo "🔄 重启任务1 va001 (配置1833-3750) - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"
echo "📈 进度: 从断点1832继续，剩余1918个配置"

module load python/3.11.11/pytorch/2025.04.01
cd /users/tzheng/scDSC/scDSC_7_13_2025/MTAB

export TASK_ID=1
python base_grid_search_restart.py

echo "✅ 重启任务1完成 - $(date)"
