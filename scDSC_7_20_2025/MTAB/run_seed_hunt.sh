#!/bin/bash
#SBATCH --job-name=seed_hunt
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/seed_hunt_%j.out
#SBATCH --error=results/logs/seed_hunt_%j.err

echo "🎯 种子猎人计划启动 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"
echo "🎰 目标: 复现ARI=0.7764, NMI=0.7025"

module load python/3.11.11/pytorch/2025.04.01

# 确保必要目录存在
mkdir -p results/logs
mkdir -p model
mkdir -p graph

echo "🚀 开始智能种子搜索..."
python seed_experiment_enhanced.py

echo "✅ 种子猎人计划完成 - $(date)"
