#!/bin/bash
#SBATCH --job-name=scdsc_hyperopt
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/hyperopt_%j.out
#SBATCH --error=results/logs/hyperopt_%j.err

echo "🚀 scDSC超参数优化 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"
echo "📁 工作目录: $(pwd)"

module load python/3.11.11/pytorch/2025.04.01

export CONFIG_NAME=${1:-baseline}
echo "🔧 配置: $CONFIG_NAME"

python hyperopt_zinb.py

echo "✅ 训练完成 - $(date)"
