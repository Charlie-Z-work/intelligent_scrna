#!/bin/bash
#SBATCH --job-name=exp_1000_hunt
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=results/logs/exp_1000_hunt_%j.out
#SBATCH --error=results/logs/exp_1000_hunt_%j.err

echo "🎯 1000次大规模种子猎捕 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🚀 目标: 寻找ARI=0.77+的传说种子"

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scdsc_env

echo "🧪 开始1000次超大规模实验..."
python random_state_experiment_fixed.py --num_experiments 1000

echo "🏆 1000次种子猎捕完成 - $(date)"
