#!/bin/bash
#SBATCH --job-name=scdsc_grid
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/grid_search_%j.out
#SBATCH --error=results/logs/grid_search_%j.err

echo "🔍 scDSC网格搜索开始 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"
echo "📁 工作目录: $(pwd)"

module load python/3.11.11/pytorch/2025.04.01

echo "🚀 启动网格搜索..."
python grid_search.py

echo ""
echo "✅ 网格搜索完成 - $(date)"
echo "📁 查看结果文件:"
ls -la results/logs/grid_search_*.csv
ls -la results/logs/best_config_*.txt

echo ""
echo "📊 快速预览最佳结果:"
tail -1 results/logs/grid_search_*.csv | cut -d',' -f10-12
