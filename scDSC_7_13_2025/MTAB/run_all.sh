#!/bin/bash
#SBATCH --job-name=scdsc_all
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=results/logs/all_%j.out
#SBATCH --error=results/logs/all_%j.err

echo "🚀 scDSC批量优化开始 - $(date)"
echo "🖥️  节点: $(hostname)"
echo "🆔 作业ID: $SLURM_JOB_ID"

module load python/3.11.11/pytorch/2025.04.01

configs=("baseline" "zinb_enhanced" "lr_tuned" "sigma_tuned" "aggressive")

for config in "${configs[@]}"; do
    echo ""
    echo "🧪 === 开始配置: $config ==="
    export CONFIG_NAME=$config
    python hyperopt_zinb.py
    echo "✅ === 配置 $config 完成 ==="
done

echo ""
echo "🎯 所有配置完成！"
echo "📁 结果文件:"
ls -la results/logs/result_*

echo ""
echo "📊 快速汇总:"
for config in "${configs[@]}"; do
    result_file=$(ls results/logs/result_${config}_${SLURM_JOB_ID}_*.txt 2>/dev/null | head -1)
    if [ -f "$result_file" ]; then
        echo "=== $config ==="
        grep -E "(最佳ARI|最佳NMI)" "$result_file"
    fi
done
