#!/bin/bash

echo "📊 汇总3GPU大规模搜索结果"
echo ""

# 检查结果文件
echo "📁 检查结果文件:"
ls -la results/logs/massive_task*_*.csv

echo ""
echo "🏆 Top 10 全局最佳配置:"

# 合并所有结果并排序
if ls results/logs/massive_task*_*.csv 1> /dev/null 2>&1; then
    echo "task_id,config_id,alpha,beta,gamma,delta,lr,sigma,epochs,use_scheduler,ari,nmi,score,time" > /tmp/all_results.csv
    tail -n +2 results/logs/massive_task*_*.csv >> /tmp/all_results.csv
    
    # 按score排序显示top 10
    sort -t',' -k13 -nr /tmp/all_results.csv | head -10 | while IFS=',' read task_id config_id alpha beta gamma delta lr sigma epochs scheduler ari nmi score time; do
        echo "ARI=${ari}, NMI=${nmi}, Score=${score}"
        echo "  α=${alpha}, β=${beta}, γ=${gamma}, δ=${delta}, lr=${lr}, σ=${sigma}"
        echo "  epochs=${epochs}, scheduler=${scheduler}, 任务${task_id}, 配置${config_id}"
        echo ""
    done
    
    # 统计信息
    total_configs=$(tail -n +2 /tmp/all_results.csv | wc -l)
    best_ari=$(tail -n +2 /tmp/all_results.csv | cut -d',' -f11 | sort -nr | head -1)
    best_nmi=$(tail -n +2 /tmp/all_results.csv | cut -d',' -f12 | sort -nr | head -1)
    above_062=$(tail -n +2 /tmp/all_results.csv | awk -F',' '$11 > 0.62 {count++} END {print count+0}')
    above_068=$(tail -n +2 /tmp/all_results.csv | awk -F',' '$12 > 0.68 {count++} END {print count+0}')
    
    echo "📈 全局统计:"
    echo "  完成配置数: ${total_configs}"
    echo "  最佳ARI: ${best_ari} (论文对比: $(echo "scale=1; $best_ari/0.62*100" | bc)%)"
    echo "  最佳NMI: ${best_nmi} (论文对比: $(echo "scale=1; $best_nmi/0.68*100" | bc)%)"
    echo "  超越论文ARI(0.62): ${above_062}/${total_configs}"
    echo "  超越论文NMI(0.68): ${above_068}/${total_configs}"
    
    rm /tmp/all_results.csv
else
    echo "❌ 未找到结果文件"
fi
