#!/bin/bash

echo "📊 汇总所有搜索结果（原始+重启）"
echo ""

# 检查所有结果文件
echo "📁 检查结果文件:"
ls -la results/logs/massive_task*_*.csv 2>/dev/null || echo "  没有原始结果文件"
ls -la results/logs/restart_task*_*.csv 2>/dev/null || echo "  没有重启结果文件"

echo ""
echo "🏆 Top 10 全局最佳配置:"

# 合并所有结果并排序
temp_file="/tmp/all_results_complete_$$"
echo "task_id,config_id,alpha,beta,gamma,delta,lr,sigma,epochs,use_scheduler,ari,nmi,score,time" > "$temp_file"

# 合并原始结果
if ls results/logs/massive_task*_*.csv 1> /dev/null 2>&1; then
    tail -n +2 results/logs/massive_task*_*.csv >> "$temp_file"
    echo "✅ 已合并原始结果"
fi

# 合并重启结果  
if ls results/logs/restart_task*_*.csv 1> /dev/null 2>&1; then
    tail -n +2 results/logs/restart_task*_*.csv >> "$temp_file"
    echo "✅ 已合并重启结果"
fi

# 检查是否有数据
if [ $(wc -l < "$temp_file") -gt 1 ]; then
    # 按score排序显示top 10
    echo ""
    sort -t',' -k13 -nr "$temp_file" | head -10 | while IFS=',' read task_id config_id alpha beta gamma delta lr sigma epochs scheduler ari nmi score time; do
        if [ "$task_id" != "task_id" ]; then  # 跳过表头
            echo "ARI=${ari}, NMI=${nmi}, Score=${score}"
            echo "  α=${alpha}, β=${beta}, γ=${gamma}, δ=${delta}, lr=${lr}, σ=${sigma}"
            echo "  epochs=${epochs}, scheduler=${scheduler}, 任务${task_id}, 配置${config_id}"
            echo ""
        fi
    done
    
    # 统计信息
    total_configs=$(tail -n +2 "$temp_file" | wc -l)
    best_ari=$(tail -n +2 "$temp_file" | cut -d',' -f11 | sort -nr | head -1)
    best_nmi=$(tail -n +2 "$temp_file" | cut -d',' -f12 | sort -nr | head -1)
    above_075=$(tail -n +2 "$temp_file" | awk -F',' '$11 > 0.75 {count++} END {print count+0}')
    above_068=$(tail -n +2 "$temp_file" | awk -F',' '$12 > 0.68 {count++} END {print count+0}')
    
    echo "📈 全局统计:"
    echo "  完成配置数: ${total_configs}"
    echo "  最佳ARI: ${best_ari} (论文对比: $(echo "scale=1; $best_ari/0.62*100" | bc -l 2>/dev/null || echo "N/A")%)"
    echo "  最佳NMI: ${best_nmi} (论文对比: $(echo "scale=1; $best_nmi/0.68*100" | bc -l 2>/dev/null || echo "N/A")%)"
    echo "  超越ARI 0.75: ${above_075}/${total_configs}"
    echo "  超越NMI 0.68: ${above_068}/${total_configs}"
else
    echo "❌ 未找到有效的结果数据"
fi

# 清理临时文件
rm -f "$temp_file"
