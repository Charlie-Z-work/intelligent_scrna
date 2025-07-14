#!/bin/bash

echo "🚀 启动3GPU大规模并行网格搜索"
echo "📊 总配置数: 11,250个"
echo "⏱️  预计时间: 每个GPU 3.5小时"
echo "🎯 目标: 寻找ARI>0.75, NMI>0.68的配置"
echo ""

# 提交3个任务
echo "📤 提交任务1 (va001): 配置1-3750..."
TASK1_ID=$(sbatch submit_massive_task1.sh | awk '{print $4}')
echo "   作业ID: $TASK1_ID"

echo "📤 提交任务2 (va002): 配置3751-7500..."
TASK2_ID=$(sbatch submit_massive_task2.sh | awk '{print $4}')
echo "   作业ID: $TASK2_ID"

echo "📤 提交任务3 (va003): 配置7501-11250..."
TASK3_ID=$(sbatch submit_massive_task3.sh | awk '{print $4}')
echo "   作业ID: $TASK3_ID"

echo ""
echo "✅ 3个任务已提交！"
echo ""
echo "📊 监控命令:"
echo "   squeue -u \$USER"
echo "   tail -f results/logs/massive_task1_*.out"
echo "   tail -f results/logs/massive_task2_*.out" 
echo "   tail -f results/logs/massive_task3_*.out"
echo ""
echo "📁 结果文件:"
echo "   results/logs/massive_task1_JOBID_TIME.csv"
echo "   results/logs/massive_task2_JOBID_TIME.csv"
echo "   results/logs/massive_task3_JOBID_TIME.csv"
echo ""
echo "🎯 预期在3.5小时内完成所有搜索！"
