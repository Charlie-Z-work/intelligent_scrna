#!/bin/bash

echo "🔄 重启大规模并行网格搜索 - 从断点继续"
echo "📊 剩余配置: 5,743个 (任务1: 1918, 任务2: 1912, 任务3: 1913)"
echo "⏱️  每任务预计时间: 约3.2小时"
echo "🎯 目标: 完成剩余搜索，寻找ARI>0.75的配置"
echo ""

# 提交重启的3个任务
echo "📤 重启任务1 (va001): 配置1833-3750..."
TASK1_ID=$(sbatch restart_task1.sh | awk '{print $4}')
echo "   作业ID: $TASK1_ID"

echo "📤 重启任务2 (va002): 配置5589-7500..."
TASK2_ID=$(sbatch restart_task2.sh | awk '{print $4}')
echo "   作业ID: $TASK2_ID"

echo "📤 重启任务3 (va003): 配置9338-11250..."
TASK3_ID=$(sbatch restart_task3.sh | awk '{print $4}')
echo "   作业ID: $TASK3_ID"

echo ""
echo "✅ 3个重启任务已提交！"
echo ""
echo "📊 监控命令:"
echo "   squeue -u \$USER"
echo "   tail -f results/logs/restart_task1_*.out"
echo "   tail -f results/logs/restart_task2_*.out" 
echo "   tail -f results/logs/restart_task3_*.out"
echo ""
echo "📁 新结果文件:"
echo "   results/logs/restart_task1_JOBID_TIME.csv"
echo "   results/logs/restart_task2_JOBID_TIME.csv"
echo "   results/logs/restart_task3_JOBID_TIME.csv"
echo ""
echo "🎯 预期在3.5小时内完成剩余5,743个配置的搜索！"
