#!/bin/bash
echo "📊 1000次实验进度监控"
while true; do
    if [ -f "results/logs/exp_1000_hunt_*.out" ]; then
        latest_out=$(ls -t results/logs/exp_1000_hunt_*.out | head -1)
        echo "$(date): 当前进度"
        tail -3 "$latest_out" | grep -E "(进度|🏆|最佳)"
        echo "---"
    fi
    sleep 30
done
