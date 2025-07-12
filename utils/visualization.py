#!/usr/bin/env python3
"""简化可视化工具"""
import matplotlib.pyplot as plt

class ResultVisualizer:
    def __init__(self, config=None):
        self.config = config
    
    def plot_learning_trajectory(self, result):
        trajectory = result.get('learning_trajectory', [])
        if not trajectory:
            print("无学习轨迹数据")
            return
        
        iterations = [step['iteration'] for step in trajectory]
        nmi_scores = [step['nmi'] for step in trajectory]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, nmi_scores, 'o-', linewidth=2, markersize=8)
        plt.xlabel('迭代次数')
        plt.ylabel('NMI 得分')
        plt.title('学习轨迹：NMI性能变化')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
