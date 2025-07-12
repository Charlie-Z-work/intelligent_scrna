# utils/__init__.py
"""工具模块包"""
pass

# utils/data_loader.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据加载工具"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

class DataLoader:
    """数据加载器"""
    
    def __init__(self, config=None):
        self.config = config
    
    def load_dataset(self, data_path: str, labels_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """加载单个数据集"""
        
        # 加载数据
        if data_path.endswith('.csv'):
            X = pd.read_csv(data_path, header=None).values
        elif data_path.endswith('.npy'):
            X = np.load(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        # 加载标签
        y_true = None
        if labels_path and Path(labels_path).exists():
            if labels_path.endswith('.csv'):
                y_true = pd.read_csv(labels_path, header=None).values.squeeze()
            elif labels_path.endswith('.npy'):
                y_true = np.load(labels_path)
        
        return X, y_true
    
    def load_datasets_config(self, config_path: str) -> Dict[str, Dict[str, str]]:
        """加载数据集配置"""
        with open(config_path, 'r') as f:
            return json.load(f)

# utils/metrics.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估指标计算"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import calinski_harabasz_score
from typing import Dict, Any

class MetricsCalculator:
    """指标计算器"""
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        """计算所有评估指标"""
        
        metrics = {}
        
        try:
            metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
        except:
            metrics['nmi'] = 0.0
        
        try:
            metrics['ari'] = adjusted_rand_score(y_true, y_pred)
        except:
            metrics['ari'] = 0.0
        
        try:
            metrics['silhouette'] = silhouette_score(X, y_pred)
        except:
            metrics['silhouette'] = 0.0
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, y_pred)
        except:
            metrics['calinski_harabasz'] = 0.0
        
        return metrics

# utils/visualization.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化工具"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, config=None):
        self.config = config
    
    def plot_learning_trajectory(self, result: Dict[str, Any]):
        """绘制学习轨迹"""
        
        trajectory = result.get('learning_trajectory', [])
        if not trajectory:
            print("无学习轨迹数据")
            return
        
        # 提取数据
        iterations = [step['iteration'] for step in trajectory]
        nmi_scores = [step['nmi'] for step in trajectory]
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, nmi_scores, 'o-', linewidth=2, markersize=8)
        plt.xlabel('迭代次数')
        plt.ylabel('NMI 得分')
        plt.title('学习轨迹：NMI性能变化')
        plt.grid(True, alpha=0.3)
        
        # 标注改进点
        for i in range(1, len(nmi_scores)):
            improvement = nmi_scores[i] - nmi_scores[i-1]
            if improvement > 0.02:
                plt.annotate(f'+{improvement:.3f}', 
                           xy=(iterations[i], nmi_scores[i]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_geometry_features(self, features: Dict[str, Any]):
        """绘制几何特征"""
        print("几何特征可视化功能待实现")
        pass