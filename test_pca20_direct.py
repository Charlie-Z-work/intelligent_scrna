#!/usr/bin/env python3
"""直接使用PCA-20配置测试"""

import sys
sys.path.append('.')

from core.geometry_analyzer import GeometryAnalyzer
from core.strategy_atlas import StrategyAtlas
from algorithms.boundary_failure import BoundaryFailureLearning
from utils.metrics import MetricsCalculator
import pandas as pd
import numpy as np
import time

def test_pca20_direct():
    """直接测试PCA-20配置"""
    
    print("🎯 直接测试PCA-20配置")
    
    # 加载数据
    try:
        X = pd.read_csv('data/in_X.csv', header=None).values
        y_true = pd.read_csv('data/true_labs.csv', header=None).values.squeeze()
        print(f"数据: {X.shape}, 类别: {len(np.unique(y_true))}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    start_time = time.time()
    
    # 1. 几何分析
    print("\n🔍 几何分析...")
    analyzer = GeometryAnalyzer()
    features = analyzer.analyze(X, verbose=False)
    
    # 2. 策略匹配
    print("\n🗺️ 策略匹配...")
    atlas = StrategyAtlas("data/atlas_knowledge.json")
    strategy = atlas.find_best_match(features)
    
    print(f"   策略: {strategy['name']}")
    print(f"   相似度: {strategy['similarity']:.3f}")
    
    # 3. 强制使用PCA-20配置
    pca20_strategy = {
        'name': 'boundary_failure_learning_pca20',
        'algorithm': 'gmm',
        'covariance_type': 'full',
        'pca_components': 20,
        'n_clusters': 4,
        'random_state': 42,
        'n_init': 10
    }
    
    print(f"\n🚀 执行PCA-20策略...")
    print(f"   配置: {pca20_strategy}")
    
    # 4. 执行算法
    algorithm = BoundaryFailureLearning()
    labels = algorithm.fit_predict(X, pca20_strategy)
    
    # 5. 计算指标
    metrics = MetricsCalculator()
    performance = metrics.calculate_all_metrics(y_true, labels, X)
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 结果:")
    print(f"   NMI: {performance['nmi']:.4f}")
    print(f"   ARI: {performance['ari']:.4f}")
    print(f"   Silhouette: {performance.get('silhouette', 0):.4f}")
    print(f"   耗时: {elapsed:.1f}s")
    
    print(f"\n🎯 对比:")
    print(f"   目标NMI: 0.9097")
    print(f"   当前NMI: {performance['nmi']:.4f}")
    print(f"   差距: {performance['nmi'] - 0.9097:+.4f}")
    
    if performance['nmi'] > 0.85:
        print("🎉 优秀！接近benchmark")
    elif performance['nmi'] > 0.7:
        print("✅ 良好表现")
    else:
        print("📈 有改进空间")

if __name__ == "__main__":
    test_pca20_direct()
