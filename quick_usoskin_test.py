#!/usr/bin/env python3
"""快速Usoskin测试 - 绕过复杂系统直接验证"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import time

def quick_usoskin_test():
    """快速测试Usoskin最佳配置"""
    
    print("🚀 快速Usoskin测试 (PCA-20 + GMM)")
    
    start_time = time.time()
    
    # 加载数据
    try:
        X = pd.read_csv('data/in_X.csv', header=None).values
        y_true = pd.read_csv('data/true_labs.csv', header=None).values.squeeze()
        print(f"数据: {X.shape}, 类别: {len(np.unique(y_true))}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA-20
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA-20: 解释方差 {pca.explained_variance_ratio_.sum():.3f}")
    
    # GMM聚类
    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',
        random_state=42,
        n_init=10,
        reg_covar=1e-6
    )
    
    y_pred = gmm.fit_predict(X_pca)
    
    # 计算结果
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 结果:")
    print(f"   NMI: {nmi:.4f}")
    print(f"   ARI: {ari:.4f}")  
    print(f"   耗时: {elapsed:.1f}s")
    
    print(f"\n🎯 与benchmark对比:")
    print(f"   目标NMI: 0.9097")
    print(f"   实际NMI: {nmi:.4f}")
    print(f"   差距: {nmi - 0.9097:+.4f}")
    
    if nmi > 0.85:
        print("🎉 性能优秀！接近benchmark")
    elif nmi > 0.7:
        print("✅ 性能良好")
    elif nmi > 0.5:
        print("📈 性能可接受")
    else:
        print("⚠️ 性能需要改进")
    
    return nmi, ari

if __name__ == "__main__":
    quick_usoskin_test()
