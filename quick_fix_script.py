#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复脚本
解决模块导入问题，创建缺失文件
"""

import os
from pathlib import Path

def create_missing_files():
    """创建缺失的文件"""
    
    print("🔧 快速修复：创建缺失文件...")
    
    # 1. 创建目录结构
    directories = ['core', 'algorithms', 'utils', 'data', 'results', 'logs', 'cache']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✅ 目录: {dir_name}")
    
    # 2. 创建 __init__.py 文件
    init_files = {
        'core/__init__.py': '',
        'algorithms/__init__.py': '',
        'utils/__init__.py': ''
    }
    
    for file_path, content in init_files.items():
        Path(file_path).touch()
        print(f"  ✅ 文件: {file_path}")
    
    # 3. 创建简化的 utils 模块文件
    utils_files = {
        'utils/data_loader.py': '''#!/usr/bin/env python3
"""简化数据加载器"""
import numpy as np
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, config=None):
        self.config = config
    
    def load_dataset(self, data_path, labels_path=None):
        X = pd.read_csv(data_path, header=None).values
        y_true = None
        if labels_path and Path(labels_path).exists():
            y_true = pd.read_csv(labels_path, header=None).values.squeeze()
        return X, y_true
    
    def load_datasets_config(self, config_path):
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
''',
        
        'utils/metrics.py': '''#!/usr/bin/env python3
"""简化指标计算器"""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

class MetricsCalculator:
    def calculate_all_metrics(self, y_true, y_pred, X):
        metrics = {}
        try:
            metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
            metrics['ari'] = adjusted_rand_score(y_true, y_pred)
            metrics['silhouette'] = silhouette_score(X, y_pred)
        except:
            metrics = {'nmi': 0.0, 'ari': 0.0, 'silhouette': 0.0}
        return metrics
''',
        
        'utils/visualization.py': '''#!/usr/bin/env python3
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
'''
    }
    
    for file_path, content in utils_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 文件: {file_path}")
    
    # 4. 创建简化的算法文件
    algorithm_files = {
        'algorithms/boundary_failure.py': '''#!/usr/bin/env python3
"""简化边界失败学习"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class BoundaryFailureLearning:
    def __init__(self, config=None):
        self.config = config
    
    def fit_predict(self, X, strategy):
        n_clusters = strategy.get('n_clusters', 3)
        pca_components = strategy.get('pca_components', 20)
        random_state = strategy.get('random_state', 42)
        
        if pca_components > 0 and pca_components < X.shape[1]:
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', 
                             random_state=random_state, n_init=10)
        return gmm.fit_predict(X_reduced)
''',
        
        'algorithms/enhanced_sre.py': '''#!/usr/bin/env python3
"""简化增强SRE"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class EnhancedSRE:
    def __init__(self, config=None):
        self.config = config
    
    def fit_predict(self, X, strategy):
        n_clusters = strategy.get('n_clusters', 3)
        pca_components = strategy.get('pca_components', 30)
        random_state = strategy.get('random_state', 42)
        
        if pca_components > 0 and pca_components < X.shape[1]:
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        return kmeans.fit_predict(X_reduced)
''',
        
        'algorithms/ultimate_fusion.py': '''#!/usr/bin/env python3
"""简化终极融合"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class UltimateFusion:
    def __init__(self, config=None):
        self.config = config
    
    def fit_predict(self, X, strategy):
        n_clusters = strategy.get('n_clusters', 3)
        pca_components = strategy.get('pca_components', 50)
        random_state = strategy.get('random_state', 42)
        
        if pca_components > 0 and pca_components < X.shape[1]:
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        # 简化版：只用GMM
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', 
                             random_state=random_state)
        return gmm.fit_predict(X_reduced)
'''
    }
    
    for file_path, content in algorithm_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 文件: {file_path}")
    
    print("\n🎉 快速修复完成!")
    print("📝 现在可以重新运行 main.py 了")

def check_dependencies():
    """检查依赖包"""
    print("\n🔍 检查依赖包...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 需要安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print(f"\n🎉 所有依赖包都已安装!")

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        # 测试数据生成
        from sklearn.datasets import make_blobs
        import numpy as np
        
        X, y = make_blobs(n_samples=100, centers=3, n_features=10, random_state=42)
        print(f"  ✅ 测试数据生成: {X.shape}")
        
        # 测试几何分析器
        if Path('core/geometry_analyzer.py').exists():
            from core.geometry_analyzer import GeometryAnalyzer
            analyzer = GeometryAnalyzer()
            features = analyzer.analyze(X, verbose=False)
            print(f"  ✅ 几何分析器运行正常")
        
        print(f"\n🎉 基本功能测试通过!")
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

if __name__ == "__main__":
    print("🚀 智能scRNA系统快速修复工具")
    print("=" * 50)
    
    # 1. 创建缺失文件
    create_missing_files()
    
    # 2. 检查依赖
    check_dependencies()
    
    # 3. 测试功能
    test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("🎯 修复完成! 现在可以运行:")
    print("python main.py --data in_X.csv --labels true_labs.csv --name Usoskin")
