#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终修复脚本
解决策略配置传递和失败分析器bug
"""

from pathlib import Path

def fix_strategy_config_passing():
    """修复策略配置传递问题"""
    
    print("🔧 修复策略配置传递...")
    
    main_file = "main.py"
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找initial_strategy赋值的地方
    if "initial_strategy = self.strategy_atlas.find_best_match" in content:
        # 在这之后添加配置合并逻辑
        
        old_pattern = "initial_strategy = self.strategy_atlas.find_best_match(geometry_features)"
        
        new_pattern = '''initial_strategy = self.strategy_atlas.find_best_match(geometry_features)
        
        # 确保策略详细配置被正确传递
        if 'strategy_details' in initial_strategy:
            strategy_details = initial_strategy['strategy_details']
            # 将详细配置合并到初始策略中
            for key, value in strategy_details.items():
                if key not in initial_strategy:
                    initial_strategy[key] = value
            print(f"   📋 应用策略配置: PCA-{initial_strategy.get('pca_components', 'unknown')}维")'''
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print("   ✅ 修复策略配置传递")
        
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(content)

def fix_failure_analyzer_bugs():
    """修复失败分析器的变量错误"""
    
    print("🔧 修复失败分析器bugs...")
    
    failure_file = "core/failure_analyzer.py"
    
    with open(failure_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复current_dim未定义的问题
    # 在各个分析方法开始处添加current_dim定义
    
    methods_to_fix = [
        "_analyze_boundary_confusion",
        "_analyze_parameter_issues", 
        "_analyze_algorithm_mismatch",
        "_analyze_cluster_number_error",
        "_analyze_convergence_issues"
    ]
    
    for method_name in methods_to_fix:
        if f"def {method_name}" in content:
            # 在方法开始后添加current_dim定义
            method_start = f"def {method_name}("
            method_pos = content.find(method_start)
            
            if method_pos != -1:
                # 找到方法体开始
                line_end = content.find("\\n", method_pos)
                next_line_end = content.find("\\n", line_end + 1)  # 跳过方法定义行
                next_line_end = content.find("\\n", next_line_end + 1)  # 跳过docstring开始
                
                # 在方法体内插入current_dim定义
                before = content[:next_line_end + 1]
                after = content[next_line_end + 1:]
                
                dim_definition = '''        
        # 获取当前策略的PCA维度
        current_dim = failure_info.get('strategy', {}).get('pca_components', 
                     failure_info.get('strategy', {}).get('dimension', 50))
'''
                
                content = before + dim_definition + after
                print(f"   ✅ 修复 {method_name} 的current_dim定义")
    
    with open(failure_file, 'w', encoding='utf-8') as f:
        f.write(content)

def create_direct_pca20_test():
    """创建直接使用PCA-20的测试"""
    
    test_code = '''#!/usr/bin/env python3
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
    print("\\n🔍 几何分析...")
    analyzer = GeometryAnalyzer()
    features = analyzer.analyze(X, verbose=False)
    
    # 2. 策略匹配
    print("\\n🗺️ 策略匹配...")
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
    
    print(f"\\n🚀 执行PCA-20策略...")
    print(f"   配置: {pca20_strategy}")
    
    # 4. 执行算法
    algorithm = BoundaryFailureLearning()
    labels = algorithm.fit_predict(X, pca20_strategy)
    
    # 5. 计算指标
    metrics = MetricsCalculator()
    performance = metrics.calculate_all_metrics(y_true, labels, X)
    
    elapsed = time.time() - start_time
    
    print(f"\\n📊 结果:")
    print(f"   NMI: {performance['nmi']:.4f}")
    print(f"   ARI: {performance['ari']:.4f}")
    print(f"   Silhouette: {performance.get('silhouette', 0):.4f}")
    print(f"   耗时: {elapsed:.1f}s")
    
    print(f"\\n🎯 对比:")
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
'''
    
    with open("test_pca20_direct.py", 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("✅ 创建PCA-20直接测试: test_pca20_direct.py")

def main():
    """最终修复"""
    print("🎯 最终修复 - 解决剩余问题")
    print("=" * 50)
    
    # 1. 修复策略配置传递
    fix_strategy_config_passing()
    
    # 2. 修复失败分析器bugs
    fix_failure_analyzer_bugs()
    
    # 3. 创建直接测试
    create_direct_pca20_test()
    
    print("=" * 50)
    print("🎉 最终修复完成!")
    print()
    print("🧪 验证方案:")
    print("1. 直接验证PCA-20: python test_pca20_direct.py")
    print("2. 完整系统测试: python test_system.py")
    print()
    print("💡 预期改进:")
    print("- PCA-20配置将被正确应用")
    print("- 失败分析器不再报错")
    print("- 性能应该显著提升")

if __name__ == "__main__":
    main()