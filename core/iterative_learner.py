#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迭代学习器模块
实现单数据集上的失败学习和持续优化
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from copy import deepcopy

# 修改为绝对导入
from core.failure_analyzer import FailureAnalyzer
from core.strategy_atlas import StrategyAtlas
from algorithms.boundary_failure import BoundaryFailureLearning
from algorithms.enhanced_sre import EnhancedSRE
from algorithms.ultimate_fusion import UltimateFusion
from utils.metrics import MetricsCalculator

class IterativeLearner:
    """
    迭代学习器
    
    核心功能:
    1. 从失败中学习，逐步优化
    2. 图谱指导 + 失败分析的双重机制
    3. 自适应策略调整
    4. 学习轨迹记录和分析
    """
    
    def __init__(self, 
                 strategy_atlas: StrategyAtlas,
                 failure_analyzer: FailureAnalyzer,
                 config=None):
        
        self.strategy_atlas = strategy_atlas
        self.failure_analyzer = failure_analyzer
        self.config = config
        self.metrics_calc = MetricsCalculator()
        
        # 算法实例化
        self.algorithms = {
            'boundary_failure_learning': BoundaryFailureLearning(config),
            'enhanced_sre': EnhancedSRE(config),
            'ultimate_fusion_framework': UltimateFusion(config)
        }
        
        # 学习参数
        self.max_iterations = self._get_max_iterations()
        self.improvement_threshold = self._get_improvement_threshold()
        self.convergence_patience = self._get_convergence_patience()
        
        # 学习历史
        self.learning_history = []
    
    def optimize(self, 
                X: np.ndarray,
                y_true: np.ndarray,
                initial_strategy: Dict[str, Any],
                max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        主要优化函数：从初始策略开始，通过失败学习逐步优化
        """
        
        print(f"🎯 开始迭代学习优化...")
        
        max_iter = max_iterations or self.max_iterations
        learning_trajectory = []
        
        # 初始化当前策略
        current_strategy = deepcopy(initial_strategy)
        
        # 第一次执行：测试初始策略
        print(f"\n🔄 迭代 1: 测试初始策略")
        first_result = self._execute_strategy(X, y_true, current_strategy)
        
        learning_trajectory.append({
            'iteration': 1,
            'strategy_name': current_strategy['name'],
            'strategy_details': current_strategy,
            'performance': first_result['performance'],
            'nmi': first_result['performance']['nmi'],
            'ari': first_result['performance']['ari'],
            'execution_time': first_result['execution_time'],
            'source': 'atlas_match'
        })
        
        current_performance = first_result['performance']
        current_labels = first_result['labels']
        
        print(f"   初始性能: NMI={current_performance['nmi']:.4f}")
        
        # 迭代优化过程
        patience_counter = 0
        best_performance = current_performance['nmi']
        
        for iteration in range(2, max_iter + 1):
            print(f"\n🔄 迭代 {iteration}: 失败分析与策略调整")
            
            # 失败分析
            failure_analysis = self.failure_analyzer.analyze_failure(
                X, y_true, current_labels, current_strategy, current_performance
            )
            
            # 获取改进建议
            next_suggestion = failure_analysis['next_iteration_suggestion']
            
            if next_suggestion['action'] == 'maintain':
                print(f"   ✅ 性能已达标，停止优化")
                break
            
            # 应用改进策略
            improved_strategy = self._apply_improvement(current_strategy, next_suggestion)
            
            # 执行改进策略
            iteration_start = time.time()
            improved_result = self._execute_strategy(X, y_true, improved_strategy)
            iteration_time = time.time() - iteration_start
            
            # 记录轨迹
            learning_trajectory.append({
                'iteration': iteration,
                'strategy_name': improved_strategy['name'],
                'strategy_details': improved_strategy,
                'performance': improved_result['performance'],
                'nmi': improved_result['performance']['nmi'],
                'ari': improved_result['performance']['ari'],
                'execution_time': iteration_time,
                'source': 'failure_learning',
                'failure_analysis': failure_analysis['comprehensive_analysis'],
                'improvement_applied': next_suggestion
            })
            
            # 评估改进效果
            new_nmi = improved_result['performance']['nmi']
            improvement = new_nmi - current_performance['nmi']
            
            print(f"   性能变化: {current_performance['nmi']:.4f} → {new_nmi:.4f} ({improvement:+.4f})")
            
            # 决定是否接受改进
            if improvement > self.improvement_threshold:
                print(f"   ✅ 接受改进 (改进={improvement:.4f})")
                current_strategy = improved_strategy
                current_performance = improved_result['performance']
                current_labels = improved_result['labels']
                
                # 更新最佳性能
                if new_nmi > best_performance:
                    best_performance = new_nmi
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
            else:
                print(f"   ❌ 拒绝改进 (改进={improvement:.4f} < 阈值={self.improvement_threshold})")
                patience_counter += 1
            
            # 检查收敛
            if patience_counter >= self.convergence_patience:
                print(f"   🛑 收敛检测：连续{patience_counter}次无显著改进，停止优化")
                break
        
        # 计算最终结果
        final_result = self._compile_final_result(learning_trajectory, first_result['performance'])
        
        # 更新学习历史
        self.learning_history.append({
            'timestamp': time.time(),
            'data_shape': X.shape,
            'trajectory': learning_trajectory,
            'final_result': final_result
        })
        
        return final_result
    
    def _execute_strategy(self, 
                         X: np.ndarray, 
                         y_true: np.ndarray,
                         strategy: Dict[str, Any]) -> Dict[str, Any]:
        """执行给定策略"""
        
        strategy_name = strategy['name']
        print(f"   执行策略: {strategy_name}")
        
        start_time = time.time()
        
        try:
            # 获取算法实例
            if strategy_name in self.algorithms:
                algorithm = self.algorithms[strategy_name]
                labels = algorithm.fit_predict(X, strategy)
            else:
                # 使用通用算法执行器
                labels = self._execute_generic_algorithm(X, strategy)
            
            # 计算性能指标
            performance = self.metrics_calc.calculate_all_metrics(y_true, labels, X)
            
            execution_time = time.time() - start_time
            
            print(f"   结果: NMI={performance['nmi']:.4f}, 耗时={execution_time:.1f}s")
            
            return {
                'labels': labels,
                'performance': performance,
                'execution_time': execution_time,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ 策略执行失败: {e}")
            
            # 返回失败结果
            return {
                'labels': np.zeros(len(y_true)),
                'performance': {'nmi': 0.0, 'ari': 0.0, 'silhouette': 0.0},
                'execution_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _execute_generic_algorithm(self, X: np.ndarray, strategy: Dict[str, Any]) -> np.ndarray:
        """执行通用算法"""
        
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import PCA
        
        algorithm = strategy.get('algorithm', 'kmeans')
        n_clusters = strategy.get('n_clusters', 3)
        
        # 预处理：降维
        if 'pca_components' in strategy and strategy['pca_components'] > 0:
            pca_dim = min(strategy['pca_components'], X.shape[1], X.shape[0]//2)
            pca = PCA(n_components=pca_dim, random_state=strategy.get('random_state', 42))
            X_processed = pca.fit_transform(X)
        else:
            X_processed = X
        
        # 执行聚类算法
        if algorithm == 'kmeans':
            model = KMeans(
                n_clusters=n_clusters,
                random_state=strategy.get('random_state', 42),
                n_init=strategy.get('n_init', 10),
                init=strategy.get('init', 'k-means++')
            )
            
        elif algorithm == 'gmm':
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=strategy.get('covariance_type', 'full'),
                random_state=strategy.get('random_state', 42),
                n_init=strategy.get('n_init', 1),
                reg_covar=strategy.get('reg_covar', 1e-6)
            )
            
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=strategy.get('linkage', 'ward')
            )
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return model.fit_predict(X_processed)
    
    def _apply_improvement(self, 
                          current_strategy: Dict[str, Any],
                          improvement_suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """应用改进建议到当前策略"""
        
        improved_strategy = deepcopy(current_strategy)
        changes = improvement_suggestion.get('changes', {})
        
        # 应用具体变更
        for key, value in changes.items():
            improved_strategy[key] = value
        
        # 更新策略名称以反映变更
        if changes:
            change_desc = "_".join([f"{k}_{v}" for k, v in changes.items()])
            improved_strategy['name'] = f"{current_strategy['name']}_improved_{change_desc}"
        
        print(f"   应用改进: {changes}")
        
        return improved_strategy
    
    def _compile_final_result(self, 
                             learning_trajectory: List[Dict[str, Any]],
                             initial_performance: Dict[str, float]) -> Dict[str, Any]:
        """编译最终结果"""
        
        if not learning_trajectory:
            return {
                'final_performance': initial_performance,
                'total_improvement': 0,
                'iterations_used': 0,
                'trajectory': []
            }
        
        # 找到最佳迭代
        best_iteration = max(learning_trajectory, key=lambda x: x['nmi'])
        final_performance = best_iteration['performance']
        
        # 计算总体改进
        initial_nmi = initial_performance['nmi']
        final_nmi = final_performance['nmi']
        total_improvement = final_nmi - initial_nmi
        
        # 分析学习模式
        learning_pattern = self._analyze_learning_pattern(learning_trajectory)
        
        return {
            'trajectory': learning_trajectory,
            'final_performance': final_performance,
            'best_iteration': best_iteration,
            'total_improvement': total_improvement,
            'iterations_used': len(learning_trajectory),
            'learning_pattern': learning_pattern,
            'convergence_achieved': learning_pattern['converged'],
            'improvement_efficiency': total_improvement / len(learning_trajectory) if learning_trajectory else 0
        }
    
    def _analyze_learning_pattern(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析学习模式"""
        
        if len(trajectory) < 2:
            return {
                'converged': False,
                'trend': 'insufficient_data',
                'peak_iteration': 1
            }
        
        # 提取NMI序列
        nmi_sequence = [step['nmi'] for step in trajectory]
        
        # 检测趋势
        if len(nmi_sequence) >= 3:
            if nmi_sequence[-1] > nmi_sequence[-2] > nmi_sequence[-3]:
                trend = 'improving'
            elif nmi_sequence[-1] < nmi_sequence[-2] < nmi_sequence[-3]:
                trend = 'declining'
            elif abs(nmi_sequence[-1] - nmi_sequence[-2]) < 0.01:
                trend = 'converged'
            else:
                trend = 'fluctuating'
        else:
            trend = 'early_stage'
        
        # 找到峰值迭代
        peak_iteration = np.argmax(nmi_sequence) + 1
        
        # 判断是否收敛
        converged = trend == 'converged' or (
            len(nmi_sequence) >= 3 and 
            all(abs(nmi_sequence[i] - nmi_sequence[i-1]) < 0.005 
                for i in range(-2, 0))
        )
        
        return {
            'converged': converged,
            'trend': trend,
            'peak_iteration': peak_iteration
        }
    
    def _get_max_iterations(self) -> int:
        """获取最大迭代次数"""
        if self.config:
            return self.config.get('failure_learning.max_iterations', 5)
        return 5
    
    def _get_improvement_threshold(self) -> float:
        """获取改进阈值"""
        if self.config:
            return self.config.get('failure_learning.improvement_threshold', 0.01)
        return 0.01
    
    def _get_convergence_patience(self) -> int:
        """获取收敛耐心值"""
        if self.config:
            return self.config.get('failure_learning.convergence_patience', 2)
        return 2

if __name__ == "__main__":
    print("迭代学习器模块测试")