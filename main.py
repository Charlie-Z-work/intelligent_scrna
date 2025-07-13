#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能单细胞RNA测序分析系统 - 主控文件
基于几何特征和边界失败学习的自适应算法选择框架

作者: [Your Name]
版本: 1.0.0
核心思想: "边界失败不是噪声，而是结构演化的信号"
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from core.geometry_analyzer import GeometryAnalyzer
from core.strategy_atlas import StrategyAtlas
from core.iterative_learner import IterativeLearner
from core.failure_analyzer import FailureAnalyzer
from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator
from utils.visualization import ResultVisualizer
from config import Config

class IntelligentScRNAAtlas:
    """
    智能单细胞RNA测序分析系统
    
    核心功能:
    1. 几何特征分析
    2. 策略图谱匹配
    3. 失败学习优化
    4. 知识库更新
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化系统"""
        self.config = Config(config_path)
        
        # 核心组件
        self.geometry_analyzer = GeometryAnalyzer(self.config)
        self.strategy_atlas = StrategyAtlas(self.config.atlas_path)
        self.failure_analyzer = FailureAnalyzer(self.config)
        self.iterative_learner = IterativeLearner(
            self.strategy_atlas, 
            self.failure_analyzer,
            self.config
        )
        
        # 工具组件
        self.data_loader = DataLoader(self.config)
        self.metrics_calc = MetricsCalculator()
        self.visualizer = ResultVisualizer(self.config)
        
        print(f"🌟 智能scRNA分析系统已启动")
        print(f"💡 核心思想: {self.config.core_philosophy}")
    
    def analyze_dataset(self, 
                       data_path: str, 
                       labels_path: Optional[str] = None,
                       dataset_name: str = "unknown",
                       max_iterations: int = 5,
                       save_results: bool = True) -> Dict[str, Any]:
        """
        分析单个数据集
        
        流程:
        1. 数据加载和预处理
        2. 几何特征分析
        3. 策略图谱匹配
        4. 迭代失败学习
        5. 结果保存和可视化
        """
        
        print(f"\n🔬 开始分析数据集: {dataset_name}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 数据加载
        print("📊 加载数据...")
        X, y_true = self.data_loader.load_dataset(data_path, labels_path)
        print(f"   数据维度: {X.shape}")
        if y_true is not None:
            print(f"   类别数: {len(np.unique(y_true))}")
        
        # 2. 几何特征分析
        print("\n🔍 分析几何特征...")
        geometry_features = self.geometry_analyzer.analyze(X, y_true)
        self._print_geometry_summary(geometry_features)
        
        # 3. 策略图谱匹配
        print("\n🗺️ 策略图谱匹配...")
        initial_strategy = self.strategy_atlas.find_best_match(geometry_features)
        
        # 确保策略详细配置被正确传递
        if 'strategy_details' in initial_strategy:
            strategy_details = initial_strategy['strategy_details']
            # 将详细配置合并到初始策略中
            for key, value in strategy_details.items():
                if key not in initial_strategy:
                    initial_strategy[key] = value
            print(f"   📋 应用策略配置: PCA-{initial_strategy.get('pca_components', 'unknown')}维")
        print(f"   推荐策略: {initial_strategy['name']}")
        print(f"   匹配相似度: {initial_strategy['similarity']:.3f}")
        print(f"   预期性能: {initial_strategy['expected_nmi']:.3f}")
        
        # 4. 迭代失败学习
        print("\n🎯 迭代学习优化...")
        learning_result = self.iterative_learner.optimize(
            X, y_true, initial_strategy, max_iterations
        )

        # 增加安全检查
        if learning_result is None:
            print("⚠️ 学习结果为空，创建备用结果")
            learning_result = {
                'trajectory': [],
                'final_performance': {'nmi': 0.0, 'ari': 0.0, 'silhouette': 0.0},
                'total_improvement': 0,
                'iterations_used': 0
            }

        # 5. 结果整理
        final_result = {
            'dataset_name': dataset_name,
            'data_shape': X.shape,
            'geometry_features': geometry_features,
            'initial_strategy': initial_strategy,
            'learning_trajectory': learning_result['trajectory'],
            'final_performance': learning_result['final_performance'],
            'total_improvement': learning_result['total_improvement'],
            'iterations_used': learning_result['iterations_used'],
            'processing_time': time.time() - start_time
        }
        
        # 6. 结果展示
        self._print_final_results(final_result)
        
        # 7. 保存和可视化
        if save_results:
            self._save_results(final_result, dataset_name)
            self.visualizer.plot_learning_trajectory(final_result)
        
        # 8. 更新图谱知识库
        self._update_atlas_knowledge(final_result)
        
        return final_result
    
    def batch_analyze(self, 
                     datasets_config: str,
                     output_dir: str = "results") -> Dict[str, Any]:
        """
        批量分析多个数据集
        """
        print(f"\n🚀 批量分析模式")
        
        # 加载数据集配置
        datasets = self.data_loader.load_datasets_config(datasets_config)
        
        batch_results = {}
        summary_stats = {
            'total_datasets': len(datasets),
            'successful_analyses': 0,
            'average_improvement': 0,
            'best_strategies': {}
        }
        
        for dataset_name, dataset_info in datasets.items():
            try:
                print(f"\n处理数据集 {dataset_name}...")
                result = self.analyze_dataset(
                    dataset_info['data_path'],
                    dataset_info.get('labels_path'),
                    dataset_name,
                    save_results=True
                )
                
                batch_results[dataset_name] = result
                summary_stats['successful_analyses'] += 1
                
            except Exception as e:
                print(f"❌ 数据集 {dataset_name} 分析失败: {e}")
                batch_results[dataset_name] = {'error': str(e)}
        
        # 生成批量分析报告
        self._generate_batch_report(batch_results, summary_stats, output_dir)
        
        return batch_results
    
    def benchmark_against_baselines(self, 
                                  datasets_config: str,
                                  baseline_methods: list = None) -> Dict[str, Any]:
        """
        与基线方法对比测试
        """
        if baseline_methods is None:
            baseline_methods = ['SIMLR', 'scDSC', 'Seurat', 'Scanpy']
        
        print(f"\n📊 基准测试模式")
        print(f"对比方法: {baseline_methods}")
        
        # 实现基线方法对比逻辑
        benchmark_results = {}
        
        # TODO: 实现具体的基线方法调用
        
        return benchmark_results
    
    def _print_geometry_summary(self, features: Dict[str, Any]):
        """打印几何特征摘要"""
        print(f"   样本数: {features['basic']['n_samples']}")
        print(f"   特征数: {features['basic']['n_features']}")
        print(f"   有效维度(90%): {features['dimension']['effective_dim_90']}")
        print(f"   内在维度: {features['dimension']['intrinsic_dim_estimate']:.1f}")
        print(f"   边界点比例: {features['boundary']['boundary_ratio']:.2%}")
        print(f"   数据椭球性: {features['shape']['eccentricity']:.2f}")
    
    def _print_final_results(self, result: Dict[str, Any]):
        """打印最终结果"""
        print(f"\n🎉 分析完成!")
        print("=" * 60)
        print(f"📈 最终性能:")
        perf = result['final_performance']
        print(f"   NMI: {perf['nmi']:.4f}")
        print(f"   ARI: {perf['ari']:.4f}")
        print(f"   总体改进: {result['total_improvement']:+.4f}")
        print(f"   迭代次数: {result['iterations_used']}")
        print(f"   处理时间: {result['processing_time']:.1f}秒")
        
        # 学习轨迹概要
        trajectory = result['learning_trajectory']
        print(f"\n📊 学习轨迹:")
        for i, step in enumerate(trajectory):
            print(f"   第{i+1}次: NMI={step['nmi']:.4f}, 策略={step['strategy_name']}")
    
    def _save_results(self, result: Dict[str, Any], dataset_name: str):
        """保存结果 - 安全版本"""
        output_dir = Path(self.config.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 创建安全的结果字典
            safe_result = {
                'dataset_name': str(result.get('dataset_name', 'unknown')),
                'data_shape': [int(x) for x in result.get('data_shape', [0, 0])],
                'final_performance': {
                    'nmi': float(result.get('final_performance', {}).get('nmi', 0)),
                    'ari': float(result.get('final_performance', {}).get('ari', 0))
                },
                'total_improvement': float(result.get('total_improvement', 0)),
                'iterations_used': int(result.get('iterations_used', 0)),
                'processing_time': float(result.get('processing_time', 0))
            }
            
            # 保存简化结果
            import json
            with open(output_dir / 'analysis_result.json', 'w') as f:
                json.dump(safe_result, f, indent=2)
            
            print(f"💾 结果已保存到: {output_dir}")
            
        except Exception as e:
            print(f"⚠️ JSON保存失败，使用文本格式: {e}")
            # 备用：保存为文本
            with open(output_dir / 'analysis_result.txt', 'w') as f:
                f.write(f"Dataset: {result.get('dataset_name', 'unknown')}\n")
                f.write(f"Data Shape: {result.get('data_shape', [0, 0])}\n")
                perf = result.get('final_performance', {})
                f.write(f"NMI: {perf.get('nmi', 0):.4f}\n")
                f.write(f"ARI: {perf.get('ari', 0):.4f}\n")
                f.write(f"Total Improvement: {result.get('total_improvement', 0):.4f}\n")
                f.write(f"Iterations: {result.get('iterations_used', 0)}\n")
                f.write(f"Time: {result.get('processing_time', 0):.1f}s\n")
            print(f"💾 备用文本结果已保存到: {output_dir}")
    
    def _update_atlas_knowledge(self, result: Dict[str, Any]):
        """更新图谱知识库"""
        self.strategy_atlas.update_knowledge(result)
        print("🧠 图谱知识库已更新")
    
    def _make_json_serializable(self, obj):
        """增强的JSON序列化处理"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (np.generic)):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, pd.DataFrame, pd.Series)):
            return str(obj)
        elif isinstance(obj, (tuple)):
            return list(obj)
        elif isinstance(obj, (dict)):
            return {self._make_json_serializable(k): self._make_json_serializable(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, (list)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            try:
                return str(obj)
            except:
                print(f"[警告] 无法序列化类型: {type(obj)}")
                return None

    
    def _generate_batch_report(self, results: Dict, stats: Dict, output_dir: str):
        """生成批量分析报告"""
        # TODO: 实现详细的批量报告生成
        pass

def main():
    """主函数"""
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="智能单细胞RNA测序分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个数据集
  python main.py --data data/usoskin.csv --labels data/usoskin_labels.csv --name Usoskin
  
  # 批量分析
  python main.py --batch datasets_config.json
  
  # 基准测试
  python main.py --benchmark datasets_config.json --baselines SIMLR,scDSC
        """
    )
    
    # 单数据集分析参数
    parser.add_argument('--data', type=str, help='数据文件路径')
    parser.add_argument('--labels', type=str, help='标签文件路径（可选）')
    parser.add_argument('--name', type=str, default='unknown', help='数据集名称')
    parser.add_argument('--max-iter', type=int, default=5, help='最大迭代次数')
    
    # 批量分析参数
    parser.add_argument('--batch', type=str, help='批量分析配置文件')
    
    # 基准测试参数
    parser.add_argument('--benchmark', type=str, help='基准测试配置文件')
    parser.add_argument('--baselines', type=str, help='基线方法列表（逗号分隔）')
    
    # 系统配置参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    
    args = parser.parse_args()
    
    # 创建系统实例
    try:
        system = IntelligentScRNAAtlas(args.config)
        
        # 根据参数选择运行模式
        if args.batch:
            # 批量分析模式
            results = system.batch_analyze(args.batch, args.output)
            
        elif args.benchmark:
            # 基准测试模式
            baselines = args.baselines.split(',') if args.baselines else None
            results = system.benchmark_against_baselines(args.benchmark, baselines)
            
        elif args.data:
            # 单数据集分析模式
            result = system.analyze_dataset(
                args.data, 
                args.labels, 
                args.name,
                args.max_iter
            )
            
        else:
            parser.print_help()
            return
        
        print(f"\n✅ 系统运行完成!")
        
    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    total_time = time.time() - start_time
    print(f"\n🕒 程序总运行时间: {total_time:.2f} 秒")

    return 0

if __name__ == "__main__":
    exit(main())
