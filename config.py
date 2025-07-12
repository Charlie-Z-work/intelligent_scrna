#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能单细胞RNA测序分析系统 - 配置文件
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """系统配置管理"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 核心理念
        self.core_philosophy = "边界失败不是噪声，而是结构演化的信号"
        
        # 默认配置
        self.default_config = {
            # 系统设置
            "system": {
                "version": "1.0.0",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": True
            },
            
            # 数据处理设置
            "data_processing": {
                "min_genes_per_cell": 200,
                "min_cells_per_gene": 3,
                "max_genes_per_cell": 30000,
                "normalization_method": "log1p",
                "scaling_method": "standard"
            },
            
            # 几何特征分析设置
            "geometry_analysis": {
                "boundary_threshold": 0.8,
                "knn_neighbors": 10,
                "pca_components_for_analysis": 50,
                "intrinsic_dim_method": "mle",
                "density_estimation_method": "knn"
            },
            
            # 策略图谱设置
            "strategy_atlas": {
                "similarity_threshold": 0.5,
                "max_similar_cases": 5,
                "feature_weights": {
                    "effective_dim_90": 0.3,
                    "boundary_ratio": 0.25,
                    "eccentricity": 0.2,
                    "intrinsic_dim": 0.15,
                    "sample_density": 0.1
                },
                "similarity_metric": "weighted_euclidean"
            },
            
            # 失败学习设置
            "failure_learning": {
                "max_iterations": 5,
                "improvement_threshold": 0.01,
                "convergence_patience": 2,
                "failure_analysis_methods": [
                    "boundary_confusion",
                    "dimension_mismatch", 
                    "parameter_suboptimal",
                    "algorithm_mismatch"
                ]
            },
            
            # 算法配置
            "algorithms": {
                "boundary_failure_learning": {
                    "failure_threshold": 0.8,
                    "pca_dimensions": [10, 15, 20, 30, 50],
                    "optimization_steps": 3
                },
                "enhanced_sre": {
                    "knn_k_candidates": [5, 10, 15, 20, 30],
                    "pca_dim_candidates": [15, 20, 30, 50, 100],
                    "covariance_types": ["full", "tied", "diag", "spherical"]
                },
                "ultimate_fusion": {
                    "dimension_candidates": [10, 15, 20, 30, 50, 100],
                    "n_trials": 3,
                    "stability_bonus": 0.05
                }
            },
            
            # 评估设置
            "evaluation": {
                "metrics": ["nmi", "ari", "silhouette", "calinski_harabasz"],
                "primary_metric": "nmi",
                "performance_targets": {
                    "excellent": 0.9,
                    "good": 0.8,
                    "acceptable": 0.6
                }
            },
            
            # 可视化设置
            "visualization": {
                "figure_size": [12, 8],
                "dpi": 300,
                "style": "seaborn-v0_8",
                "color_palette": "viridis",
                "save_formats": ["png", "pdf"]
            },
            
            # 文件路径设置
            "paths": {
                "atlas_knowledge": "data/atlas_knowledge.json",
                "benchmark_results": "data/benchmark_results.json",
                "output_dir": "results",
                "log_dir": "logs",
                "cache_dir": "cache"
            },
            
            # 知识库初始化
            "initial_knowledge": {
                "ultra_high_dim_medium_classes": {
                    "pattern": {
                        "n_features_range": [15000, 50000],
                        "n_classes_range": [3, 5],
                        "effective_dim_90_range": [15, 35],
                        "boundary_ratio_range": [0.15, 0.35],
                        "sample_density": "medium"
                    },
                    "best_strategies": [
                        {
                            "name": "boundary_failure_learning",
                            "expected_nmi": 0.90,
                            "confidence": 0.9,
                            "evidence": [
                                {"dataset": "Usoskin", "nmi": 0.9097, "features": 17772, "classes": 4}
                            ]
                        }
                    ]
                },
                "high_dim_few_classes": {
                    "pattern": {
                        "n_features_range": [8000, 15000],
                        "n_classes_range": [1, 3],
                        "effective_dim_90_range": [10, 25],
                        "boundary_ratio_range": [0.05, 0.25],
                        "sample_density": "low_medium"
                    },
                    "best_strategies": [
                        {
                            "name": "ultimate_fusion_framework",
                            "expected_nmi": 0.95,
                            "confidence": 0.85,
                            "evidence": [
                                {"dataset": "mESC", "nmi": 0.9636, "features": 8989, "classes": 3},
                                {"dataset": "Kolod", "nmi": 0.9915, "features": 10685, "classes": 3}
                            ]
                        }
                    ]
                },
                "ultra_high_dim_many_classes": {
                    "pattern": {
                        "n_features_range": [30000, 50000],
                        "n_classes_range": [8, 15],
                        "effective_dim_90_range": [20, 50],
                        "boundary_ratio_range": [0.3, 0.5],
                        "sample_density": "high"
                    },
                    "best_strategies": [
                        {
                            "name": "ultimate_fusion_framework",
                            "expected_nmi": 0.75,
                            "confidence": 0.8,
                            "evidence": [
                                {"dataset": "E-MTAB", "nmi": 0.7022, "features": 33501, "classes": 8},
                                {"dataset": "Pollen", "nmi": 1.0000, "features": 14805, "classes": 11}
                            ]
                        }
                    ]
                }
            }
        }
        
        # 加载自定义配置
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self.config = self.default_config.copy()
        
        # 设置路径属性
        self._setup_paths()
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 深度合并配置
            self.config = self._deep_merge(self.default_config, user_config)
            print(f"✅ 配置文件加载成功: {config_path}")
            
        except Exception as e:
            print(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
            self.config = self.default_config.copy()
    
    def save_config(self, config_path: str):
        """保存当前配置"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ 配置已保存: {config_path}")
        except Exception as e:
            print(f"❌ 配置保存失败: {e}")
    
    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """深度合并字典"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _setup_paths(self):
        """设置路径属性"""
        paths = self.config["paths"]
        
        # 创建必要的目录
        for dir_key in ["output_dir", "log_dir", "cache_dir"]:
            if dir_key in paths:
                Path(paths[dir_key]).mkdir(parents=True, exist_ok=True)
        
        # 设置属性以便快速访问
        self.atlas_path = paths["atlas_knowledge"]
        self.output_dir = paths["output_dir"]
        self.log_dir = paths["log_dir"]
        self.cache_dir = paths["cache_dir"]
    
    def get(self, key_path: str, default=None):
        """
        获取配置值，支持点分隔的路径
        例如: config.get("algorithms.boundary_failure_learning.failure_threshold")
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值，支持点分隔的路径
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到最后一级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """从字典批量更新配置"""
        for key_path, value in updates.items():
            self.set(key_path, value)
    
    def get_algorithm_config(self, algorithm_name: str) -> Dict[str, Any]:
        """获取特定算法的配置"""
        return self.get(f"algorithms.{algorithm_name}", {})
    
    def get_initial_knowledge(self) -> Dict[str, Any]:
        """获取初始知识库"""
        return self.get("initial_knowledge", {})
    
    def __getattr__(self, name):
        """支持直接属性访问"""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __repr__(self):
        return f"Config(version={self.get('system.version')}, core_philosophy='{self.core_philosophy}')"
    
    def print_summary(self):
        """打印配置摘要"""
        print("📋 系统配置摘要:")
        print(f"   版本: {self.get('system.version')}")
        print(f"   核心理念: {self.core_philosophy}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   图谱路径: {self.atlas_path}")
        print(f"   主要评估指标: {self.get('evaluation.primary_metric')}")
        print(f"   最大迭代次数: {self.get('failure_learning.max_iterations')}")

# 创建默认配置实例
default_config = Config()

if __name__ == "__main__":
    # 配置文件测试
    config = Config()
    config.print_summary()
    
    # 测试配置访问
    print(f"\n测试配置访问:")
    print(f"边界阈值: {config.get('geometry_analysis.boundary_threshold')}")
    print(f"算法列表: {list(config.get('algorithms', {}).keys())}")
    
    # 保存示例配置
    config.save_config("example_config.json")
    print(f"\n示例配置已保存到 example_config.json")