#!/usr/bin/env python3
"""最简单的Usoskin测试"""

# 直接测试策略图谱是否工作
from core.strategy_atlas import StrategyAtlas

def test_strategy_match():
    atlas = StrategyAtlas("data/atlas_knowledge.json")
    
    # 模拟Usoskin几何特征
    usoskin_features = {
        'basic': {
            'n_samples': 622,
            'n_features': 17772
        },
        'dimension': {'effective_dim_90': 491},
        'boundary': {'boundary_ratio': 0.201},
        'shape': {'eccentricity': 1000}
    }
    
    result = atlas.find_best_match(usoskin_features)
    
    print("🧪 策略匹配测试:")
    print(f"   策略名称: {result['name']}")
    print(f"   相似度: {result['similarity']:.3f}")
    print(f"   预期NMI: {result['expected_nmi']:.3f}")
    
    if 'strategy_details' in result:
        details = result['strategy_details']
        if 'pca_components' in details:
            print(f"   PCA维度: {details['pca_components']}")
    
    if result['similarity'] > 0.8:
        print("✅ 策略匹配成功!")
    else:
        print("❌ 策略匹配失败")

if __name__ == "__main__":
    test_strategy_match()
