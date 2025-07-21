import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import os

# 确保graph目录存在
os.makedirs('graph', exist_ok=True)

topk = 10

def construct_graph(features, label, method):
    fname = 'graph/mtab_processed_graph.txt'  # 🔧 修正文件名
    num = len(label)

    dist = None
    
    # 🔧 修复相似度计算方法
    if method == 'heat':
        dist = -0.5 * pair(features, metric='euclidean') ** 2
        sigma = np.std(dist)
        dist = np.exp(dist / (2 * sigma ** 2))

    elif method == 'cos':
        # 余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        dist = cosine_similarity(features)

    elif method == 'ncos':
        # 归一化余弦相似度
        features_norm = normalize(features, axis=1, norm='l2')
        dist = np.dot(features_norm, features_norm.T)

    elif method == 'p':
        # Pearson相关系数 - 🔧 修复计算方法
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        dist = np.corrcoef(features_scaled)
        # 处理NaN值
        dist = np.nan_to_num(dist, nan=0.0)

    # 🔧 添加数值稳定性检查
    if np.any(np.isnan(dist)) or np.any(np.isinf(dist)):
        print("⚠️ 距离矩阵包含NaN或Inf，进行修复...")
        dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 确保对角线为最大值（自相似度）
    np.fill_diagonal(dist, np.max(dist))
    
    print(f"距离矩阵统计: min={np.min(dist):.4f}, max={np.max(dist):.4f}, mean={np.mean(dist):.4f}")

    # 🔧 改进KNN图构建
    inds = []
    for i in range(dist.shape[0]):
        # 获取top-k最相似的邻居（包括自己）
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        # 按相似度排序
        ind = ind[np.argsort(-dist[i, ind])]
        inds.append(ind)

    # 写入图文件
    with open(fname, 'w') as f:
        counter = 0
        total_edges = 0
        
        for i, neighbors in enumerate(inds):
            for neighbor in neighbors:
                if neighbor != i:  # 排除自环
                    if label[neighbor] != label[i]:
                        counter += 1
                    f.write(f'{i} {neighbor}\n')
                    total_edges += 1
    
    error_rate = counter / total_edges if total_edges > 0 else 0
    print(f'KNN图构建完成:')
    print(f'  - 总边数: {total_edges}')
    print(f'  - 错误边数: {counter}')
    print(f'  - 错误率: {error_rate:.4f}')
    print(f'  - 文件保存至: {fname}')

def main():
    # 🔧 使用预处理后的数据文件
    data_file = 'data/mtab_processed.txt'
    label_file = 'data/mtab_processed_label.txt'
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行 python preprocess_mtab.py")
        return
    
    if not os.path.exists(label_file):
        print(f"❌ 标签文件不存在: {label_file}")
        print("请先运行 python preprocess_mtab.py") 
        return

    print("📖 读取预处理数据...")
    features = np.loadtxt(data_file, dtype=float)
    labels = np.loadtxt(label_file, dtype=int)
    
    print(f"数据维度: {features.shape}")
    print(f"标签数: {len(np.unique(labels))}")
    print(f"标签分布: {np.bincount(labels)}")
    
    # 🔧 尝试不同的相似度计算方法
    methods = ['p', 'cos', 'ncos', 'heat']  # Pearson效果通常最好
    method_names = ['Pearson', 'Cosine', 'Normalized Cosine', 'Heat Kernel']
    
    print("\n🔍 测试不同相似度方法的错误率:")
    for method, name in zip(methods, method_names):
        print(f"\n--- {name} 方法 ---")
        try:
            construct_graph(features, labels, method)
        except Exception as e:
            print(f"❌ {name} 方法失败: {e}")
    
    print(f"\n✅ 推荐使用 Pearson 方法构建的图文件")

if __name__ == "__main__":
    main()