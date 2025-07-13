import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

print("🔧 ZINB友好的预处理...")

# 1. 读取原始计数数据
X_raw = np.loadtxt("data/mtab.txt", dtype=float)
labels = np.loadtxt("data/mtab_label.txt", dtype=int)

print(f"原始计数数据: {X_raw.shape}")

# 2. 基本过滤（保持计数性质）
# 过滤表达量很低的基因
gene_counts = np.sum(X_raw > 0, axis=0)  # 每个基因在多少细胞中表达
keep_genes = gene_counts >= 3  # 至少在3个细胞中表达
X_filtered = X_raw[:, keep_genes]

print(f"基因过滤后: {X_filtered.shape}")

# 3. 选择高变异基因（基于计数数据）
if X_filtered.shape[1] > 5000:
    gene_var = np.var(X_filtered, axis=0)
    top_indices = np.argsort(gene_var)[-5000:]
    X_final = X_filtered[:, top_indices]
else:
    X_final = X_filtered

print(f"最终计数数据: {X_final.shape}")

# 4. 确保数据为非负
X_final = np.maximum(X_final, 0)

# 5. 计算正确的size factors
library_sizes = np.sum(X_final, axis=1)
median_library = np.median(library_sizes)
size_factors = library_sizes / median_library

print(f"Size factors范围: [{size_factors.min():.3f}, {size_factors.max():.3f}]")

# 6. 编码标签
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# 7. 保存原始计数数据（用于ZINB）
np.savetxt("data/mtab_counts_raw.txt", X_final, delimiter='\t', fmt='%.0f')
np.savetxt("data/mtab_processed_label.txt", labels_encoded, fmt='%d')

# 8. 创建标准化版本（用于其他模块）
X_normalized = X_final / library_sizes[:, np.newaxis] * median_library
X_log = np.log1p(X_normalized)
# 轻度标准化
X_scaled = (X_log - np.mean(X_log, axis=0)) / (np.std(X_log, axis=0) + 1e-8)
X_scaled = np.clip(X_scaled, -10, 10)  # 限制范围

np.savetxt("data/mtab_processed.txt", X_scaled, delimiter='\t')

# 9. 创建H5文件（同时包含原始和处理后的数据）
with h5py.File("data/mtab.h5", "w") as f:
    f.create_dataset("X", data=X_scaled)        # 标准化数据（用于GNN等）
    f.create_dataset("X_raw", data=X_final)     # 原始计数（用于ZINB）
    f.create_dataset("Y", data=labels_encoded)
    f.create_dataset("size_factors", data=size_factors)

print("✅ ZINB友好预处理完成!")
print(f"原始计数: {X_final.shape}, size factors正常: {size_factors.min():.3f}-{size_factors.max():.3f}")
