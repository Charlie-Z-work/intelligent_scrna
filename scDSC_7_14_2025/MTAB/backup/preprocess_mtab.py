import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置scanpy参数
sc.settings.verbosity = 3  # 显示详细信息
# sc.settings.set_figure_params(dpi=80, facecolor='white')



def preprocess_mtab_data():
    """
    对MTAB-3929数据进行标准单细胞预处理
    """
    print("🔄 开始MTAB-3929数据预处理...")
    
    # 1. 读取数据
    print("📖 读取原始数据...")
    X = pd.read_csv("data/mtab.txt", sep='\t', header=None).values
    labels = pd.read_csv("data/mtab_label.txt", header=None).values.squeeze()
    
    print(f"原始数据维度: {X.shape}")
    print(f"标签数量: {len(labels)}")
    
    # 2. 创建AnnData对象
    adata = sc.AnnData(X)
    adata.obs['cell_type'] = labels
    
    # 添加基因和细胞的基本统计信息
    adata.var_names = [f"Gene_{i}" for i in range(adata.n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(adata.n_obs)]
    
    print(f"创建AnnData对象: {adata}")
    
    # 3. 基本质控指标计算
    print("📊 计算质控指标...")
    
    # 每个细胞的总UMI数
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    
    # 每个细胞检测到的基因数
    adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
    
    # 每个基因在多少细胞中表达
    adata.var['n_cells'] = np.array((adata.X > 0).sum(axis=0)).flatten()
    
    # 每个基因的总表达量
    adata.var['total_counts'] = np.array(adata.X.sum(axis=0)).flatten()
    
    print("质控指标统计:")
    print(f"细胞总UMI数: {adata.obs['total_counts'].describe()}")
    print(f"细胞基因数: {adata.obs['n_genes'].describe()}")
    print(f"基因表达细胞数: {adata.var['n_cells'].describe()}")
    
    # 4. 过滤低质量细胞和基因
    print("🔍 过滤低质量数据...")
    
    # 过滤表达基因数太少的细胞（<200个基因）
    print(f"过滤前细胞数: {adata.n_obs}")
    sc.pp.filter_cells(adata, min_genes=200)
    print(f"过滤后细胞数: {adata.n_obs}")
    
    # 过滤在太少细胞中表达的基因（<3个细胞）
    print(f"过滤前基因数: {adata.n_vars}")
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"过滤后基因数: {adata.n_vars}")
    
    # 5. 高变异基因选择
    print("🎯 选择高变异基因...")
    
    # 保存原始计数
    adata.raw = adata
    
    # 归一化到10,000 reads per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # 对数变换
    sc.pp.log1p(adata)
    
    # 识别高变异基因
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    print(f"高变异基因数: {adata.var.highly_variable.sum()}")
    
    # 只保留高变异基因用于下游分析
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    # 6. 标准化
    print("📏 标准化数据...")
    sc.pp.scale(adata_hvg, max_value=10)
    
    # 7. 进一步降维（如果基因数仍然很多）
    if adata_hvg.n_vars > 5000:
        print(f"基因数({adata_hvg.n_vars})仍然较多，进一步选择top 5000基因...")
        # 按方差排序，选择top 5000
        gene_var = np.var(adata_hvg.X, axis=0)
        top_genes_idx = np.argsort(gene_var)[-5000:]
        adata_hvg = adata_hvg[:, top_genes_idx].copy()
    
    print(f"最终数据维度: {adata_hvg.shape}")
    
    # 8. 保存处理后的数据
    print("💾 保存预处理数据...")
    
    # 保存特征矩阵
    X_processed = adata_hvg.X
    if hasattr(X_processed, 'toarray'):  # 如果是sparse matrix
        X_processed = X_processed.toarray()
    
    # 保存为txt格式（用于scDSC）
    np.savetxt("data/mtab_processed.txt", X_processed, delimiter='\t')
    
    # 保存标签（过滤后的）
    labels_processed = adata_hvg.obs['cell_type'].values
    
    # 数字编码标签
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_processed)
    
    np.savetxt("data/mtab_processed_label.txt", labels_encoded, fmt='%d')
    
    # 保存标签映射
    with open("data/mtab_label_mapping_processed.txt", "w") as f:
        for i, label in enumerate(le.classes_):
            f.write(f"{i}\t{label}\n")
    
    # 保存预处理参数信息
    with open("data/preprocessing_info.txt", "w") as f:
        f.write(f"原始数据维度: {X.shape}\n")
        f.write(f"处理后维度: {adata_hvg.shape}\n")
        f.write(f"过滤后细胞数: {adata_hvg.n_obs}\n")
        f.write(f"最终基因数: {adata_hvg.n_vars}\n")
        f.write(f"聚类数: {len(le.classes_)}\n")
        f.write("细胞类型分布:\n")
        for i, (label, count) in enumerate(zip(*np.unique(labels_processed, return_counts=True))):
            f.write(f"  {i}: {label} ({count} cells)\n")
    
    print("✅ 预处理完成！")
    print(f"原始维度: {X.shape}")
    print(f"处理后维度: {adata_hvg.shape}")
    print(f"聚类数: {len(le.classes_)}")
    
    return adata_hvg, le.classes_

if __name__ == "__main__":
    # 运行预处理
    adata_processed, cell_types = preprocess_mtab_data()
    
    print("\n📋 预处理摘要:")
    print(f"- 数据维度: {adata_processed.shape}")
    print(f"- 细胞类型: {cell_types}")
    print(f"- 输出文件: mtab_processed.txt, mtab_label_processed.txt")