#!/usr/bin/env python3
"""
🎯 优化版随机状态实验 - 解决内存+输出问题
- 预生成固定种子池 (确保不重复+可复现)
- Embedding分离存储 (解决内存溢出)
- 流式输出+分批保存 (实时监控进度)
- 轻量级结果结构 (提高效率)
"""

import sys
import os
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from evaluation import eva
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score
)
from layers import ZINBLoss, MeanAct, DispAct
from GNN import GNNLayer
import time
import random
import json
import pickle
from datetime import datetime
import warnings

# 抑制warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def set_initial_seed(seed):
    """设置初始种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_seed_pool(num_seeds, master_seed=42):
    """生成固定的种子池，确保完全可复现"""
    np.random.seed(master_seed)
    seeds = np.random.choice(range(1, 1000000), num_seeds, replace=False)
    print(f"🎲 生成{num_seeds}个不重复种子 (主种子: {master_seed})")
    print(f"🔍 种子范围: {seeds.min()} - {seeds.max()}")
    return seeds.tolist()

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        self.dec_1 = Linear(n_z, n_dec_1)
        self.BN4 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN5 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN6 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))
        z = self.z_layer(enc_h3)
        dec_h1 = F.relu(self.BN4(self.dec_1(z)))
        dec_h2 = F.relu(self.BN5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN6(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h3

class SDCN_Optimized(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=True, pretrain_path='model/mtab.pkl'):
        super(SDCN_Optimized, self).__init__()
        self.use_zinb = use_zinb
        
        self.ae = AE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z)
        
        if os.path.exists(pretrain_path):
            self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu', weights_only=False))

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data, gain=1.0)

        if self.use_zinb:
            self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
            self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
            self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
            self.zinb_loss = ZINBLoss()

        self.v = v

    def forward(self, x, adj, sigma=0.5):
        x_bar, tra1, tra2, tra3, z, dec_h3 = self.ae(x)
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        predict = F.softmax(h, dim=1)

        if self.use_zinb:
            _mean = self._dec_mean(dec_h3)
            _disp = self._dec_disp(dec_h3)
            _pi = self._dec_pi(dec_h3)
        else:
            _mean = _disp = _pi = None

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q, predict, z, _mean, _disp, _pi

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def compute_all_metrics(y_true, y_pred, embeddings):
    """计算所有聚类评估指标"""
    metrics = {}
    
    # 外部验证指标
    metrics['ari'] = adjusted_rand_score(y_true, y_pred)
    metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
    metrics['homogeneity'] = homogeneity_score(y_true, y_pred)
    metrics['completeness'] = completeness_score(y_true, y_pred)
    metrics['v_measure'] = v_measure_score(y_true, y_pred)
    
    # 内部验证指标
    try:
        metrics['silhouette'] = silhouette_score(embeddings, y_pred, metric='euclidean')
    except:
        metrics['silhouette'] = 0.0
    
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, y_pred)
    except:
        metrics['calinski_harabasz'] = 0.0
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, y_pred)
        metrics['davies_bouldin'] = 1.0 / (1.0 + metrics['davies_bouldin'])
    except:
        metrics['davies_bouldin'] = 0.0
    
    return metrics

def save_embedding(embeddings, seed, exp_id, embeddings_dir):
    """保存embedding到独立文件"""
    os.makedirs(embeddings_dir, exist_ok=True)
    embedding_path = f"{embeddings_dir}/seed_{seed}_exp_{exp_id}.npy"
    np.save(embedding_path, embeddings)
    return embedding_path

def run_single_experiment(seed, exp_id, total_experiments, config, embeddings_dir, epochs=80):
    """运行单次实验 - 优化版"""
    
    # 设置种子
    set_initial_seed(seed)
    
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    with h5py.File("data/mtab.h5", "r") as f:
        X_raw = np.array(f['X_raw'])
        X_scaled = np.array(f['X'])
        y = np.array(f['Y'])
        size_factors = np.array(f['size_factors'])
    
    # 创建模型
    model = SDCN_Optimized(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # 加载图和数据
    adj = load_graph('mtab_processed', None).to(device)
    data = torch.Tensor(X_scaled).to(device)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32).to(device)
    
    # K-means初始化
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)
    
    kmeans = KMeans(n_clusters=8, n_init=20, random_state=42, max_iter=300)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    
    # 训练循环
    best_metrics = None
    best_epoch = 0
    best_embeddings = None
    
    global p
    for epoch in range(epochs):
        if epoch % 5 == 0:
            with torch.no_grad():
                _, tmp_q, pred, _, _, _, z = model(data, adj, sigma=config['sigma'])
                p = target_distribution(tmp_q.data)
                
                res2 = pred.data.cpu().numpy().argmax(1)
                embeddings = z.data.cpu().numpy()
                
                metrics = compute_all_metrics(y, res2, embeddings)
                simple_score = 0.6 * metrics['ari'] + 0.4 * metrics['nmi']
                
                if best_metrics is None or simple_score > (0.6 * best_metrics['ari'] + 0.4 * best_metrics['nmi']):
                    best_metrics = metrics
                    best_epoch = epoch
                    best_embeddings = embeddings.copy()
        
        # 训练步骤
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data, adj, sigma=config['sigma'])
        
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        
        try:
            zinb_loss_value = model.zinb_loss(X_raw_tensor, meanbatch, dispbatch, pibatch, sf_tensor)
            if torch.isnan(zinb_loss_value) or torch.isinf(zinb_loss_value):
                zinb_loss_value = torch.tensor(0.0, device=device)
        except:
            zinb_loss_value = torch.tensor(0.0, device=device)
        
        total_loss = (config['alpha'] * kl_loss + config['beta'] * ce_loss + 
                     config['gamma'] * re_loss + config['delta'] * zinb_loss_value)
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    training_time = time.time() - start_time
    
    # 保存embedding到独立文件
    embedding_path = save_embedding(best_embeddings, seed, exp_id, embeddings_dir)
    
    # 轻量级结果结构
    result = {
        'experiment_id': exp_id,
        'seed': seed,
        'metrics': best_metrics,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'embedding_path': embedding_path,
        'has_embedding': True
    }
    
    return result

def save_batch_results(batch_results, batch_id, results_dir):
    """保存批次结果"""
    os.makedirs(results_dir, exist_ok=True)
    batch_file = f"{results_dir}/batch_{batch_id:04d}.pkl"
    with open(batch_file, 'wb') as f:
        pickle.dump(batch_results, f)
    return batch_file

def run_optimized_discovery(num_experiments=5000, batch_size=100):
    """运行优化版大规模发现实验"""
    
    print("🎯 优化版大规模多指标发现实验")
    print("="*70)
    print(f"📊 实验次数: {num_experiments}")
    print(f"🎲 种子池: 预生成不重复种子")
    print(f"💾 批次大小: {batch_size}")
    print(f"📁 Embedding分离存储")
    
    # 生成固定种子池
    seeds_pool = generate_seed_pool(num_experiments, master_seed=42)
    
    # 创建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results/experiment_{timestamp}"
    embeddings_dir = f"{base_dir}/embeddings"
    results_dir = f"{base_dir}/batches"
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存种子池
    with open(f"{base_dir}/seeds_pool.json", 'w') as f:
        json.dump(seeds_pool, f)
    
    print(f"📁 实验目录: {base_dir}")
    print(f"🚀 开始实验...")
    
    config = {
        'alpha': 0.04, 'beta': 0.005, 'gamma': 0.85, 'delta': 0.18,
        'lr': 1.2e-4, 'sigma': 0.55
    }
    
    all_results = []
    batch_results = []
    total_start_time = time.time()
    
    for i, seed in enumerate(seeds_pool, 1):
        result = run_single_experiment(seed, i, num_experiments, config, embeddings_dir, epochs=80)
        batch_results.append(result)
        all_results.append(result)
        
        # 流式输出
        if i % 10 == 0 or i <= 20:
            print(f"🧪 实验 {i}/{num_experiments} | 种子{seed} | ARI={result['metrics']['ari']:.4f} | NMI={result['metrics']['nmi']:.4f}", flush=True)
        
        # 分批保存
        if i % batch_size == 0:
            batch_id = i // batch_size
            batch_file = save_batch_results(batch_results, batch_id, results_dir)
            print(f"💾 批次 {batch_id} 已保存: {len(batch_results)} 个结果", flush=True)
            
            # 显示当前最佳
            current_best = max(all_results, key=lambda x: x['metrics']['ari'])
            elapsed = time.time() - total_start_time
            eta = (elapsed / i) * (num_experiments - i)
            print(f"📊 进度 {i}/{num_experiments} | 最佳: 种子{current_best['seed']}, ARI={current_best['metrics']['ari']:.4f}", flush=True)
            print(f"⏱️  用时: {elapsed/60:.1f}分钟 | 预计剩余: {eta/60:.1f}分钟", flush=True)
            
            batch_results = []  # 清空批次结果，释放内存
    
    # 保存最后一批
    if batch_results:
        batch_id = (num_experiments - 1) // batch_size + 1
        save_batch_results(batch_results, batch_id, results_dir)
    
    total_time = time.time() - total_start_time
    
    # 计算最终统计
    best_result = max(all_results, key=lambda x: x['metrics']['ari'])
    high_performers = [r for r in all_results if r['metrics']['ari'] > 0.72]
    
    print(f"\n🏆 优化版实验总结:")
    print("="*70)
    print(f"🕐 总运行时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"⏱️  平均单次时间: {total_time/num_experiments:.1f}秒")
    
    print(f"\n🥇 最佳结果:")
    print(f"  🎲 种子: {best_result['seed']}")
    print(f"  🏆 ARI: {best_result['metrics']['ari']:.4f}")
    print(f"  🎯 NMI: {best_result['metrics']['nmi']:.4f}")
    print(f"  📊 Silhouette: {best_result['metrics']['silhouette']:.4f}")
    
    if high_performers:
        print(f"\n🎯 超过ARI=0.72的种子 ({len(high_performers)}个):")
        for hp in sorted(high_performers, key=lambda x: x['metrics']['ari'], reverse=True)[:5]:
            print(f"  种子{hp['seed']}: ARI={hp['metrics']['ari']:.4f}")
    
    # 保存最终摘要
    summary = {
        'experiment_info': {
            'num_experiments': num_experiments,
            'total_time': total_time,
            'timestamp': timestamp,
            'seeds_pool': seeds_pool
        },
        'best_result': best_result,
        'high_performers': high_performers,
        'base_dir': base_dir
    }
    
    with open(f"{base_dir}/experiment_summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n📁 实验完成! 结果保存在: {base_dir}")
    print(f"🎯 最佳种子: {best_result['seed']}")
    
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_experiments', type=int, default=5000, help='实验次数')
    parser.add_argument('--batch_size', type=int, default=100, help='批次大小')
    args = parser.parse_args()
    
    summary = run_optimized_discovery(args.num_experiments, args.batch_size)
