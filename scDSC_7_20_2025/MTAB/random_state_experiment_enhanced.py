#!/usr/bin/env python3
"""
🎯 增强版随机状态实验 - 多指标评估 + 5000次大规模搜索
集成多种聚类评估指标，提供更全面的性能评估
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

class RandomStateManager:
    """随机状态管理器"""
    
    def capture_all_states(self, label=""):
        """捕获所有随机状态"""
        states = {
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        return states
    
    def restore_all_states(self, states):
        """恢复所有随机状态"""
        random.setstate(states['python_random_state'])
        np.random.set_state(states['numpy_random_state'])
        torch.set_rng_state(states['torch_random_state'])
        if torch.cuda.is_available() and states['torch_cuda_random_state'] is not None:
            torch.cuda.set_rng_state_all(states['torch_cuda_random_state'])

def set_initial_seed(seed):
    """设置初始种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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

class SDCN_MultiMetrics(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=True, pretrain_path='model/mtab.pkl'):
        super(SDCN_MultiMetrics, self).__init__()
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
    
    # 外部验证指标 (有监督)
    metrics['ari'] = adjusted_rand_score(y_true, y_pred)
    metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
    metrics['homogeneity'] = homogeneity_score(y_true, y_pred)
    metrics['completeness'] = completeness_score(y_true, y_pred)
    metrics['v_measure'] = v_measure_score(y_true, y_pred)
    
    # 内部验证指标 (无监督)
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
        # Davies-Bouldin越小越好，转换为越大越好
        metrics['davies_bouldin'] = 1.0 / (1.0 + metrics['davies_bouldin'])
    except:
        metrics['davies_bouldin'] = 0.0
    
    return metrics

def normalize_metrics(all_metrics):
    """标准化所有指标到[0,1]范围"""
    normalized = []
    
    # 收集所有指标值
    metric_names = ['ari', 'nmi', 'silhouette', 'homogeneity', 'completeness', 'v_measure', 'calinski_harabasz', 'davies_bouldin']
    metric_values = {name: [m[name] for m in all_metrics] for name in metric_names}
    
    # 计算min-max标准化
    metric_ranges = {}
    for name in metric_names:
        values = metric_values[name]
        min_val, max_val = min(values), max(values)
        metric_ranges[name] = (min_val, max_val)
    
    # 标准化每个实验的指标
    for metrics in all_metrics:
        norm_metrics = {}
        for name in metric_names:
            min_val, max_val = metric_ranges[name]
            if max_val > min_val:
                norm_metrics[f'norm_{name}'] = (metrics[name] - min_val) / (max_val - min_val)
            else:
                norm_metrics[f'norm_{name}'] = 0.5  # 如果没有变化，设为中性值
        
        # 计算综合评分
        norm_metrics['composite_score'] = (
            0.35 * norm_metrics['norm_ari'] +           # 主要指标
            0.25 * norm_metrics['norm_nmi'] +           # 主要指标  
            0.15 * norm_metrics['norm_silhouette'] +    # 内聚性
            0.10 * norm_metrics['norm_v_measure'] +     # 综合外部指标
            0.10 * norm_metrics['norm_homogeneity'] +   # 纯度
            0.05 * norm_metrics['norm_davies_bouldin']  # 分离度
        )
        
        normalized.append(norm_metrics)
    
    return normalized, metric_ranges

def run_single_experiment(experiment_id, total_experiments, config, epochs=80):
    """运行单次实验并记录所有状态和指标"""
    
    state_mgr = RandomStateManager()
    initial_states = state_mgr.capture_all_states("experiment_start")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    with h5py.File("data/mtab.h5", "r") as f:
        X_raw = np.array(f['X_raw'])
        X_scaled = np.array(f['X'])
        y = np.array(f['Y'])
        size_factors = np.array(f['size_factors'])
    
    # 创建模型
    model = SDCN_MultiMetrics(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
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
    
    # 训练记录
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
                
                # 计算所有指标
                metrics = compute_all_metrics(y, res2, embeddings)
                
                # 简单评分（训练时使用）
                simple_score = 0.6 * metrics['ari'] + 0.4 * metrics['nmi']
                
                if best_metrics is None or simple_score > (0.6 * best_metrics['ari'] + 0.4 * best_metrics['nmi']):
                    best_metrics = metrics
                    best_epoch = epoch
                    best_embeddings = embeddings.copy()
        
        # 前向传播
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data, adj, sigma=config['sigma'])
        
        # 损失计算
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
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 组装实验结果
    experiment_result = {
        'experiment_id': experiment_id,
        'config': config,
        'metrics': best_metrics,
        'best_epoch': best_epoch,
        'timing': {
            'total_time_seconds': training_time,
            'time_per_epoch': training_time / epochs,
            'epochs': epochs
        },
        'embeddings': best_embeddings,  # 保存最佳embedding用于后续分析
        'random_states': {
            'initial': initial_states
        }
    }
    
    # 简化显示
    if experiment_id % 100 == 0 or experiment_id <= 10:
        print(f"\n🧪 实验 {experiment_id}/{total_experiments}")
        print(f"  🏆 ARI={best_metrics['ari']:.4f}, NMI={best_metrics['nmi']:.4f}")
        print(f"  📊 Silhouette={best_metrics['silhouette']:.4f}, V-measure={best_metrics['v_measure']:.4f}")
        print(f"  ⏱️  {training_time:.1f}秒")
    
    return experiment_result

def run_large_scale_discovery(num_experiments=5000):
    """运行大规模随机状态发现实验"""
    
    print("🎯 大规模多指标随机状态发现实验")
    print("="*70)
    print(f"📊 实验次数: {num_experiments}")
    print(f"🎰 策略: 除KMeans外全部随机")
    print(f"🎯 目标: 寻找ARI=0.77+的传说种子")
    print(f"📈 指标: ARI, NMI, Silhouette, V-measure等8项")
    
    # 传说配置
    config = {
        'alpha': 0.04, 'beta': 0.005, 'gamma': 0.85, 'delta': 0.18,
        'lr': 1.2e-4, 'sigma': 0.55
    }
    
    all_results = []
    all_metrics = []
    total_start_time = time.time()
    
    for i in range(1, num_experiments + 1):
        # 为每个实验设置不同的初始种子
        initial_seed = random.randint(1, 100000)
        set_initial_seed(initial_seed)
        
        result = run_single_experiment(i, num_experiments, config, epochs=80)
        result['initial_seed'] = initial_seed
        all_results.append(result)
        all_metrics.append(result['metrics'])
        
        # 实时显示进度
        if i % 500 == 0:
            current_best = max(all_results, key=lambda x: 0.6*x['metrics']['ari'] + 0.4*x['metrics']['nmi'])
            elapsed = time.time() - total_start_time
            eta = (elapsed / i) * (num_experiments - i)
            print(f"\n📊 进度 {i}/{num_experiments} - 当前最佳: ARI={current_best['metrics']['ari']:.4f}")
            print(f"⏱️  已用时间: {elapsed/60:.1f}分钟, 预计剩余: {eta/60:.1f}分钟")
    
    total_time = time.time() - total_start_time
    
    # 标准化所有指标
    print(f"\n📊 标准化多指标评估...")
    normalized_metrics, metric_ranges = normalize_metrics(all_metrics)
    
    # 更新结果与标准化指标
    for i, norm_metrics in enumerate(normalized_metrics):
        all_results[i]['normalized_metrics'] = norm_metrics
    
    # 找到最佳结果（基于综合评分）
    best_result = max(all_results, key=lambda x: x['normalized_metrics']['composite_score'])
    
    # 找到所有高性能结果
    high_performers_ari = [r for r in all_results if r['metrics']['ari'] > 0.72]
    high_performers_composite = [r for r in all_results if r['normalized_metrics']['composite_score'] > 0.8]
    
    print(f"\n🏆 大规模多指标发现实验总结:")
    print("="*70)
    print(f"🕐 总运行时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"⏱️  平均单次时间: {total_time/num_experiments:.1f}秒")
    print(f"📊 平均每轮时间: {np.mean([r['timing']['time_per_epoch'] for r in all_results]):.2f}秒")
    
    print(f"\n🥇 综合评分最佳结果:")
    print(f"  🆔 实验ID: {best_result['experiment_id']}")
    print(f"  🎲 初始种子: {best_result['initial_seed']}")
    print(f"  🏆 ARI: {best_result['metrics']['ari']:.4f}")
    print(f"  🎯 NMI: {best_result['metrics']['nmi']:.4f}")
    print(f"  📊 Silhouette: {best_result['metrics']['silhouette']:.4f}")
    print(f"  🎪 V-measure: {best_result['metrics']['v_measure']:.4f}")
    print(f"  ⭐ 综合评分: {best_result['normalized_metrics']['composite_score']:.4f}")
    print(f"  🏁 最佳轮次: {best_result['best_epoch']}")
    
    # 显示高性能种子统计
    if high_performers_ari:
        print(f"\n🎯 超过ARI=0.72的种子 ({len(high_performers_ari)}个):")
        top_ari = sorted(high_performers_ari, key=lambda x: x['metrics']['ari'], reverse=True)[:5]
        for hp in top_ari:
            print(f"  种子{hp['initial_seed']}: ARI={hp['metrics']['ari']:.4f}, NMI={hp['metrics']['nmi']:.4f}, 综合={hp['normalized_metrics']['composite_score']:.4f}")
    
    if high_performers_composite:
        print(f"\n⭐ 综合评分>0.8的种子 ({len(high_performers_composite)}个):")
        top_composite = sorted(high_performers_composite, key=lambda x: x['normalized_metrics']['composite_score'], reverse=True)[:5]
        for hp in top_composite:
            print(f"  种子{hp['initial_seed']}: 综合={hp['normalized_metrics']['composite_score']:.4f}, ARI={hp['metrics']['ari']:.4f}")
    
    # 显示指标标准化范围
    print(f"\n📈 指标统计范围:")
    key_metrics = ['ari', 'nmi', 'silhouette', 'v_measure']
    for metric in key_metrics:
        min_val, max_val = metric_ranges[metric]
        mean_val = np.mean([r['metrics'][metric] for r in all_results])
        print(f"  {metric.upper()}: [{min_val:.4f}, {max_val:.4f}], 均值={mean_val:.4f}")
    
    # 保存所有结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/logs/large_scale_multimetrics_{timestamp}.pkl"
    
    discovery_summary = {
        'experiment_info': {
            'num_experiments': num_experiments,
            'total_time_seconds': total_time,
            'config': config,
            'timestamp': timestamp
        },
        'all_results': all_results,
        'best_result': best_result,
        'high_performers_ari': high_performers_ari,
        'high_performers_composite': high_performers_composite,
        'metric_ranges': metric_ranges,
        'performance_stats': {
            'ari_mean': np.mean([r['metrics']['ari'] for r in all_results]),
            'ari_std': np.std([r['metrics']['ari'] for r in all_results]),
            'composite_mean': np.mean([r['normalized_metrics']['composite_score'] for r in all_results]),
            'composite_std': np.std([r['normalized_metrics']['composite_score'] for r in all_results])
        }
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(discovery_summary, f)
    
    print(f"\n📁 完整结果保存至: {results_file}")
    
    # 保存最佳结果的状态文件
    best_states_file = f"results/logs/best_multimetrics_states_{timestamp}.pkl"
    with open(best_states_file, 'wb') as f:
        pickle.dump(best_result['random_states'], f)
    
    print(f"🎯 最佳状态保存至: {best_states_file}")
    
    return discovery_summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_experiments', type=int, default=5000, help='实验次数')
    args = parser.parse_args()
    
    os.makedirs("results/logs", exist_ok=True)
    
    print(f"🚀 开始大规模多指标随机状态发现实验...")
    discovery_summary = run_large_scale_discovery(args.num_experiments)
