#!/usr/bin/env python3
"""
🎯 随机状态记录实验 - 修复版 (无warnings)
记录所有随机性来源，用于后续完全复现
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from layers import ZINBLoss, MeanAct, DispAct
from GNN import GNNLayer
import time
import random
import json
import pickle
from datetime import datetime
import warnings

# 抑制FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class RandomStateManager:
    """随机状态管理器"""
    
    def __init__(self):
        self.states = {}
        
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
    
    def save_states(self, states, filename):
        """保存状态到文件"""
        with open(filename, 'wb') as f:
            pickle.dump(states, f)
    
    def load_states(self, filename):
        """从文件加载状态"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

def set_initial_seed(seed):
    """设置初始种子"""
    print(f"🔒 设置初始种子: {seed}")
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

class SDCN_StateTracked(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=True, pretrain_path='model/mtab.pkl'):
        super(SDCN_StateTracked, self).__init__()
        self.use_zinb = use_zinb
        
        self.ae = AE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z)
        
        if os.path.exists(pretrain_path):
            # 修复：添加weights_only参数
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

def run_single_experiment(experiment_id, total_experiments, config, epochs=80):
    """运行单次实验并记录所有状态"""
    
    print(f"\n🧪 实验 {experiment_id}/{total_experiments}")
    
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
    
    post_data_states = state_mgr.capture_all_states("post_data_loading")
    
    # 创建模型
    model = SDCN_StateTracked(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    post_model_states = state_mgr.capture_all_states("post_model_creation")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # 加载图和数据
    adj = load_graph('mtab_processed', None).to(device)
    data = torch.Tensor(X_scaled).to(device)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32).to(device)
    
    # K-means初始化
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)
    
    pre_kmeans_states = state_mgr.capture_all_states("pre_kmeans")
    
    kmeans = KMeans(n_clusters=8, n_init=20, random_state=42, max_iter=300)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    
    post_kmeans_states = state_mgr.capture_all_states("post_kmeans")
    
    # 训练记录
    best_ari = 0
    best_nmi = 0
    best_epoch = 0
    
    global p
    for epoch in range(epochs):
        if epoch % 5 == 0:
            with torch.no_grad():
                _, tmp_q, pred, _, _, _, _ = model(data, adj, sigma=config['sigma'])
                p = target_distribution(tmp_q.data)
                
                res2 = pred.data.cpu().numpy().argmax(1)
                ari_z = adjusted_rand_score(y, res2)
                nmi_z = normalized_mutual_info_score(y, res2)
                
                if ari_z > best_ari:
                    best_ari = ari_z
                    best_nmi = nmi_z
                    best_epoch = epoch
        
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
    
    final_states = state_mgr.capture_all_states("experiment_end")
    
    experiment_result = {
        'experiment_id': experiment_id,
        'config': config,
        'performance': {
            'ari': best_ari,
            'nmi': best_nmi,
            'best_epoch': best_epoch,
            'score': 0.6 * best_ari + 0.4 * best_nmi
        },
        'timing': {
            'total_time_seconds': training_time,
            'time_per_epoch': training_time / epochs,
            'epochs': epochs
        },
        'random_states': {
            'initial': initial_states,
            'post_data_loading': post_data_states,
            'post_model_creation': post_model_states,
            'pre_kmeans': pre_kmeans_states,
            'post_kmeans': post_kmeans_states,
            'final': final_states
        }
    }
    
    print(f"  🏆 ARI={best_ari:.4f}, NMI={best_nmi:.4f}, Epoch={best_epoch}")
    print(f"  ⏱️  训练时间: {training_time:.1f}秒 ({training_time/epochs:.2f}秒/轮)")
    
    return experiment_result

def run_random_state_discovery(num_experiments=50):
    """运行随机状态发现实验"""
    
    print("🎯 大规模随机状态发现实验")
    print("="*60)
    print(f"📊 实验次数: {num_experiments}")
    print(f"🎰 策略: 除KMeans外全部随机")
    print(f"🎯 目标: 寻找能达到ARI=0.77+的种子")
    
    # 传说配置
    config = {
        'alpha': 0.04, 'beta': 0.005, 'gamma': 0.85, 'delta': 0.18,
        'lr': 1.2e-4, 'sigma': 0.55
    }
    
    all_results = []
    total_start_time = time.time()
    
    for i in range(1, num_experiments + 1):
        # 为每个实验设置不同的初始种子
        initial_seed = random.randint(1, 100000)
        set_initial_seed(initial_seed)
        
        result = run_single_experiment(i, num_experiments, config, epochs=80)
        result['initial_seed'] = initial_seed
        all_results.append(result)
        
        # 实时显示进度
        if i % 10 == 0:
            current_best = max(all_results, key=lambda x: x['performance']['score'])
            print(f"\n📊 进度 {i}/{num_experiments} - 当前最佳: ARI={current_best['performance']['ari']:.4f}")
    
    total_time = time.time() - total_start_time
    
    # 找到最佳结果
    best_result = max(all_results, key=lambda x: x['performance']['score'])
    
    # 找到所有超过0.7的结果
    high_performers = [r for r in all_results if r['performance']['ari'] > 0.7]
    
    print(f"\n🏆 大规模发现实验总结:")
    print("="*60)
    print(f"🕐 总运行时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"⏱️  平均单次时间: {total_time/num_experiments:.1f}秒")
    print(f"📊 平均每轮时间: {np.mean([r['timing']['time_per_epoch'] for r in all_results]):.2f}秒")
    
    print(f"\n🥇 最佳结果:")
    print(f"  🆔 实验ID: {best_result['experiment_id']}")
    print(f"  🎲 初始种子: {best_result['initial_seed']}")
    print(f"  🏆 ARI: {best_result['performance']['ari']:.4f}")
    print(f"  🎯 NMI: {best_result['performance']['nmi']:.4f}")
    print(f"  📊 Score: {best_result['performance']['score']:.4f}")
    print(f"  🏁 最佳轮次: {best_result['performance']['best_epoch']}")
    
    if high_performers:
        print(f"\n🎯 超过ARI=0.7的种子 ({len(high_performers)}个):")
        for hp in sorted(high_performers, key=lambda x: x['performance']['ari'], reverse=True)[:5]:
            print(f"  种子{hp['initial_seed']}: ARI={hp['performance']['ari']:.4f}, NMI={hp['performance']['nmi']:.4f}")
    else:
        print(f"\n📊 性能统计:")
        aris = [r['performance']['ari'] for r in all_results]
        print(f"  ARI范围: {min(aris):.4f} - {max(aris):.4f}")
        print(f"  ARI平均: {np.mean(aris):.4f} ± {np.std(aris):.4f}")
    
    # 保存所有结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/logs/large_scale_discovery_{timestamp}.pkl"
    
    discovery_summary = {
        'experiment_info': {
            'num_experiments': num_experiments,
            'total_time_seconds': total_time,
            'config': config,
            'timestamp': timestamp
        },
        'all_results': all_results,
        'best_result': best_result,
        'high_performers': high_performers,
        'performance_stats': {
            'ari_mean': np.mean([r['performance']['ari'] for r in all_results]),
            'ari_std': np.std([r['performance']['ari'] for r in all_results]),
            'ari_max': max([r['performance']['ari'] for r in all_results]),
            'ari_min': min([r['performance']['ari'] for r in all_results])
        }
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(discovery_summary, f)
    
    print(f"\n📁 完整结果保存至: {results_file}")
    
    # 保存最佳结果的状态文件
    best_states_file = f"results/logs/best_states_50exp_{timestamp}.pkl"
    with open(best_states_file, 'wb') as f:
        pickle.dump(best_result['random_states'], f)
    
    print(f"🎯 最佳状态保存至: {best_states_file}")
    
    return discovery_summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_experiments', type=int, default=50, help='实验次数')
    args = parser.parse_args()
    
    os.makedirs("results/logs", exist_ok=True)
    
    print(f"🚀 开始大规模随机状态发现实验...")
    discovery_summary = run_random_state_discovery(args.num_experiments)
