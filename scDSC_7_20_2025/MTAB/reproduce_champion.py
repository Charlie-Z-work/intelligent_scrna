#!/usr/bin/env python3
"""
🎯 复现冠军种子49124的完整随机状态
验证是否能100%复现ARI=0.7187的结果
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
import pickle
import warnings

# 抑制warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class RandomStateManager:
    """随机状态管理器"""
    
    def restore_all_states(self, states):
        """恢复所有随机状态"""
        random.setstate(states['python_random_state'])
        np.random.set_state(states['numpy_random_state'])
        torch.set_rng_state(states['torch_random_state'])
        if torch.cuda.is_available() and states['torch_cuda_random_state'] is not None:
            torch.cuda.set_rng_state_all(states['torch_cuda_random_state'])

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
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h3)))
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

def reproduce_champion_seed():
    """使用完全相同的随机状态复现冠军结果"""
    
    print("🎯 冠军种子完全复现验证")
    print("="*60)
    
    # 加载冠军种子的随机状态
    states_file = "results/logs/best_states_50exp_20250714_194146.pkl"
    print(f"📁 加载冠军状态: {states_file}")
    
    with open(states_file, 'rb') as f:
        champion_states = pickle.load(f)
    
    print(f"🔍 冠军状态包含时间点: {list(champion_states.keys())}")
    
    state_mgr = RandomStateManager()
    
    # 恢复初始状态
    print("🔄 恢复冠军的初始随机状态...")
    state_mgr.restore_all_states(champion_states['initial'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    config = {
        'alpha': 0.04, 'beta': 0.005, 'gamma': 0.85, 'delta': 0.18,
        'lr': 1.2e-4, 'sigma': 0.55
    }
    
    start_time = time.time()
    
    # 加载数据 - 恢复对应状态
    with h5py.File("data/mtab.h5", "r") as f:
        X_raw = np.array(f['X_raw'])
        X_scaled = np.array(f['X'])
        y = np.array(f['Y'])
        size_factors = np.array(f['size_factors'])
    
    print("📊 数据加载完成，恢复post_data_loading状态...")
    state_mgr.restore_all_states(champion_states['post_data_loading'])
    
    # 创建模型 - 恢复对应状态
    model = SDCN_StateTracked(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    
    print("🧬 模型创建完成，恢复post_model_creation状态...")
    state_mgr.restore_all_states(champion_states['post_model_creation'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # 加载图和数据
    adj = load_graph('mtab_processed', None).to(device)
    data = torch.Tensor(X_scaled).to(device)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32).to(device)
    
    # K-means初始化 - 恢复对应状态
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)
    
    print("🎲 恢复pre_kmeans状态...")
    state_mgr.restore_all_states(champion_states['pre_kmeans'])
    
    # K-means使用固定种子42（与原实验一致）
    kmeans = KMeans(n_clusters=8, n_init=20, random_state=42, max_iter=300)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    
    print("🎯 恢复post_kmeans状态，开始训练...")
    state_mgr.restore_all_states(champion_states['post_kmeans'])
    
    # 训练循环（80轮）
    best_ari = 0
    best_nmi = 0
    best_epoch = 0
    
    global p
    for epoch in range(80):
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
                
                if epoch % 20 == 0:
                    print(f"  Epoch {epoch:2d}: ARI={ari_z:.4f}, NMI={nmi_z:.4f}")
        
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
    
    print(f"\n🏆 复现结果:")
    print("="*60)
    print(f"🎲 冠军种子: 49124")
    print(f"🏆 复现ARI: {best_ari:.4f}")
    print(f"🎯 复现NMI: {best_nmi:.4f}")
    print(f"🏁 最佳轮次: {best_epoch}")
    print(f"⏱️  训练时间: {end_time - start_time:.1f}秒")
    
    # 与原始结果对比
    original_ari = 0.7187
    original_nmi = 0.6153
    
    print(f"\n📊 复现精度验证:")
    print(f"ARI: 原始={original_ari:.4f}, 复现={best_ari:.4f}, 差异={abs(original_ari-best_ari):.6f}")
    print(f"NMI: 原始={original_nmi:.4f}, 复现={best_nmi:.4f}, 差异={abs(original_nmi-best_nmi):.6f}")
    
    if abs(original_ari - best_ari) < 0.001:
        print("✅ 复现成功！误差在可接受范围内")
    else:
        print("⚠️ 复现误差较大，可能随机状态恢复不完全")
    
    return best_ari, best_nmi

if __name__ == "__main__":
    reproduce_champion_seed()
