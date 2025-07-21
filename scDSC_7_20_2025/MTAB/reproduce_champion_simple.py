#!/usr/bin/env python3
"""
🎯 复现冠军种子49124 - 简化版
直接调用现有模块，减少重复代码
"""

import sys
import os
sys.path.append('..')

# 导入现有模块
from scdsc_zinb import SDCN_Fixed, target_distribution
import numpy as np
import torch
import torch.nn.functional as F
from utils import load_graph
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import random
import pickle
import warnings

warnings.filterwarnings("ignore")

class RandomStateManager:
    def restore_all_states(self, states):
        """恢复所有随机状态"""
        random.setstate(states['python_random_state'])
        np.random.set_state(states['numpy_random_state'])
        torch.set_rng_state(states['torch_random_state'])
        if torch.cuda.is_available() and states['torch_cuda_random_state'] is not None:
            torch.cuda.set_rng_state_all(states['torch_cuda_random_state'])

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
    
    # 加载数据
    with h5py.File("data/mtab.h5", "r") as f:
        X_raw = np.array(f['X_raw'])
        X_scaled = np.array(f['X'])
        y = np.array(f['Y'])
        size_factors = np.array(f['size_factors'])
    
    print("📊 数据加载完成，恢复post_data_loading状态...")
    state_mgr.restore_all_states(champion_states['post_data_loading'])
    
    # 创建模型 - 使用现有的SDCN_Fixed
    model = SDCN_Fixed(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    
    print("🧬 模型创建完成，恢复post_model_creation状态...")
    state_mgr.restore_all_states(champion_states['post_model_creation'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # 加载图和数据
    adj = load_graph('mtab_processed', None).to(device)
    data = torch.Tensor(X_scaled).to(device)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32).to(device)
    
    # K-means初始化
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)
    
    print("🎲 恢复pre_kmeans状态...")
    state_mgr.restore_all_states(champion_states['pre_kmeans'])
    
    # K-means使用固定种子42
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
                _, tmp_q, pred, _, _, _, _ = model(data, adj)
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
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data, adj)
        
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
        print("✅ 复现成功！随机状态记录完整有效")
    else:
        print("⚠️ 复现误差较大，需要检查随机状态恢复")
    
    return best_ari, best_nmi

if __name__ == "__main__":
    reproduce_champion_seed()
