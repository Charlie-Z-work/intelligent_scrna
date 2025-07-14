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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

class SDCN_Analysis(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=True, pretrain_path='model/mtab.pkl'):
        super(SDCN_Analysis, self).__init__()
        self.use_zinb = use_zinb
        
        self.ae = AE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z)
        
        if os.path.exists(pretrain_path):
            self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

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

def analyze_best_config():
    """分析最优配置的训练过程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    # 最优配置
    config = {
        'alpha': 0.04, 'beta': 0.004, 'gamma': 0.75, 'delta': 0.15,
        'lr': 8e-5, 'sigma': 0.5, 'epochs': 250, 'use_scheduler': False
    }
    
    print(f"🔍 分析最优配置:")
    print(f"   α={config['alpha']}, β={config['beta']}, γ={config['gamma']}, δ={config['delta']}")
    print(f"   lr={config['lr']}, σ={config['sigma']}, epochs={config['epochs']}")
    
    # 加载数据
    with h5py.File("data/mtab.h5", "r") as f:
        X_raw = np.array(f['X_raw'])
        X_scaled = np.array(f['X'])
        y = np.array(f['Y'])
        size_factors = np.array(f['size_factors'])
    
    print(f"📊 数据: {X_scaled.shape}, 类别: {len(np.unique(y))}")
    
    # 创建模型
    model = SDCN_Analysis(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # 加载图和数据
    adj = load_graph('mtab_processed', None).to(device)
    data = torch.Tensor(X_scaled).to(device)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32).to(device)
    
    # K-means初始化
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)
    kmeans = KMeans(n_clusters=8, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    ari_init = adjusted_rand_score(y, y_pred)
    nmi_init = normalized_mutual_info_score(y, y_pred)
    print(f"📈 初始: ARI={ari_init:.4f}, NMI={nmi_init:.4f}")
    
    # 训练记录
    epochs_log = []
    ari_q_log = []
    nmi_q_log = []
    ari_z_log = []
    nmi_z_log = []
    loss_log = []
    
    best_ari = 0
    best_nmi = 0
    best_ari_epoch = 0
    best_nmi_epoch = 0
    best_ari_nmi = 0  # ARI最高时的NMI
    best_nmi_ari = 0  # NMI最高时的ARI
    
    print(f"\n🚀 开始详细训练分析...")
    start_time = time.time()
    
    global p
    for epoch in range(config['epochs']):
        # 每5轮评估一次
        if epoch % 5 == 0:
            with torch.no_grad():
                _, tmp_q, pred, _, _, _, _ = model(data, adj, sigma=config['sigma'])
                p = target_distribution(tmp_q.data)
                
                res1 = tmp_q.cpu().numpy().argmax(1)
                res2 = pred.data.cpu().numpy().argmax(1)
                
                ari_q = adjusted_rand_score(y, res1)
                ari_z = adjusted_rand_score(y, res2)
                nmi_q = normalized_mutual_info_score(y, res1)
                nmi_z = normalized_mutual_info_score(y, res2)
                
                # 记录数据
                epochs_log.append(epoch)
                ari_q_log.append(ari_q)
                nmi_q_log.append(nmi_q)
                ari_z_log.append(ari_z)
                nmi_z_log.append(nmi_z)
                
                # 更新最佳结果
                if ari_z > best_ari:
                    best_ari = ari_z
                    best_ari_epoch = epoch
                    best_ari_nmi = nmi_z
                
                if nmi_z > best_nmi:
                    best_nmi = nmi_z
                    best_nmi_epoch = epoch
                    best_nmi_ari = ari_z
                
                print(f"Epoch {epoch:3d}: Q_ARI={ari_q:.4f}, Q_NMI={nmi_q:.4f}, Z_ARI={ari_z:.4f}, Z_NMI={nmi_z:.4f}")
        
        # 前向传播
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data, adj, sigma=config['sigma'])
        
        # 损失计算
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        
        # ZINB损失
        try:
            zinb_loss_value = model.zinb_loss(X_raw_tensor, meanbatch, dispbatch, pibatch, sf_tensor)
            if torch.isnan(zinb_loss_value) or torch.isinf(zinb_loss_value):
                zinb_loss_value = torch.tensor(0.0, device=device)
        except:
            zinb_loss_value = torch.tensor(0.0, device=device)
        
        total_loss = (config['alpha'] * kl_loss + config['beta'] * ce_loss + 
                     config['gamma'] * re_loss + config['delta'] * zinb_loss_value)
        
        loss_log.append(total_loss.item())
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # 训练完成分析
    training_time = time.time() - start_time
    print(f"\n⏱️  训练完成: {training_time:.1f}秒")
    
    print(f"\n🎯 最佳结果分析:")
    print("="*60)
    print(f"🏆 ARI最高点:")
    print(f"   Epoch {best_ari_epoch}: ARI={best_ari:.4f}, NMI={best_ari_nmi:.4f}")
    print(f"   论文对比: ARI={best_ari/0.62*100:.1f}%, NMI={best_ari_nmi/0.68*100:.1f}%")
    
    print(f"\n🎯 NMI最高点:")
    print(f"   Epoch {best_nmi_epoch}: ARI={best_nmi_ari:.4f}, NMI={best_nmi:.4f}")
    print(f"   论文对比: ARI={best_nmi_ari/0.62*100:.1f}%, NMI={best_nmi/0.68*100:.1f}%")
    
    print(f"\n📊 综合评分:")
    score_at_best_ari = 0.6 * best_ari + 0.4 * best_ari_nmi
    score_at_best_nmi = 0.6 * best_nmi_ari + 0.4 * best_nmi
    print(f"   ARI最高时Score: {score_at_best_ari:.4f}")
    print(f"   NMI最高时Score: {score_at_best_nmi:.4f}")
    
    # 保存详细日志
    log_file = f"results/logs/best_config_analysis_{int(time.time())}.txt"
    with open(log_file, 'w') as f:
        f.write("🔍 最优配置详细训练分析\n")
        f.write("="*50 + "\n")
        f.write(f"配置: {config}\n\n")
        f.write(f"ARI最高点: Epoch {best_ari_epoch}, ARI={best_ari:.4f}, NMI={best_ari_nmi:.4f}\n")
        f.write(f"NMI最高点: Epoch {best_nmi_epoch}, ARI={best_nmi_ari:.4f}, NMI={best_nmi:.4f}\n")
        f.write(f"训练时长: {training_time:.1f}秒\n\n")
        f.write("Epoch,ARI_Q,NMI_Q,ARI_Z,NMI_Z,Loss\n")
        for i, epoch in enumerate(epochs_log):
            f.write(f"{epoch},{ari_q_log[i]:.4f},{nmi_q_log[i]:.4f},{ari_z_log[i]:.4f},{nmi_z_log[i]:.4f},{loss_log[epoch]:.4f}\n")
    
    print(f"\n📁 详细日志保存至: {log_file}")
    
    # 创建训练曲线图
    plt.figure(figsize=(15, 10))
    
    # ARI曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs_log, ari_z_log, 'b-', label='ARI (Z)', linewidth=2)
    plt.axvline(x=best_ari_epoch, color='red', linestyle='--', alpha=0.7, label=f'Max ARI @ Epoch {best_ari_epoch}')
    plt.axhline(y=best_ari, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('ARI')
    plt.title('ARI Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # NMI曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs_log, nmi_z_log, 'g-', label='NMI (Z)', linewidth=2)
    plt.axvline(x=best_nmi_epoch, color='red', linestyle='--', alpha=0.7, label=f'Max NMI @ Epoch {best_nmi_epoch}')
    plt.axhline(y=best_nmi, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('NMI')
    plt.title('NMI Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ARI vs NMI散点图
    plt.subplot(2, 2, 3)
    plt.scatter(ari_z_log, nmi_z_log, c=epochs_log, cmap='viridis', alpha=0.7)
    plt.scatter(best_ari, best_ari_nmi, color='red', s=100, marker='*', label=f'Max ARI: {best_ari:.3f}')
    plt.scatter(best_nmi_ari, best_nmi, color='blue', s=100, marker='*', label=f'Max NMI: {best_nmi:.3f}')
    plt.xlabel('ARI')
    plt.ylabel('NMI')
    plt.title('ARI vs NMI Trajectory')
    plt.colorbar(label='Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失曲线
    plt.subplot(2, 2, 4)
    plt.plot(loss_log, 'r-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = f"results/logs/best_config_curves_{int(time.time())}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 训练曲线保存至: {plot_file}")
    
    return best_ari, best_ari_epoch, best_ari_nmi, best_nmi, best_nmi_epoch, best_nmi_ari

if __name__ == "__main__":
    os.makedirs("results/logs", exist_ok=True)
    analyze_best_config()
