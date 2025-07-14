import sys
sys.path.append('..')

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import load_data, load_graph
from evaluation import eva
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from layers import ZINBLoss, MeanAct, DispAct
from GNN import GNNLayer

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# 复制模型定义 (保持原来的代码不变)
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.BN4 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.BN5 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.BN6 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

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

class SDCN_Track(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=True, pretrain_path='model/mtab.pkl'):
        super(SDCN_Track, self).__init__()
        self.use_zinb = use_zinb
        self.ae = AE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z)
        
        if os.path.exists(pretrain_path):
            self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
            print(f"✅ 加载预训练模型: {pretrain_path}")

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

def track_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 设备: {device}")
    print("🎯 追踪最佳配置训练过程")
    print("📋 参数: α=0.04, β=0.004, γ=0.75, δ=0.15, lr=8e-05, σ=0.5")
    print("🎲 随机种子: 42 (固定)")
    
    # 加载数据
    with h5py.File("data/mtab.h5", "r") as f:
        X_raw = np.array(f['X_raw'])
        X_scaled = np.array(f['X'])
        y = np.array(f['Y'])
        size_factors = np.array(f['size_factors'])
    
    print(f"📊 数据检查:")
    print(f"   X_raw: {X_raw.shape}, 范围: [{X_raw.min():.0f}, {X_raw.max():.0f}]")
    print(f"   X_scaled: {X_scaled.shape}, 范围: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"   Size factors: [{size_factors.min():.3f}, {size_factors.max():.3f}]")
    
    # 创建模型
    model = SDCN_Track(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-05, weight_decay=1e-5)
    
    # 准备数据
    adj = load_graph('mtab_processed', None).to(device)
    data = torch.Tensor(X_scaled).to(device)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32).to(device)
    
    # K-means初始化 (固定随机种子)
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)
    kmeans = KMeans(n_clusters=8, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    print(f"📈 初始聚类: ARI={adjusted_rand_score(y, y_pred):.4f}")
    
    # 训练追踪
    best_ari = 0
    best_nmi = 0
    best_ari_epoch = 0
    best_nmi_epoch = 0
    best_ari_nmi = 0
    best_nmi_ari = 0
    
    alpha, beta, gamma, delta = 0.04, 0.004, 0.75, 0.15
    sigma = 0.5
    
    print("🚀 开始训练...")
    print("Epoch | ARI    | NMI    | ARI_Best | NMI_Best | Total_Loss")
    print("-" * 60)
    
    global p
    for epoch in range(250):
        if epoch % 10 == 0:
            with torch.no_grad():
                _, tmp_q, pred, _, _, _, _ = model(data, adj, sigma)
                p = target_distribution(tmp_q.data)
                
                res1 = tmp_q.cpu().numpy().argmax(1)
                res2 = pred.data.cpu().numpy().argmax(1)
                
                ari_q = adjusted_rand_score(y, res1)
                ari_z = adjusted_rand_score(y, res2)
                nmi_z = normalized_mutual_info_score(y, res2)
                
                # 更新最佳记录
                if ari_z > best_ari:
                    best_ari = ari_z
                    best_ari_epoch = epoch
                    best_ari_nmi = nmi_z
                
                if nmi_z > best_nmi:
                    best_nmi = nmi_z
                    best_nmi_epoch = epoch
                    best_nmi_ari = ari_z
                
                print(f"{epoch:5d} | {ari_z:.4f} | {nmi_z:.4f} | {best_ari:.4f}  | {best_nmi:.4f}  |", end="")

        # 前向传播
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data, adj, sigma)
        
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
        
        total_loss = alpha * kl_loss + beta * ce_loss + gamma * re_loss + delta * zinb_loss_value
        
        if epoch % 10 == 0:
            print(f" {total_loss.item():8.4f}")
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print("\n" + "=" * 60)
    print("🎯 训练完成！详细结果:")
    print(f"📈 ARI最高: {best_ari:.4f} (Epoch {best_ari_epoch}), 此时NMI: {best_ari_nmi:.4f}")
    print(f"📊 NMI最高: {best_nmi:.4f} (Epoch {best_nmi_epoch}), 此时ARI: {best_nmi_ari:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    track_training()
