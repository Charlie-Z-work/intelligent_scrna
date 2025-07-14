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
import datetime

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

class SDCN_Enhanced(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=True, pretrain_path='model/mtab.pkl'):
        super(SDCN_Enhanced, self).__init__()
        self.use_zinb = use_zinb
        
        self.ae = AE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z)
        
        if os.path.exists(pretrain_path):
            print(f"✅ 加载预训练模型: {pretrain_path}")
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

def train_config(config_name, config):
    print(f"\n🧪 开始配置: {config_name}")
    print(f"🔧 参数: {config}")
    
    # 创建日志文件
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    log_file = f"results/logs/{config_name}_{job_id}_{timestamp}.log"
    result_file = f"results/logs/result_{config_name}_{job_id}_{timestamp}.txt"
    
    def log_and_save(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_and_save(f"🖥️  设备: {device}")
    
    # 加载数据
    try:
        with h5py.File("data/mtab.h5", "r") as f:
            X_raw = np.array(f['X_raw'])
            X_scaled = np.array(f['X'])
            y = np.array(f['Y'])
            size_factors = np.array(f['size_factors'])
        log_and_save(f"📊 数据: {X_scaled.shape}, 类别: {len(np.unique(y))}")
    except Exception as e:
        log_and_save(f"❌ 数据加载失败: {e}")
        return 0, 0
    
    # 创建模型
    model = SDCN_Enhanced(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # 学习率调度器
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
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
    log_and_save(f"📈 初始ARI: {ari_init:.4f}")
    
    # 训练记录
    best_ari = 0
    best_nmi = 0
    patience_counter = 0
    
    log_and_save("🚀 训练开始...")
    training_start = time.time()
    
    global p
    for epoch in range(config['epochs']):
        # 更新目标分布
        if epoch % 10 == 0:
            with torch.no_grad():
                _, tmp_q, pred, _, _, _, _ = model(data, adj, sigma=config.get('sigma', 0.5))
                p = target_distribution(tmp_q.data)
                
                res1 = tmp_q.cpu().numpy().argmax(1)
                res2 = pred.data.cpu().numpy().argmax(1)
                
                ari_q = adjusted_rand_score(y, res1)
                ari_z = adjusted_rand_score(y, res2)
                nmi_z = normalized_mutual_info_score(y, res2)
                
                log_and_save(f"Epoch {epoch:3d}: Q_ARI={ari_q:.4f}, Z_ARI={ari_z:.4f}, Z_NMI={nmi_z:.4f}")
                
                # 更新最佳结果
                improved = False
                if ari_z > best_ari:
                    best_ari = ari_z
                    improved = True
                    log_and_save(f"   🎯 新最佳ARI: {ari_z:.4f}")
                
                if nmi_z > best_nmi:
                    best_nmi = nmi_z
                    if not improved:
                        log_and_save(f"   🎯 新最佳NMI: {nmi_z:.4f}")
                
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= 5:  # 50 epochs without improvement
                    log_and_save(f"🛑 早停: patience={patience_counter}")
                    break
        
        # 前向传播
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data, adj, sigma=config.get('sigma', 0.5))
        
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
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if config.get('use_scheduler', False):
            scheduler.step()
        
        # 记录损失
        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            log_and_save(f"Epoch {epoch:3d}: loss={total_loss.item():.4f}, "
                        f"kl={kl_loss.item():.4f}, ce={ce_loss.item():.4f}, "
                        f"re={re_loss.item():.4f}, zinb={zinb_loss_value.item():.4f}, lr={lr:.2e}")
    
    # 训练完成
    training_time = time.time() - training_start
    log_and_save(f"⏱️  训练完成: {training_time:.1f}秒")
    log_and_save(f"🏆 最佳结果: ARI={best_ari:.4f}, NMI={best_nmi:.4f}")
    log_and_save(f"📈 论文对比: ARI={best_ari/0.62*100:.1f}%, NMI={best_nmi/0.68*100:.1f}%")
    
    # 保存结果文件
    with open(result_file, 'w') as f:
        f.write(f"🎯 scDSC训练结果 - {config_name}\n")
        f.write(f"配置: {config}\n")
        f.write(f"最佳ARI: {best_ari:.4f}\n")
        f.write(f"最佳NMI: {best_nmi:.4f}\n")
        f.write(f"训练时长: {training_time:.1f}秒\n")
        f.write(f"论文对比: ARI={best_ari/0.62*100:.1f}%, NMI={best_nmi/0.68*100:.1f}%\n")
        f.write(f"作业ID: {os.environ.get('SLURM_JOB_ID', 'local')}\n")
        f.write(f"节点: {os.environ.get('SLURMD_NODENAME', 'unknown')}\n")
    
    print(f"✅ {config_name} 完成: ARI={best_ari:.4f}, NMI={best_nmi:.4f}")
    return best_ari, best_nmi

# 主程序
if __name__ == "__main__":
    # 确保results/logs目录存在
    os.makedirs("results/logs", exist_ok=True)
    
    # 配置字典
    configs = {
        'baseline': {
            'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0, 'delta': 0.1,
            'lr': 1e-4, 'sigma': 0.5, 'epochs': 100
        },
        'zinb_enhanced': {
            'alpha': 0.08, 'beta': 0.02, 'gamma': 1.0, 'delta': 0.15,
            'lr': 1e-4, 'sigma': 0.5, 'epochs': 100
        },
        'lr_tuned': {
            'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0, 'delta': 0.1,
            'lr': 8e-5, 'sigma': 0.5, 'epochs': 100, 'use_scheduler': True
        },
        'sigma_tuned': {
            'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0, 'delta': 0.1,
            'lr': 1e-4, 'sigma': 0.6, 'epochs': 100
        },
        'aggressive': {
            'alpha': 0.12, 'beta': 0.03, 'gamma': 0.8, 'delta': 0.18,
            'lr': 1.2e-4, 'sigma': 0.55, 'epochs': 100, 'use_scheduler': True
        }
    }
    
    # 从环境变量获取配置
    config_name = os.environ.get('CONFIG_NAME', 'baseline')
    
    if config_name in configs:
        print(f"🔧 运行配置: {config_name}")
        config = configs[config_name]
        ari, nmi = train_config(config_name, config)
    else:
        print(f"❌ 未知配置: {config_name}")
        print(f"可用配置: {list(configs.keys())}")
