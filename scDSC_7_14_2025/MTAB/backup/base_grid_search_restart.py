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
import itertools

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

class SDCN_Massive(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=True, pretrain_path='model/mtab.pkl'):
        super(SDCN_Massive, self).__init__()
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

def quick_train_config(config, config_id, task_id):
    """快速训练单个配置"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 全局数据（已加载）
    global X_raw_tensor, data_global, y_global, sf_tensor, adj_global
    
    # 创建模型
    model = SDCN_Massive(500, 500, 2000, 2000, 500, 500, 5000, 20, 8, v=1, use_zinb=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # 学习率调度器
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    # K-means初始化
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data_global)
    kmeans = KMeans(n_clusters=8, n_init=5, random_state=42)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    # 训练记录
    best_ari = 0
    best_nmi = 0
    patience_counter = 0
    
    start_time = time.time()
    
    global p
    for epoch in range(config['epochs']):
        # 更新目标分布 (每20轮)
        if epoch % 20 == 0:
            with torch.no_grad():
                _, tmp_q, pred, _, _, _, _ = model(data_global, adj_global, sigma=config.get('sigma', 0.5))
                p = target_distribution(tmp_q.data)
                
                res2 = pred.data.cpu().numpy().argmax(1)
                ari_z = adjusted_rand_score(y_global, res2)
                nmi_z = normalized_mutual_info_score(y_global, res2)
                
                # 更新最佳结果
                if ari_z > best_ari:
                    best_ari = ari_z
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if nmi_z > best_nmi:
                    best_nmi = nmi_z
                
                # 早停 (更宽松)
                if patience_counter >= 8:  # 160 epochs without improvement
                    break
        
        # 前向传播
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data_global, adj_global, sigma=config.get('sigma', 0.5))
        
        # 损失计算
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data_global)
        
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
    
    training_time = time.time() - start_time
    
    return {
        'task_id': task_id,
        'config_id': config_id,
        'config': config,
        'ari': best_ari,
        'nmi': best_nmi,
        'time': training_time,
        'score': 0.6 * best_ari + 0.4 * best_nmi
    }

def run_massive_grid_search_restart(task_id, start_idx, end_idx):
    """运行重启的大规模网格搜索"""
    
    # 全局数据加载
    global X_raw_tensor, data_global, y_global, sf_tensor, adj_global
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 重启任务{task_id} 设备: {device}")
    
    # 加载数据 (只加载一次)
    with h5py.File("data/mtab.h5", "r") as f:
        X_raw = np.array(f['X_raw'])
        X_scaled = np.array(f['X'])
        y_global = np.array(f['Y'])
        size_factors = np.array(f['size_factors'])
    
    print(f"📊 任务{task_id} 数据: {X_scaled.shape}")
    
    # 转换为张量 (全局)
    adj_global = load_graph('mtab_processed', None).to(device)
    data_global = torch.Tensor(X_scaled).to(device)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32).to(device)
    
    # 完整参数网格
    param_grid = {
        'alpha': [0.03, 0.04, 0.05, 0.06, 0.07],
        'beta': [0.003, 0.004, 0.005, 0.006, 0.008],
        'gamma': [0.75, 0.8, 0.85],
        'delta': [0.08, 0.1, 0.12, 0.15, 0.18, 0.2],
        'lr': [8e-5, 9e-5, 1e-4, 1.1e-4, 1.2e-4],
        'sigma': [0.4, 0.45, 0.5, 0.55, 0.6],
        'epochs': [250],
        'use_scheduler': [False, True]
    }
    
    # 生成所有组合
    keys = param_grid.keys()
    values = param_grid.values()
    all_combinations = list(itertools.product(*values))
    
    # 选择当前任务的配置子集
    task_combinations = all_combinations[start_idx:end_idx]
    
    print(f"🔍 重启任务{task_id} 搜索范围: 配置{start_idx+1}-{end_idx}")
    print(f"📋 本任务配置数: {len(task_combinations)}")
    
    # 创建结果文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    results_file = f"results/logs/restart_task{task_id}_{job_id}_{timestamp}.csv"
    
    # CSV表头
    with open(results_file, 'w') as f:
        f.write("task_id,config_id,alpha,beta,gamma,delta,lr,sigma,epochs,use_scheduler,ari,nmi,score,time\n")
    
    print(f"📁 结果文件: {results_file}")
    print(f"\n🚀 重启任务{task_id} 开始搜索...")
    
    start_time = time.time()
    results = []
    
    for i, combination in enumerate(task_combinations):
        config = dict(zip(keys, combination))
        global_config_id = start_idx + i + 1
        
        try:
            result = quick_train_config(config, global_config_id, task_id)
            results.append(result)
            
            # 实时保存结果
            with open(results_file, 'a') as f:
                f.write(f"{task_id},{global_config_id},{config['alpha']},{config['beta']},{config['gamma']},"
                       f"{config['delta']},{config['lr']},{config['sigma']},{config['epochs']},"
                       f"{config['use_scheduler']},{result['ari']:.4f},{result['nmi']:.4f},"
                       f"{result['score']:.4f},{result['time']:.2f}\n")
            
            # 进度显示
            if (i+1) % 100 == 0 or result['ari'] > 0.75 or result['nmi'] > 0.68:
                elapsed = time.time() - start_time
                remaining = (len(task_combinations) - i - 1) * (elapsed / (i + 1))
                print(f"重启任务{task_id} [{i+1:4d}/{len(task_combinations)}] "
                      f"ARI={result['ari']:.4f}, NMI={result['nmi']:.4f}, "
                      f"Score={result['score']:.4f} (剩余:{remaining/60:.1f}min)")
                
                # 发现超级结果时详细显示
                if result['ari'] > 0.75 or result['nmi'] > 0.68:
                    print(f"   🎯🔥 重启任务{task_id} 发现突破配置: {config}")
        
        except Exception as e:
            print(f"❌ 重启任务{task_id} 配置 {global_config_id} 失败: {e}")
            continue
    
    # 总结
    total_time = time.time() - start_time
    print(f"\n🏁 重启任务{task_id} 完成！用时: {total_time/60:.1f}分钟")
    
    if results:
        # 按评分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 重启任务{task_id} Top 5 配置:")
        print("="*80)
        for i, result in enumerate(results[:5]):
            config = result['config']
            print(f"{i+1}. ARI={result['ari']:.4f}, NMI={result['nmi']:.4f}, Score={result['score']:.4f}")
            print(f"   α={config['alpha']}, β={config['beta']}, γ={config['gamma']}, δ={config['delta']}")
            print(f"   lr={config['lr']}, σ={config['sigma']}, epochs={config['epochs']}, scheduler={config['use_scheduler']}")

# 从环境变量获取任务信息
if __name__ == "__main__":
    os.makedirs("results/logs", exist_ok=True)
    
    task_id = int(os.environ.get('TASK_ID', '1'))
    
    # 从断点继续的配置范围
    if task_id == 1:
        start_idx, end_idx = 1832, 3750  # 继续剩余的1918个配置
    elif task_id == 2:
        start_idx, end_idx = 5588, 7500  # 继续剩余的1912个配置
    elif task_id == 3:
        start_idx, end_idx = 9337, 11250  # 继续剩余的1913个配置
    else:
        print(f"❌ 无效任务ID: {task_id}")
        exit(1)
    
    print(f"🔄 重启任务{task_id} - 从断点继续")
    run_massive_grid_search_restart(task_id, start_idx, end_idx)
