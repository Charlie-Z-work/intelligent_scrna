from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from evaluation import eva
from torch.utils.data import DataLoader, TensorDataset
import h5py
import scanpy as sc
from layers import ZINBLoss, MeanAct, DispAct
from GNN import GNNLayer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
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


class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, use_zinb=False):
        super(SDCN, self).__init__()
        
        self.use_zinb = use_zinb
        
        # AE to obtain internal information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        # 加载预训练模型
        pretrain_path = args.pretrain_path
        if os.path.exists(pretrain_path):
            print(f"✅ 加载预训练模型: {pretrain_path}")
            self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
        else:
            print(f"❌ 预训练模型不存在: {pretrain_path}")
            raise FileNotFoundError(f"预训练模型不存在: {pretrain_path}")

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # ZINB layers
        if self.use_zinb:
            self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
            self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
            self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
            self.zinb_loss = ZINBLoss()

        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, dec_h3 = self.ae(x)
        
        # sigma参数
        sigma = 0.5
        
        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        predict = F.softmax(h, dim=1)

        # ZINB parameters
        if self.use_zinb:
            _mean = self._dec_mean(dec_h3)
            _disp = self._dec_disp(dec_h3)
            _pi = self._dec_pi(dec_h3)
        else:
            _mean = _disp = _pi = None

        # Student t-distribution for clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, _mean, _disp, _pi


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, X_raw, sf):
    global p
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_zinb = device.type == "cuda"  # 只在GPU上启用ZINB
    
    print(f"🖥️  使用设备: {device}")
    print(f"🧬 ZINB模块: {'启用' if use_zinb else '禁用 (CPU模式)'}")
    
    model = SDCN(
        n_enc_1=args.n_enc_1,
        n_enc_2=args.n_enc_2,
        n_enc_3=args.n_enc_3,
        n_dec_1=args.n_dec_1,
        n_dec_2=args.n_dec_2,
        n_dec_3=args.n_dec_3,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1,
        use_zinb=use_zinb).to(device)
    
    print(f"📋 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器设置
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    
    # 加载图
    adj = load_graph(args.graph, args.k)
    adj = adj.to(device)
    print(f"📊 图尺寸: {adj.shape}")

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    
    print(f"📊 数据尺寸: {data.shape}")
    print(f"🏷️  标签数量: {len(np.unique(y))}")
    
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)

    # K-means初始化
    print("🎯 初始化聚类中心...")
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    print("📈 初始聚类结果:")
    eva(y, y_pred, 0)
    
    # 早停机制
    best_metrics = {'ari': 0, 'nmi': 0, 'acc': 0, 'epoch': 0}
    patience = 30
    patience_counter = 0
    
    # 更新频率
    update_interval = 10
    
    # 损失权重
    if use_zinb:
        alpha, beta, gamma, delta = 0.1, 0.01, 1.0, 0.1
        print(f"📊 损失权重: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    else:
        alpha, beta, gamma, delta = 0.1, 0.01, 1.0, 0.0
        print(f"📊 损失权重: α={alpha}, β={beta}, γ={gamma}, δ={delta} (ZINB禁用)")
    
    print(f"\n🚀 开始训练 (最多{args.epochs}轮)...")
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        if epoch % update_interval == 0:
            print(f"\n📍 Epoch {epoch}: 更新目标分布")
            with torch.no_grad():
                _, tmp_q, pred, _, _, _, _ = model(data, adj)
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                res2 = pred.data.cpu().numpy().argmax(1)  # Z
                res3 = p.data.cpu().numpy().argmax(1)  # P
                
                # 计算指标
                ari_q = adjusted_rand_score(y, res1)
                nmi_q = normalized_mutual_info_score(y, res1)
                ari_z = adjusted_rand_score(y, res2)
                nmi_z = normalized_mutual_info_score(y, res2)
                
                print(f"   Q分布: ARI={ari_q:.4f}, NMI={nmi_q:.4f}")
                print(f"   Z分布: ARI={ari_z:.4f}, NMI={nmi_z:.4f}")
                
                # 早停检查（使用Z分布的ARI）
                if ari_z > best_metrics['ari']:
                    best_metrics.update({
                        'ari': ari_z, 'nmi': nmi_z, 'epoch': epoch
                    })
                    patience_counter = 0
                    print(f"   🎯 新最佳: ARI={ari_z:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience // update_interval:
                    print(f"🛑 早停 (patience={patience_counter})!")
                    break

        # 前向传播
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(data, adj)

        # 损失计算
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean') 
        re_loss = F.mse_loss(x_bar, data)

        # ZINB损失
        if use_zinb and meanbatch is not None:
            X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
            sf_tensor = torch.tensor(sf, dtype=torch.float32).to(device)
            zinb_loss_value = model.zinb_loss(X_raw_tensor, meanbatch, dispbatch, pibatch, sf_tensor)
        else:
            zinb_loss_value = torch.tensor(0.0, device=device)

        # 总损失
        total_loss = alpha * kl_loss + beta * ce_loss + gamma * re_loss + delta * zinb_loss_value

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 每10个epoch输出损失
        if epoch % 10 == 0:
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch:3d}: total={total_loss.item():.4f}, "
                  f"kl={kl_loss.item():.4f}, ce={ce_loss.item():.4f}, "
                  f"re={re_loss.item():.4f}, zinb={zinb_loss_value.item():.4f} "
                  f"({epoch_time:.1f}s)")

    # 最终结果
    training_time = time.time() - training_start_time
    print(f"\n⏱️  训练完成，总耗时: {training_time:.1f}秒")
    
    print("\n🎯 计算最终结果...")
    with torch.no_grad():
        _, final_q, final_pred, _, _, _, _ = model(data, adj)
        final_res1 = final_q.cpu().numpy().argmax(1)
        final_res2 = final_pred.cpu().numpy().argmax(1)
        
        # 计算最终指标
        final_ari_q = adjusted_rand_score(y, final_res1)
        final_nmi_q = normalized_mutual_info_score(y, final_res1)
        final_ari_z = adjusted_rand_score(y, final_res2)
        final_nmi_z = normalized_mutual_info_score(y, final_res2)
        
        print("\n📊 最终结果总结:")
        print("="*50)
        print(f"Q分布 (聚类层): ARI={final_ari_q:.4f}, NMI={final_nmi_q:.4f}")
        print(f"Z分布 (GNN层):  ARI={final_ari_z:.4f}, NMI={final_nmi_z:.4f}")
        print(f"训练最佳:       ARI={best_metrics['ari']:.4f} (epoch {best_metrics['epoch']})")
        print(f"设备:          {device}")
        print(f"ZINB:          {'启用' if use_zinb else '禁用'}")
        print("="*50)
        
        # 与论文结果对比
        paper_results = {'ACC': 0.93, 'NMI': 0.68, 'ARI': 0.62}
        print(f"\n📈 与论文结果对比 (使用Z分布):")
        print(f"ARI: {final_ari_z:.4f} / {paper_results['ARI']:.2f} = {final_ari_z/paper_results['ARI']*100:.1f}%")
        print(f"NMI: {final_nmi_z:.4f} / {paper_results['NMI']:.2f} = {final_nmi_z/paper_results['NMI']*100:.1f}%")


if __name__ == "__main__":
    # 参数配置
    File = ['mtab_processed', "mtab_processed", 'model/mtab.pkl', "data/mtab.h5"]
    model_para = [500, 500, 2000]
    Para = [256, 1e-4, 200]  # batch_size, lr, epochs
    Cluster_para = [8, 20, 5000, 50]

    parser = argparse.ArgumentParser(
        description='scDSC训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, default=File[0])
    parser.add_argument('--graph', type=str, default=File[1])
    parser.add_argument('--pretrain_path', type=str, default=File[2])
    parser.add_argument('--n_enc_1', default=model_para[0], type=int)
    parser.add_argument('--n_enc_2', default=model_para[1], type=int)
    parser.add_argument('--n_enc_3', default=model_para[2], type=int)
    parser.add_argument('--n_dec_1', default=model_para[2], type=int)
    parser.add_argument('--n_dec_2', default=model_para[1], type=int)
    parser.add_argument('--n_dec_3', default=model_para[0], type=int)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=Para[1])
    parser.add_argument('--epochs', type=int, default=Para[2])
    parser.add_argument('--n_clusters', default=Cluster_para[0], type=int)
    parser.add_argument('--n_z', default=Cluster_para[1], type=int)
    parser.add_argument('--n_input', type=int, default=Cluster_para[2])
    parser.add_argument('--n_init', type=int, default=Cluster_para[3])

    args = parser.parse_args()
    
    print("🚀 scDSC 训练启动")
    print(f"📊 配置: {args.n_clusters}聚类, {args.n_input}基因, {args.epochs}轮训练")
    
    # 检查必要文件
    required_files = [
        'data/mtab_processed.txt', 
        'data/mtab_processed_label.txt', 
        'data/mtab.h5', 
        'graph/mtab_processed_graph.txt', 
        'model/mtab.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 缺少必要文件:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    print("✅ 所有必要文件已就绪")
    
    # 加载数据
    print("📖 加载数据...")
    dataset = load_data(args.name)
    print(f"   数据维度: {dataset.x.shape}")
    print(f"   标签数量: {len(np.unique(dataset.y))}")
    
    # 读取h5文件中的原始数据
    data_mat = h5py.File(File[3], "r")
    x_raw = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()
    
    print(f"   原始数据: {x_raw.shape}")
    
    # 计算size factors
    cell_totals = np.sum(x_raw, axis=1)
    median_total = np.median(cell_totals)
    sf = (cell_totals / median_total).astype(np.float32)
    
    print(f"   Size factors范围: [{sf.min():.3f}, {sf.max():.3f}]")
    
    # 数据验证
    assert dataset.x.shape[0] == x_raw.shape[0] == len(sf) == len(y), "样本数不一致！"
    print("✅ 数据一致性检查通过")
    
    # 启动训练
    print("\n" + "="*60)
    print("🎯 开始 scDSC 训练")
    print("="*60)
    train_sdcn(dataset, x_raw, sf)