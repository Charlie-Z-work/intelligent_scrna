import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Linear
from sklearn.cluster import KMeans
from evaluation import eva
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自编码器定义
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
        return x_bar, z

# 数据加载器
class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), torch.tensor(idx)

# 预训练函数
def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=Para[0], shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=Para[1])

    for epoch in range(Para[2]):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print(f'epoch {epoch} loss: {loss.item():.4f}')
            kmeans = KMeans(n_clusters=Cluster_para[0], n_init=Cluster_para[3]).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

    torch.save(model.state_dict(), File[0])
    print(f"✅ 模型已保存至: {File[0]}")

# 参数设置 - 修正后的配置
File = ['model/mtab.pkl', 'data/mtab_processed.txt', 'data/mtab_processed_label.txt']  # ✅ 使用预处理后的文件
Para = [1024, 1e-3, 200]
model_para = [500, 500, 2000]
Cluster_para = [8, 20, 5000, 20]  # ✅ 8个聚类，5000基因

# 读取表达矩阵和标签，使用 pandas 更稳健
x_df = pd.read_csv(File[1], sep='\t', header=None)
y_df = pd.read_csv(File[2], header=None)

x = x_df.values
y = y_df.values.squeeze()

print(f"数据维度: {x.shape}")
print(f"标签数量: {len(np.unique(y))}")

# 校验维度一致性
assert x.shape[1] == Cluster_para[2], f"❌ 输入维度 {x.shape[1]} 与配置不一致，应为 {Cluster_para[2]}"
assert x.shape[0] == y.shape[0], f"❌ 样本数不一致: 表达矩阵 {x.shape[0]} vs 标签 {y.shape[0]}"

dataset = LoadDataset(x)

# 初始化模型并训练
model = AE(
    n_enc_1=model_para[0], n_enc_2=model_para[1], n_enc_3=model_para[2],
    n_dec_1=model_para[2], n_dec_2=model_para[1], n_dec_3=model_para[0],
    n_input=Cluster_para[2], n_z=Cluster_para[1]  # ✅ 修正：n_z使用Cluster_para[1]=20
).to(device)

pretrain_ae(model, dataset, y)