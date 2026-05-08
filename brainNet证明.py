import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# ===================================================
# 参数设置
# ===================================================
NPZ_PATH = "D:/DEC/subs100_90_1200.npz"  # 多被试 npz 文件
AAL_NODE_PATH = "D:/DEC/AAL90.node"       # AAL90 节点文件
SAVE_NODE_PATH = "D:/DEC/ROI_cluster_AAL90.node"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
win_size = 100
stride = 10
latent_dim = 32
batch_size = 256
ae_epochs = 30
n_clusters = 5  # ROI聚类类别数
# ===================================================
# 辅助函数
# ===================================================
def sliding_windows(ts, win=100, stride=10):
    n_roi, T = ts.shape
    windows = [ts[:, i:i + win] for i in range(0, T - win + 1, stride)]
    return np.stack(windows, axis=0)
def fc_features(ts, win=100, stride=10):
    """生成每个被试的动态功能连接（FC矩阵序列）"""
    windows = sliding_windows(ts, win, stride)
    n_win, n_roi, _ = windows.shape
    fc_list = []
    for w in windows:
        fc = np.corrcoef(w)
        fc_list.append(fc)
    return np.array(fc_list)  # (n_win, n_roi, n_roi)
# ===================================================
# 自编码器
# ===================================================
class AE(nn.Module):
    def __init__(self, input_dim, z_dim=32):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z
# ===================================================
# 1. 读取数据并提取ROI特征
# ===================================================
data = np.load(NPZ_PATH)
arr = data["data"]
print(f"加载数据 shape: {arr.shape}")  # (Nsub, 90, T)
Nsub, Nroi, T = arr.shape
# 为每个ROI提取特征（其与其他ROI的平均连接模式 across 被试+时间）
roi_feats = []
for r in range(Nroi):
    # 收集该ROI在所有被试的FC连接变化
    roi_all = []
    for s in range(Nsub):
        fc_series = fc_features(arr[s], win_size, stride)  # (Nwin, Nroi, Nroi)
        roi_conn = fc_series[:, r, :]                      # (Nwin, Nroi)
        roi_all.append(roi_conn.mean(axis=0))              # 对时间平均
    roi_feats.append(np.mean(roi_all, axis=0))             # 再对被试平均
roi_feats = np.array(roi_feats)  # (90, 90)
print("ROI特征 shape:", roi_feats.shape)
# 标准化
scaler = StandardScaler()
roi_feats = scaler.fit_transform(roi_feats)
# ===================================================
# 2. 训练自编码器降维
# ===================================================
input_dim = roi_feats.shape[1]
ae = AE(input_dim, latent_dim).to(DEVICE)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()
dataset = TensorDataset(torch.tensor(roi_feats, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(ae_epochs):
    ae.train()
    total_loss = 0
    for (x,) in loader:
        x = x.to(DEVICE)
        x_rec, _ = ae(x)
        loss = criterion(x_rec, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(x)
    print(f"AE epoch {epoch + 1}, loss={total_loss / len(dataset):.6f}")
# 提取ROI嵌入表示
ae.eval()
with torch.no_grad():
    _, z = ae(torch.tensor(roi_feats, dtype=torch.float32).to(DEVICE))
    roi_embed = z.cpu().numpy()
print("编码后ROI特征 shape:", roi_embed.shape)
# ===================================================
# 3. 对ROI进行聚类
# ===================================================
kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
roi_labels = kmeans.fit_predict(roi_embed)
print("ROI聚类结果:", roi_labels)
# ===================================================
# 4. 写入 .node 文件（颜色=聚类标签）
# ===================================================
coords = np.loadtxt(AAL_NODE_PATH, dtype=str)
if coords.shape[1] < 4:
    raise ValueError("AAL90.node 文件应包含至少4列（前三列为坐标，第4列颜色值）")
coords[:, 3] = roi_labels.astype(str)
np.savetxt(SAVE_NODE_PATH, coords, fmt="%s", delimiter="\t")
print(f"聚类结果已保存到: {SAVE_NODE_PATH}")
