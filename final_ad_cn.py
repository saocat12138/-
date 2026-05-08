import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
import pandas as pd
from itertools import groupby
from scipy import stats
from scipy.optimize import linear_sum_assignment  # 用于状态匹配
import seaborn as sns
import matplotlib
# 本地环境配置 (如果不显示图窗可以去掉 'Agg' 或者保留用于直接存图)
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
# === 1. 参数与路径设置 (已更新为你的本地路径) ===
BASE_DIR = r"/longlab4090/bnuer_1"
DATA_PATH = os.path.join(BASE_DIR, "aal90_reordered.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "matrix_outputs_DEC_aligned_final123")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
N_STATES = 3
N_CLUSTERS = 17
THRESHOLD = 0.4
WIN, STRIDE = 20, 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === 2. 定义分组 ID (根据你提供的列表) ===
ad_subjects = [
    '002_S_5018', '006_S_4153', '006_S_4192', '006_S_4546', '006_S_4867',
    '013_S_5071', '018_S_4696', '018_S_4733', '018_S_5240', '019_S_4252',
    '019_S_4477', '019_S_4549', '019_S_5012', '019_S_5019', '031_S_4024',
    '053_S_5070', '053_S_5208', '100_S_5106', '130_S_4589', '130_S_4641',
    '130_S_4660', '130_S_4730', '130_S_4971', '130_S_4982', '130_S_4984',
    '130_S_4990', '130_S_4997', '130_S_5006', '130_S_5059', '136_S_4993'
]
cn_subjects = [
    '002_S_0413', '002_S_4213', '002_S_4225', '002_S_4262', '002_S_4264',
    '002_S_4270', '006_S_4150', '006_S_4357', '006_S_4449', '006_S_4485',
    '010_S_4442', '012_S_4026', '013_S_4579', '013_S_4580', '013_S_4616',
    '018_S_4257', '018_S_4313', '018_S_4349', '018_S_4400', '019_S_4367',
    '019_S_4835', '031_S_4032', '031_S_4218', '031_S_4474', '053_S_4578',
    '100_S_4469', '130_S_4343', '130_S_4352', '136_S_4269', '136_S_4433'
]
# === 3. 数据提取 (分AD和CN提取) ===
def extract_data_for_group(data_all, subject_indices):
    fcs, features = [], []
    n_roi, n_time = data_all.shape[1], data_all.shape[2]
    iu = np.triu_indices(n_roi, 1)
    for s in subject_indices:
        for start in range(0, n_time - WIN + 1, STRIDE):
            fc = np.nan_to_num(np.corrcoef(data_all[s, :, start:start + WIN]))
            fcs.append(fc)
            features.append(fc[iu])
    return np.array(fcs), np.array(features)
print(f" 正在加载数据: {DATA_PATH} ...")
data = np.load(DATA_PATH)
all_data = data['data']
subject_ids = data['subject_ids']
ad_indices, cn_indices = [], []
for i, sid in enumerate(subject_ids):
    if sid in ad_subjects:
        ad_indices.append(i)
    elif sid in cn_subjects:
        cn_indices.append(i)
print(f" 匹配到 AD 索引: {len(ad_indices)} 人, CN 索引: {len(cn_indices)} 人")
ad_fcs, ad_features = extract_data_for_group(all_data, ad_indices)
cn_fcs, cn_features = extract_data_for_group(all_data, cn_indices)
# === 4. 深度嵌入聚类 (DEC) 模型定义 ===
class DEC(nn.Module):
    def __init__(self, input_dim, z_dim=64, n_clusters=3):
        super(DEC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, z_dim)
        )
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, z_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def forward(self, x):
        z = self.encoder(x)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2))
        q = q / q.sum(dim=1, keepdim=True)
        return z, q
def get_target_distribution(q):
    p = q ** 2 / q.sum(0)
    return (p.t() / p.sum(1)).t()
# === 5. DEC 训练封装函数 ===
def train_dec_model(features_np, group_name="Group"):
    print(f" 开始 {group_name} 组的 DEC 训练...")
    X_tensor = torch.tensor(features_np, dtype=torch.float32).to(device)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    best_loss = float('inf')
    best_labels = None
    min_required_windows = int(len(features_np) * 0.02)
    for run in range(15):  # 运行多次防坍塌
        model = DEC(input_dim=features_np.shape[1], z_dim=64, n_clusters=N_STATES).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        with torch.no_grad():
            z_init = model.encoder(X_tensor)
            km = KMeans(n_clusters=N_STATES, n_init=10).fit(z_init.cpu().numpy())
            model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(device)
        for epoch in range(50):
            model.train()
            z, q = model(X_tensor)
            p = get_target_distribution(q).detach()
            loss = kl_loss(q.log(), p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        _, q_final = model(X_tensor)
        labels = q_final.argmax(1).cpu().numpy()
        counts = np.bincount(labels, minlength=N_STATES)
        if counts.min() >= min_required_windows and loss.item() < best_loss:
            best_loss = loss.item()
            best_labels = labels
    if best_labels is None:
        best_labels = labels  # 强制使用最后一次
    print(f" {group_name} 训练完成！最终状态分布: {np.bincount(best_labels, minlength=N_STATES).tolist()}")
    return best_labels
ad_labels = train_dec_model(ad_features, "AD")
cn_labels = train_dec_model(cn_features, "CN")
# === 6. 状态匹配 (匈牙利算法对齐 AD 和 CN 的状态) ===
print("\n 正在匹配对齐 AD 和 CN 的状态...")
def get_state_mean_fcs(fcs, labels):
    return [np.mean(fcs[labels == s], axis=0) if np.sum(labels == s) > 0 else np.zeros(fcs.shape[1:]) for s in
            range(N_STATES)]
ad_state_means = get_state_mean_fcs(ad_fcs, ad_labels)
cn_state_means = get_state_mean_fcs(cn_fcs, cn_labels)
iu = np.triu_indices(ad_fcs.shape[1], 1)
cost_matrix = np.zeros((N_STATES, N_STATES))
for i in range(N_STATES):
    for j in range(N_STATES):
        corr = np.corrcoef(ad_state_means[i][iu], cn_state_means[j][iu])[0, 1]
        cost_matrix[i, j] = -corr
row_ind, col_ind = linear_sum_assignment(cost_matrix)
print(" 状态匹配结果 (基于FC矩阵皮尔逊相关):")
for ad_idx, cn_idx in zip(row_ind, col_ind):
    print(f"  AD State {ad_idx + 1} <---> CN State {cn_idx + 1} (相关系数 r = {-cost_matrix[ad_idx, cn_idx]:.4f})")
mapping_cn_to_ad = {cn_idx: ad_idx for ad_idx, cn_idx in zip(row_ind, col_ind)}
cn_labels_aligned = np.array([mapping_cn_to_ad[l] for l in cn_labels])
cn_state_means_aligned = [cn_state_means[cn_idx] for cn_idx in col_ind]
# === 7. 超图计算与双排可视化 (精细 90x90 像素网格版) ===
print("\n 生成匹配对齐后的超图矩阵 (带90x90细网格与数字坐标)...")
def compute_hypergraph(state_fc):
    n_roi = state_fc.shape[0]
    km_roi = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=42).fit(state_fc)
    centers = km_roi.cluster_centers_
    H_soft = np.array([[np.corrcoef(state_fc[i], centers[j])[0, 1] for j in range(N_CLUSTERS)] for i in range(n_roi)])
    H = (H_soft >= THRESHOLD).astype(float)
    for i in np.where(H.sum(1) == 0)[0]: H[i, np.argmax(H_soft[i])] = 1.0
    W = np.diag(np.where(H.sum(0) > 0, 1.0 / H.sum(0), 0.0))
    S_raw = H @ W @ H.T
    s_min, s_max = S_raw.min(), S_raw.max()
    return ((S_raw - s_min) / (s_max - s_min + 1e-12))
network_labels = ['SMN', 'CON', 'AUN', 'DMN', 'VN', 'FPN', 'SN', 'SCN', 'none']
network_sizes = [8, 4, 8, 22, 12, 10, 8, 10, 8]
boundaries = np.cumsum(network_sizes)
label_positions = boundaries - np.array(network_sizes) / 2
# 画布稍微变大一点，确保 90 个数字能印得下
fig, axes = plt.subplots(2, N_STATES, figsize=(8 * N_STATES, 16))
for row, (group_name, state_means) in enumerate([("AD", ad_state_means), ("CN", cn_state_means_aligned)]):
    for s in range(N_STATES):
        S_norm = compute_hypergraph(state_means[s])
        ax = axes[row, s] if N_STATES > 1 else axes[row]
        im = ax.imshow(S_norm, cmap='jet', vmin=-1, vmax=1, aspect='auto')
        # ---------------- 核心改进：双层网格 ----------------
        # 1. 画 90x90 的微观细网格 (每一个脑区都隔开)
        for i in range(1, 90):
            ax.axvline(x=i - 0.5, color='white', linestyle=':', linewidth=0.3, alpha=0.5)
            ax.axhline(y=i - 0.5, color='white', linestyle=':', linewidth=0.3, alpha=0.5)
        # 2. 画 9大网络 的宏观边界 (粗虚线)
        for b in boundaries[:-1]:
            ax.axvline(x=b - 0.5, color='black', linestyle='-', linewidth=1.0, alpha=0.9)
            ax.axhline(y=b - 0.5, color='black', linestyle='-', linewidth=1.0, alpha=0.9)
        # ---------------- 核心改进：双重坐标 ----------------
        # 3. 底部和左侧：保留网络标签 (宏观)
        ax.set_xticks(label_positions)
        ax.set_xticklabels(network_labels, rotation=90, ha='right', fontsize=14, fontweight='bold')
        ax.set_yticks(label_positions)
        ax.set_yticklabels(network_labels, fontsize=14, fontweight='bold')
        # 4. 顶部和右侧：添加 1-90 的数字编号 (微观)
        ax_top = ax.twiny()
        ax_right = ax.twinx()
        ax_top.set_xlim(ax.get_xlim())
        ax_right.set_ylim(ax.get_ylim())
        # 设置 1-90 的刻度
        ax_top.set_xticks(np.arange(90))
        ax_top.set_xticklabels(np.arange(1, 91), rotation=90, fontsize=4)  # 字体设为4，避免重叠
        ax_right.set_yticks(np.arange(90))
        ax_right.set_yticklabels(np.arange(1, 91), fontsize=4)
        # 去掉自带的小短线让图面干净
        ax_top.tick_params(axis='x', length=1, pad=1, color='gray')
        ax_right.tick_params(axis='y', length=1, pad=1, color='gray')
        ax.set_title(f"{group_name} State {s + 1}", fontsize=20, fontweight='bold', pad=20)
        # 给colorbar留点空间
        plt.colorbar(im, ax=ax_right, shrink=0.7, pad=0.08)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "DEC_Aligned_Hypergraphs_90Grid.png")
#  注意：DPI 改为了 600，因为 90 个数字字号很小，保存高分辨率才能在电脑上放大看清！
plt.savefig(save_path, dpi=600)
plt.close()
print(f" 带有精细 90x90 网格与编号的超图已保存至: {save_path}")
# === 8. 计算对齐后的时间指标与 T检验 ===
print("\n 计算并对比对齐后的时间指标...")
ad_seqs = ad_labels.reshape(len(ad_indices), -1)
cn_seqs = cn_labels_aligned.reshape(len(cn_indices), -1)
results = []
def append_metrics(seqs, group_name):
    for subject_id, seq in enumerate(seqs):
        total_len = len(seq)
        for state in range(N_STATES):
            count = np.sum(seq == state)
            dc = count / total_len
            durs = [sum(1 for _ in group) for k, group in groupby(seq) if k == state]
            avg_dur = np.mean(durs) if len(durs) > 0 else np.nan
            results.append({
                'Group': group_name,
                'State': f'State {state + 1}',
                'Duty_Cycle': dc,
                'Avg_Duration': avg_dur
            })
append_metrics(ad_seqs, 'AD')
append_metrics(cn_seqs, 'CN')
df_metrics = pd.DataFrame(results)
p_values_dc = []
p_values_dur = []
print("\n========== T-Test Results (Aligned States) ==========")
for i in range(N_STATES):
    state_name = f'State {i + 1}'
    df_s = df_metrics[df_metrics['State'] == state_name]
    ad_dc = df_s[df_s['Group'] == 'AD']['Duty_Cycle']
    cn_dc = df_s[df_s['Group'] == 'CN']['Duty_Cycle']
    ad_dur = df_s[df_s['Group'] == 'AD']['Avg_Duration'].dropna()
    cn_dur = df_s[df_s['Group'] == 'CN']['Avg_Duration'].dropna()
    _, p_dc = stats.ttest_ind(ad_dc, cn_dc, equal_var=False)
    _, p_dur = stats.ttest_ind(ad_dur, cn_dur, equal_var=False)
    p_values_dc.append(p_dc)
    p_values_dur.append(p_dur)
    print(f"{state_name}:")
    print(f"  - 占空比 p-value:     {p_dc:.4f} {'(显著 *)' if p_dc < 0.05 else ''}")
    print(f"  - 持续时间 p-value:   {p_dur:.4f} {'(显著 *)' if p_dur < 0.05 else ''}")
# === 9. 绘制指标柱状图并加显著性标注 (放宽P值阈值版) ===
sns.set_theme(style="whitegrid")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
palette_colors = {'AD': '#2CA02C', 'CN': '#FF7F0E'}
def add_stat_annotation(ax, df, metric, p_values):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for i, p in enumerate(p_values):
        # ！！！修改点：将显著性阈值从 0.05 放宽到 0.1 ！！！
        if p < 0.5:
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = "#"  # 用 # 或 + 表示边缘显著趋势 (0.05 < p < 0.1)
            state_name = f'State {i + 1}'
            grouped = df[df['State'] == state_name].groupby('Group')[metric]
            ci_tops = grouped.mean() + 1.96 * grouped.sem()
            error_top = ci_tops.max()
            if pd.isna(error_top): error_top = df[df['State'] == state_name][metric].max()
            line_y = error_top + 0.04 * y_range
            text_y = line_y + 0.01 * y_range
            x1, x2 = i - 0.2, i + 0.2
            ax.plot([x1, x1, x2, x2], [line_y - 0.015 * y_range, line_y, line_y, line_y - 0.015 * y_range], lw=1.5,
                    c='black')
            # 微调星号和 # 的显示位置
            font_size = 14 if stars != "#" else 12
            ax.text((x1 + x2) * 0.5, text_y, stars, ha='center', va='bottom', color='black', fontsize=font_size,
                    fontweight='bold')
            if text_y + 0.05 * y_range > ax.get_ylim()[1]:
                ax.set_ylim(y_min, text_y + 0.1 * y_range)
# 图 1：占空比
sns.barplot(data=df_metrics, x='State', y='Duty_Cycle', hue='Group', palette=palette_colors, capsize=0.1, ax=axes2[0])
add_stat_annotation(axes2[0], df_metrics, 'Duty_Cycle', p_values_dc)
axes2[0].set_title(f'Aligned Duty Cycle (AD vs CN)', fontsize=14, fontweight='bold')
axes2[0].set_xlabel('Microstates', fontsize=12)
axes2[0].set_ylabel('Duty Cycle (Proportion)', fontsize=12)
# 图 2：平均持续时间
sns.barplot(data=df_metrics, x='State', y='Avg_Duration', hue='Group', palette=palette_colors, capsize=0.1, ax=axes2[1])
add_stat_annotation(axes2[1], df_metrics, 'Avg_Duration', p_values_dur)
axes2[1].set_title(f'Aligned Average Duration (AD vs CN)', fontsize=14, fontweight='bold')
axes2[1].set_xlabel('Microstates', fontsize=12)
axes2[1].set_ylabel('Average Duration (Timepoints)', fontsize=12)
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "AD_CN_Aligned_Differences_BarChart.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"🎉 统计直方图已生成并保存至: {chart_path} (已标注边缘显著 #)")
print("=================== 运行结束 ===================")
