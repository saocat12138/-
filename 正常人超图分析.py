import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
# === 新增的统计与绘图库 ===
import pandas as pd
from itertools import groupby
from scipy import stats
import seaborn as sns
# 服务器环境配置
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
# === 1. 参数与路径设置 ===
BASE_DIR = "/longlab4090/bnuer_1"
DATA_PATH = os.path.join(BASE_DIR, "subs100_90_1200.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "matrix_outputs_DEC12")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
# ★ 你已经改成了 3 个状态，非常完美！
N_STATES = 3
N_CLUSTERS = 17
THRESHOLD = 0.4
WIN, STRIDE = 20, 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === 1.5 男女被试索引配置 ===
male = [1, 2, 3, 5, 9, 11, 12, 16, 17, 19, 22, 23, 25, 26, 27, 33, 36, 37, 42, 45, 46, 50, 54, 55, 56, 58, 61, 62, 63,
        64, 67, 69, 71, 75, 76, 77, 79, 80, 82, 88, 89, 90, 93, 94, 95, 98]
female = [0, 4, 6, 7, 8, 10, 13, 14, 15, 18, 20, 21, 24, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 41, 43, 44, 47, 48, 49,
          51, 52, 53, 57, 59, 60, 65, 66, 68, 70, 72, 73, 74, 78, 81, 83, 84, 85, 86, 87, 91, 92, 96, 97, 99]
gender_map = {idx: 'Male' for idx in male}
gender_map.update({idx: 'Female' for idx in female})
# === 2. 深度嵌入聚类 (DEC) 模型定义 ===
class DEC(nn.Module):
    def __init__(self, input_dim, z_dim=64, n_clusters=4):
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
# === 3. 数据提取与准备 ===
print(f" 提取 100 个样本的窗口特征...")
data = np.load(DATA_PATH)
all_data = data[list(data.keys())[0]][:100]
n_subs, n_roi, n_time = all_data.shape
all_fcs = []
features = []
iu = np.triu_indices(n_roi, 1)
for s in range(n_subs):
    for start in range(0, n_time - WIN + 1, STRIDE):
        fc = np.nan_to_num(np.corrcoef(all_data[s, :, start:start + WIN]))
        all_fcs.append(fc)
        features.append(fc[iu])
features = np.array(features)
X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
print(f" 提取完成，总窗口数: {len(features)}")
# === 4. DEC 训练过程 (多次运行寻找最优且平衡的解) ===
print(f" 开始 DEC 状态聚类训练 (正在聚类为 {N_STATES} 个状态)...")
N_RUNS = 20
best_loss = float('inf')
best_labels = None
kl_loss = nn.KLDivLoss(reduction='batchmean')
total_windows = len(features)
min_required_windows = int(total_windows * 0.02)
for run in range(N_RUNS):
    model = DEC(input_dim=features.shape[1], z_dim=64, n_clusters=N_STATES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        z_init = model.encoder(X_tensor)
        km = KMeans(n_clusters=N_STATES, n_init=20).fit(z_init.cpu().numpy())
        model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(device)
    for epoch in range(50):
        model.train()
        z, q = model(X_tensor)
        p = get_target_distribution(q).detach()
        loss = kl_loss(q.log(), p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = loss.item()
    model.eval()
    _, q_final = model(X_tensor)
    current_labels = q_final.argmax(1).cpu().numpy()
    counts = np.bincount(current_labels, minlength=N_STATES)
    min_count = counts.min()
    is_valid_solution = min_count >= min_required_windows
    print(f"Run {run + 1:02d}/{N_RUNS} | Loss: {final_loss:.5f} | 状态分布: {counts.tolist()}", end="")
    if is_valid_solution and final_loss < best_loss:
        best_loss = final_loss
        best_labels = current_labels
        print("  -->  接受！当前最佳模型")
    elif not is_valid_solution:
        print(f"  -->  丢弃！发生聚类坍塌 (某状态仅有 {min_count} 个)")
    else:
        print("")
if best_labels is None:
    print(f" 警告：所有运行都发生了坍塌。强制使用最后一次结果。")
    best_labels = current_labels
print(f" 搜索结束！保留有效且最低 Loss 的聚类结果进行后续分析。")
labels = best_labels
# === 5. 超图生成与可视化 ===
print(f" 生成 {N_STATES} 个状态的超图矩阵...")
# ★ 修改点 1：这里的子图数量和画板宽度改为了自适应 N_STATES，避免固定 4 导致报错
fig, axes = plt.subplots(1, N_STATES, figsize=(6 * N_STATES, 6))
for s in range(N_STATES):
    idx = np.where(labels == s)[0]
    if len(idx) == 0: continue
    state_fc = np.mean([all_fcs[i] for i in idx], axis=0)
    km_roi = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=42).fit(state_fc)
    centers = km_roi.cluster_centers_
    H_soft = np.array([[np.corrcoef(state_fc[i], centers[j])[0, 1] for j in range(N_CLUSTERS)] for i in range(n_roi)])
    H = (H_soft >= THRESHOLD).astype(float)
    for i in np.where(H.sum(1) == 0)[0]: H[i, np.argmax(H_soft[i])] = 1.0
    W = np.diag(np.where(H.sum(0) > 0, 1.0 / H.sum(0), 0.0))
    S_raw = H @ W @ H.T
    s_min, s_max = S_raw.min(), S_raw.max()
    S_norm = ((S_raw - s_min) / (s_max - s_min + 1e-12))
    # 根据状态数量，axes 的索引方式也自适应
    ax = axes[s] if N_STATES > 1 else axes
    im = ax.imshow(S_norm, cmap='jet', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f"DEC State {s + 1}\n(n={len(idx)} windows)")
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "DEC_states_hypergraph.png")
plt.savefig(save_path, dpi=300)
print(f" 超图已保存至: {save_path}")
plt.close()
# ========================================================================
# === 6. 计算指标、T检验，并绘制带有显著性星号的柱状图 ===
# ========================================================================
print(" 正在计算男女的占空比与平均持续时间...")
state_data = labels.reshape(n_subs, -1)
results = []
for subject_id in range(n_subs):
    seq = state_data[subject_id]
    total_len = len(seq)
    for state in range(N_STATES):
        state_count = np.sum(seq == state)
        duty_cycle = state_count / total_len
        durations = [sum(1 for _ in group) for key, group in groupby(seq) if key == state]
        avg_duration = np.mean(durations) if len(durations) > 0 else np.nan
        results.append({
            'Subject_ID': subject_id,
            'Gender': gender_map[subject_id],
            'State': f'State {state + 1}',
            'Duty_Cycle': duty_cycle,
            'Avg_Duration': avg_duration
        })
df_metrics = pd.DataFrame(results)
# -- 提取 T 检验 P 值 --
print("\n========== 男女差异 T检验 P-value ==========")
p_values_dc = []
p_values_dur = []
for state in [f'State {i + 1}' for i in range(N_STATES)]:
    df_state = df_metrics[df_metrics['State'] == state]
    male_data = df_state[df_state['Gender'] == 'Male']
    female_data = df_state[df_state['Gender'] == 'Female']
    _, p_dc = stats.ttest_ind(male_data['Duty_Cycle'], female_data['Duty_Cycle'], equal_var=False)
    _, p_dur = stats.ttest_ind(male_data['Avg_Duration'].dropna(), female_data['Avg_Duration'].dropna(),
                               equal_var=False)
    p_values_dc.append(p_dc)
    p_values_dur.append(p_dur)
    print(f"{state}:")
    print(f"  - 占空比 p-value:     {p_dc:.4f} {'(显著 *)' if p_dc < 0.05 else ''}")
    print(f"  - 持续时间 p-value:   {p_dur:.4f} {'(显著 *)' if p_dur < 0.05 else ''}")
# -- 绘制柱状图并添加显著性标记 --
sns.set_theme(style="whitegrid")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
palette_colors = {'Male': '#2CA02C', 'Female': '#FF7F0E'}
# 辅助函数：在图表上画显著性星号
def add_stat_annotation(ax, df, metric, p_values):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for i, p in enumerate(p_values):
        if p < 0.05:
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            else:
                stars = "*"
            state_name = f'State {i + 1}'
            grouped = df[df['State'] == state_name].groupby('Gender')[metric]
            ci_tops = grouped.mean() + 1.96 * grouped.sem()
            error_top = ci_tops.max()
            if pd.isna(error_top): error_top = df[df['State'] == state_name][metric].max()
            line_y = error_top + 0.04 * y_range
            text_y = line_y + 0.01 * y_range
            x1, x2 = i - 0.2, i + 0.2
            ax.plot([x1, x1, x2, x2], [line_y - 0.015 * y_range, line_y, line_y, line_y - 0.015 * y_range], lw=1.5,
                    c='black')
            ax.text((x1 + x2) * 0.5, text_y, stars, ha='center', va='bottom', color='black', fontsize=14,
                    fontweight='bold')
            if text_y + 0.05 * y_range > ax.get_ylim()[1]:
                ax.set_ylim(y_min, text_y + 0.1 * y_range)
# 图 1：占空比
sns.barplot(data=df_metrics, x='State', y='Duty_Cycle', hue='Gender', palette=palette_colors, capsize=0.1, ax=axes2[0])
add_stat_annotation(axes2[0], df_metrics, 'Duty_Cycle', p_values_dc)
# ★ 修改点 2：将原来写死的 '4 States' 改为自适应的变量
axes2[0].set_title(f'Duty Cycle across {N_STATES} States (Male vs Female)', fontsize=14, fontweight='bold')
axes2[0].set_xlabel('Microstates', fontsize=12)
axes2[0].set_ylabel('Duty Cycle (Proportion)', fontsize=12)
axes2[0].legend(title='Gender')
# 图 2：平均持续时间
sns.barplot(data=df_metrics, x='State', y='Avg_Duration', hue='Gender', palette=palette_colors, capsize=0.1,
            ax=axes2[1])
add_stat_annotation(axes2[1], df_metrics, 'Avg_Duration', p_values_dur)
# ★ 修改点 3：同上
axes2[1].set_title(f'Average Duration across {N_STATES} States (Male vs Female)', fontsize=14, fontweight='bold')
axes2[1].set_xlabel('Microstates', fontsize=12)
axes2[1].set_ylabel('Average Duration (Timepoints)', fontsize=12)
axes2[1].legend(title='Gender')
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "Gender_Differences_BarChart.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f" 带有显著性标注的图表已生成，保存至: {chart_path}")
# === 5. 超图生成与可视化 (优化标签版) ===
print(f" 生成 {N_STATES} 个状态的超图矩阵并标注网络标签...")
# --- 网络标签与边界计算 ---
network_labels = ['SMN', 'CON', 'AUN', 'DMN', 'VN', 'FPN', 'SN', 'SCN', 'none']
network_sizes = [8, 4, 8, 22, 12, 10, 8, 10, 8]
# 计算边界位置
boundaries = np.cumsum(network_sizes)
# 计算标签居中的位置
label_positions = boundaries - np.array(network_sizes) / 2
fig, axes = plt.subplots(1, N_STATES, figsize=(7 * N_STATES, 7))
for s in range(N_STATES):
    idx = np.where(labels == s)[0]
    if len(idx) == 0: continue
    state_fc = np.mean([all_fcs[i] for i in idx], axis=0)
    # 保持你原有的超图生成逻辑
    km_roi = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=42).fit(state_fc)
    centers = km_roi.cluster_centers_
    H_soft = np.array([[np.corrcoef(state_fc[i], centers[j])[0, 1] for j in range(N_CLUSTERS)] for i in range(n_roi)])
    H = (H_soft >= THRESHOLD).astype(float)
    for i in np.where(H.sum(1) == 0)[0]: H[i, np.argmax(H_soft[i])] = 1.0
    W = np.diag(np.where(H.sum(0) > 0, 1.0 / H.sum(0), 0.0))
    S_raw = H @ W @ H.T
    s_min, s_max = S_raw.min(), S_raw.max()
    S_norm = ((S_raw - s_min) / (s_max - s_min + 1e-12))
    ax = axes[s] if N_STATES > 1 else axes
    im = ax.imshow(S_norm, cmap='jet', vmin=-1, vmax=1, aspect='auto')
    # --- 核心改进：添加标签和网格线 ---
    # 绘制网络边界线
    for b in boundaries[:-1]:
        ax.axvline(x=b - 0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axhline(y=b - 0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
    # 设置刻度标签
    ax.set_xticks(label_positions)
    ax.set_xticklabels(network_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(label_positions)
    ax.set_yticklabels(network_labels, fontsize=9)
    ax.set_title(f"DEC State {s + 1}\n(n={len(idx)} windows)", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "DEC_states_with_labels.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f" 带网络标签的超图已保存至: {save_path}")
