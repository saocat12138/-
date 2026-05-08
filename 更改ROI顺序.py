import numpy as np
# ==================== 1. 加载原始数据 ====================
data = np.load('D:/BaiduNetdiskDownload/ADCNSUM/aal90_final.npz')
timeseries = data['data']        # (56, 90, 140)
subject_ids = data['subject_ids']
print(f"原始数据形状: {timeseries.shape}")
# ==================== 2. 定义新的顺序 ====================
order = [1,2,19,20,57,58,69,70,
         11,12,83,84,
         17,18,63,64,79,80,81,82,
         23,24,25,26,31,32,35,36,37,38,39,40,65,66,67,68,85,86,87,88,43,44,
         45,46,47,48,49,50,51,52,53,54,55,56,
         3,4,5,6,7,8,9,10,61,62,
         13,14,29,30,33,34,59,60,
         71,72,73,74,75,76,77,78,15,16,
         21,22,27,28,41,42,89,90]
# 转换为0-based索引
order = np.array(order) - 1
print(f"新顺序长度: {len(order)}")  # 应该是90
print(f"新顺序前10个: {order[:10]}")
print(f"新顺序后10个: {order[-10:]}")
# ==================== 3. 按照新顺序重排脑区 ====================
timeseries_reordered = timeseries[:, order, :]  # (56, 90, 140)
print(f"\n重排后数据形状: {timeseries_reordered.shape}")
# ==================== 4. 保存为新的npz文件 ====================
output_file = 'D:/BaiduNetdiskDownload/ADCNSUM/aal90_reordered.npz'
np.savez_compressed(
    output_file,
    data=timeseries_reordered,
    subject_ids=subject_ids,
    order=order  # 保存顺序信息
)
print(f"\n✓ 已保存: {output_file}")
# ==================== 5. 同时保存脑区名称（按新顺序） ====================
# 原始AAL-90脑区名称
aal90_names = [
    'Precentral_L', 'Precentral_R',                    # 1,2
    'Frontal_Sup_L', 'Frontal_Sup_R',                  # 3,4
    'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R',          # 5,6
    'Frontal_Mid_L', 'Frontal_Mid_R',                  # 7,8
    'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',          # 9,10
    'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R',        # 11,12
    'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',          # 13,14
    'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',          # 15,16
    'Rolandic_Oper_L', 'Rolandic_Oper_R',              # 17,18
    'Supp_Motor_Area_L', 'Supp_Motor_Area_R',          # 19,20
    'Olfactory_L', 'Olfactory_R',                      # 21,22
    'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',    # 23,24
    'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R',          # 25,26
    'Rectus_L', 'Rectus_R',                            # 27,28
    'Insula_L', 'Insula_R',                            # 29,30
    'Cingulate_Ant_L', 'Cingulate_Ant_R',              # 31,32
    'Cingulate_Mid_L', 'Cingulate_Mid_R',              # 33,34
    'Cingulate_Post_L', 'Cingulate_Post_R',            # 35,36
    'Hippocampus_L', 'Hippocampus_R',                  # 37,38
    'Parahippocampal_L', 'Parahippocampal_R',          # 39,40
    'Amygdala_L', 'Amygdala_R',                        # 41,42
    'Calcarine_L', 'Calcarine_R',                      # 43,44
    'Cuneus_L', 'Cuneus_R',                            # 45,46
    'Lingual_L', 'Lingual_R',                          # 47,48
    'Occipital_Sup_L', 'Occipital_Sup_R',              # 49,50
    'Occipital_Mid_L', 'Occipital_Mid_R',              # 51,52
    'Occipital_Inf_L', 'Occipital_Inf_R',              # 53,54
    'Fusiform_L', 'Fusiform_R',                        # 55,56
    'Postcentral_L', 'Postcentral_R',                  # 57,58
    'Parietal_Sup_L', 'Parietal_Sup_R',                # 59,60
    'Parietal_Inf_L', 'Parietal_Inf_R',                # 61,62
    'Supramarginal_L', 'Supramarginal_R',              # 63,64
    'Angular_L', 'Angular_R',                          # 65,66
    'Precuneus_L', 'Precuneus_R',                      # 67,68
    'Paracentral_Lobule_L', 'Paracentral_Lobule_R',    # 69,70
    'Caudate_L', 'Caudate_R',                          # 71,72
    'Putamen_L', 'Putamen_R',                          # 73,74
    'Pallidum_L', 'Pallidum_R',                        # 75,76
    'Thalamus_L', 'Thalamus_R',                        # 77,78
    'Heschl_L', 'Heschl_R',                            # 79,80
    'Temporal_Sup_L', 'Temporal_Sup_R',                # 81,82
    'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R',      # 83,84
    'Temporal_Mid_L', 'Temporal_Mid_R',                # 85,86
    'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',      # 87,88
    'Temporal_Inf_L', 'Temporal_Inf_R'                 # 89,90
]
# 按新顺序重排脑区名称
reordered_names = [aal90_names[i] for i in order]
# 保存对照表
import pandas as pd
df = pd.DataFrame({
    'new_index': range(1, 91),
    'original_index': order + 1,
    'region_name': reordered_names
})
df.to_csv('D:/BaiduNetdiskDownload/ADCNSUM/aal90_reordered_names.csv', index=False)
print(f"✓ 已保存脑区名称对照表: aal90_reordered_names.csv")
# ==================== 6. 验证 ====================
print("\n新顺序前10个脑区:")
for i in range(10):
    print(f"  新索引{i+1}: 原索引{order[i]+1} -> {reordered_names[i]}")
print(f"\n新顺序后10个脑区:")
for i in range(-10, 0):
    print(f"  新索引{90+i+1}: 原索引{order[i]+1} -> {reordered_names[i]}")
