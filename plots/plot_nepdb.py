
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. Nature期刊全局审美配置
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 13

# 统一三色阵营 (Red, Blue, Grey)
COLOR_OURS = '#E64B35'     # 深赤红 (Epi-UQ)
COLOR_TRANSHLA = '#4DBBD5' # 蔚蓝 (TransHLA)
COLOR_BASELINE = '#888888' # 深灰 (雷达图基线线色 / 图例统称)
COLOR_OTHER = '#B0B0B0'    # 浅灰 (森林图基线散点色)

# ==========================================
# 2. 实验数据结构化
# ==========================================
hla1_data = {
    "Our model":     [0.879, 0.013, 0.761, 0.022, 0.884, 0.003, 0.948, 0.005, "Agnostic"],
    "TransHLA":      [0.8664, 0.0126, 0.7305, 0.0149, 0.8508, 0.0223, 0.9315, 0.0032, "Agnostic"],
    "Mhcflurry":     [0.8725, 0.0214, 0.7503, 0.0290, 0.7744, 0.0338, 0.9704, 0.0109, "Dependent"],
    "NetMHCPan4.1b": [0.8677, 0.0157, 0.7343, 0.0252, 0.7909, 0.0272, 0.9625, 0.0105, "Dependent"],
    "MixMHCpred":    [0.7349, 0.0230, 0.4328, 0.0185, 0.5813, 0.0287, 0.9556, 0.0156, "Dependent"],
    "Anthem":        [0.6942, 0.0243, 0.2978, 0.0354, 0.5333, 0.0281, 0.9275, 0.0078, "Dependent"],
    "TransPHLA":     [0.5667, 0.0675, 0.1806, 0.1045, 0.5896, 0.0657, 0.6366, 0.0468, "Dependent"],
    "Mhcnuggets":    [0.6527, 0.0265, 0.0000, 0.0000, 0.4849, 0.0292, 0.7261, 0.0140, "Dependent"]
}

hla2_data = {
    "Our model":      [0.685, 0.107, 0.514, 0.122, 0.860, 0.028, 0.843, 0.046, "Agnostic"],
    "MixMHC2pred":    [0.7209, 0.1508, 0.3764, 0.0534, 0.5816, 0.1917, 0.5598, 0.1259, "Dependent"],
    "TransHLA":       [0.5681, 0.1183, 0.2046, 0.2223, 0.6222, 0.1397, 0.6314, 0.0372, "Agnostic"],
    "NetMHCIIpan4.3b":[0.6336, 0.0825, 0.1521, 0.1202, 0.4993, 0.0992, 0.6406, 0.0819, "Dependent"],
    "Mhcnuggets":     [0.6589, 0.1232, 0.1503, 0.2575, 0.5853, 0.1737, 0.4147, 0.1763, "Dependent"],
    "DeepSeqPanII":   [0.5416, 0.1473, -0.0759, 0.1439, 0.4982, 0.2234, 0.5224, 0.1577, "Dependent"]
}

# ==========================================
# 3. 核心绘图函数
# ==========================================
def plot_forest(ax, data, title):
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1][2]))
    models = list(sorted_data.keys())
    mcc_means = [val[2] for val in sorted_data.values()]
    mcc_stds  = [val[3] for val in sorted_data.values()]
    categories = [val[8] for val in sorted_data.values()]
    
    y_pos = np.arange(len(models))
    
    # ★ 修复 1：强制扩展 Y 轴顶部空间 (加高 1.5 个单位)，为文字腾出专属安全区
    ax.set_ylim(-0.5, len(models) + 1.2) 
    
    ax.axvline(x=0.0, color='#E64B35', linestyle='--', linewidth=1.5, zorder=1)
    
    # 文字现在安全地放置在所有模型之上，绝对不会发生任何横向或纵向重叠
    ax.text(0.02, len(models) - 0.2, 'Representation Collapse\n(MCC ≤ 0)', color='#E64B35', 
            fontsize=11, fontweight='bold', va='bottom', ha='left', zorder=5)
    
    ax.axvline(x=0.5, color='#DDDDDD', linestyle=':', zorder=0)
    ax.axvline(x=0.8, color='#DDDDDD', linestyle=':', zorder=0)

    for i in range(len(models)):
        model = models[i]
        mean = mcc_means[i]
        std = mcc_stds[i]
        
        if model == "Our model":
            color, marker, size = COLOR_OURS, 's', 100
        elif model == "TransHLA":
            color, marker, size = COLOR_TRANSHLA, 's', 80
        else:
            color, marker, size = COLOR_OTHER, 'o', 60
            
        ax.errorbar(mean, y_pos[i], xerr=std, fmt=marker, color=color, 
                    ecolor=color, elinewidth=2, capsize=4, capthick=1.5, 
                    markersize=np.sqrt(size), markeredgecolor='white', zorder=3)

    ax.set_yticks(y_pos)
    yticklabels = ax.set_yticklabels(models, fontweight='bold')
    for idx, label in enumerate(yticklabels):
        if categories[idx] == 'Dependent':
            label.set_color('#777777') 
        if models[idx] == "Our model":
            label.set_color(COLOR_OURS)

    ax.set_xlabel('MCC', fontweight='bold', fontsize=13) 
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

def plot_radar(ax, data, models_to_plot, title):
    metrics = ['AUC', 'F1', 'MCC', 'Precision']
    num_vars = len(metrics)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', pad=18)
    
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="grey", size=10)
    ax.grid(color='#DDDDDD', linestyle='--', linewidth=1)
    ax.spines['polar'].set_color('#BBBBBB')

    for model_info in models_to_plot:
        model_name = model_info['name']
        color = model_info['color']
        
        m_data = data[model_name]
        means = [m_data[6], m_data[0], m_data[2], m_data[4]]
        stds  = [m_data[7], m_data[1], m_data[3], m_data[5]]
        
        means += means[:1]
        stds += stds[:1]
        
        # 统一线宽与图层
        ax.plot(angles, means, color=color, linewidth=2.5, zorder=3)
        upper_bound = np.clip(np.array(means) + np.array(stds), 0, 1.0)
        lower_bound = np.clip(np.array(means) - np.array(stds), 0, 1.0)
        ax.fill_between(angles, lower_bound, upper_bound, color=color, alpha=0.15, zorder=2)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

# ==========================================
# 4. 组装 1x4 图像矩阵
# ==========================================
# 🔴 修复 1：将画布稍微加宽 (从 20 增加到 22)，为长文本提供更多物理像素
fig = plt.figure(figsize=(22, 5.2))

# 🔴 修复 2：将 wspace（子图间的横向留白）从 0.35 大幅增加到 0.65！
# 这会在 B图(雷达) 和 C图(森林) 之间扯开一道巨大的安全沟壑，彻底隔离文字
gs = GridSpec(1, 4, width_ratios=[1.6, 1.0, 1.6, 1.0], wspace=0.65)

# A. HLA-I Forest
ax1 = fig.add_subplot(gs[0, 0])
plot_forest(ax1, hla1_data, "A. HLA-I Neoepitope Detection")

# B. HLA-I Radar (更换基线颜色为统一个深灰)
ax2 = fig.add_subplot(gs[0, 1], polar=True)
models_radar_hla1 = [
    {'name': 'Our model', 'color': COLOR_OURS},
    {'name': 'TransHLA', 'color': COLOR_TRANSHLA},
    {'name': 'NetMHCPan4.1b', 'color': COLOR_BASELINE} # 统一基线灰
]
plot_radar(ax2, hla1_data, models_radar_hla1, "B. HLA-I")

# C. HLA-II Forest
ax3 = fig.add_subplot(gs[0, 2])
plot_forest(ax3, hla2_data, "C. HLA-II Neoepitope Detection")

# D. HLA-II Radar (更换基线颜色为统一个深灰)
ax4 = fig.add_subplot(gs[0, 3], polar=True)
models_radar_hla2 = [
    {'name': 'Our model', 'color': COLOR_OURS},
    {'name': 'TransHLA', 'color': COLOR_TRANSHLA},
    {'name': 'MixMHC2pred', 'color': COLOR_BASELINE} # 统一基线灰
]
plot_radar(ax4, hla2_data, models_radar_hla2, "D. HLA-II")

# ==========================================
# 5. ★ 修复 2：添加全局极简统一三色图例
# ==========================================
unified_legend = [
    mlines.Line2D([], [], color=COLOR_OURS, marker='s', linestyle='-', linewidth=2.5, markersize=10, label='Epi-UQ (Ours, Allele-Agnostic)'),
    mlines.Line2D([], [], color=COLOR_TRANSHLA, marker='s', linestyle='-', linewidth=2.5, markersize=9, label='TransHLA (Allele-Agnostic)'),
    mlines.Line2D([], [], color=COLOR_BASELINE, marker='o', linestyle='-', linewidth=2.5, markersize=9, label='Allele-Dependent Baselines')
]

# 🔴 修改点 1：将 Y 轴位置从 -0.05 往上提到 0.00
fig.legend(handles=unified_legend, loc='lower center', bbox_to_anchor=(0.5, 0.00), ncol=3, frameon=False, fontsize=14)

# 🔴 修改点 2：将 bottom 留白从 0.08 增加到 0.12，防止子图挤压图例
plt.tight_layout(rect=[0, 0.12, 1, 1])

# 保存高清图 (bbox_inches='tight' 会自动包裹包含图例的所有元素)
plt.savefig("plot_nepdb_1x4_final.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.savefig("plot_nepdb_1x4_final.png", format='png', dpi=600, bbox_inches='tight')

print("图例截断完美修复！")
plt.show()
