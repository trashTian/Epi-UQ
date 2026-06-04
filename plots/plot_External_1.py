import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ==========================================
# 1. Nature期刊审美配置
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 12

# 颜色配置 (Nature风格)
COLOR_OURS = '#E64B35'     # 深赤红 (Our model)
COLOR_TRANSHLA = '#4DBBD5' # 蔚蓝 (TransHLA)
COLOR_BASELINE = '#999999' # 高级灰 (Allele-dependent baselines)

# ==========================================
# 2. 实验数据输入
# ==========================================
metrics_order = ['AUC', 'F1', 'Precision', 'MCC']

# HLA-I 数据
hla1_models = ["Epi-UQ", "TransHLA", "Mhcflurry", "NetMHCPan4.1b", "MixMHCpred", "Anthem", "TransPHLA", "Mhcnuggets"]
hla1_colors = [COLOR_OURS, COLOR_TRANSHLA, COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE]
hla1_data = {
    'AUC':       [(0.910, 0.002), (0.8886, 0.0034), (0.9001, 0.0011), (0.8410, 0.0021), (0.8861, 0.0010), (0.8596, 0.0011), (0.5787, 0.0043), (0.6183, 0.0047)],
    'F1':        [(0.824, 0.003), (0.8101, 0.0140), (0.8145, 0.0007), (0.7699, 0.0022), (0.7274, 0.0017), (0.7057, 0.0009), (0.5408, 0.0050), (0.6663, 0.0006)],
    'Precision': [(0.877, 0.012), (0.8262, 0.0204), (0.7308, 0.0012), (0.7597, 0.0023), (0.5838, 0.0020), (0.5527, 0.0013), (0.5824, 0.0062), (0.5000, 0.0007)],
    'MCC':       [(0.673, 0.001), (0.6294, 0.0098), (0.6019, 0.0018), (0.5341, 0.0041), (0.3663, 0.0046), (0.2906, 0.0016), (0.1446, 0.0089), (0.0167, 0.0027)]
}

# HLA-II 数据
hla2_models = ["Epi-UQ", "TransHLA", "MixMHC2pred", "NetMHCIIpan4.3b", "Mhcnuggets", "DeepSeqPanII"]
hla2_colors = [COLOR_OURS, COLOR_TRANSHLA, COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE]
hla2_data = {
    'AUC':       [(0.756, 0.008), (0.7029, 0.0129), (0.6611, 0.0046), (0.5632, 0.0033), (0.5296, 0.0039), (0.4971, 0.0043)],
    'F1':        [(0.607, 0.031), (0.6025, 0.0177), (0.6595, 0.0037), (0.6304, 0.0045), (0.6054, 0.0057), (0.5638, 0.0042)],
    'Precision': [(0.800, 0.021), (0.6830, 0.0129), (0.5383, 0.0038), (0.5092, 0.0063), (0.5019, 0.0071), (0.5011, 0.0062)],
    'MCC':       [(0.398, 0.009), (0.2958, 0.0235), (0.1432, 0.0057), (0.0287, 0.0084), (0.0006, 0.0114), (-0.0009, 0.0080)]
}

# ==========================================
# 3. 核心绘图函数
# ==========================================
def plot_cleveland_panel(axes_row, models, data, colors):
    n_models = len(models)
    y_positions = []
    current_y = n_models
    
    # 动态计算Y轴坐标，保留视觉 Gap
    for i in range(n_models):
        y_positions.append(current_y)
        current_y -= 1
        if i == 1: 
            current_y -= 0.8 
            
    for col_idx, metric in enumerate(metrics_order):
        ax = axes_row[col_idx]
        
        means = np.array([item[0] for item in data[metric]])
        stds = np.array([item[1] for item in data[metric]])
        our_mean = means[0]
        
        # 1. 绘制极简横向背景引导线
        for y in y_positions:
            ax.axhline(y, color='#EEEEEE', linestyle='--', zorder=1)
            
        # 2. 绘制贯穿的赤红色对比标尺线
        ax.axvline(our_mean, color=COLOR_OURS, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
        
        # 3. 绘制误差线与中心圆点
        ax.hlines(y_positions, means - stds, means + stds, color=colors, linewidth=2.5, zorder=3)
        ax.scatter(means, y_positions, color=colors, s=100, zorder=4, edgecolors='white', linewidth=1.5)
        
        # 4. 隐藏多余边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel(metric, fontweight='bold', fontsize=13)
        
        # 5. 动态计算 X 轴范围
        x_min, x_max = np.min(means - stds), np.max(means + stds)
        padding = (x_max - x_min) * 0.15
        if metric == 'MCC':
            ax.set_xlim(min(0.0, x_min - padding), min(1.0, x_max + padding))
        else:
            ax.set_xlim(max(0.0, x_min - padding), min(1.0, x_max + padding))
        
        # 6. 处理最左侧的一列（名字变色方案）
        ax.set_yticks(y_positions)
        if col_idx == 0:
            ax.set_yticklabels(models, fontsize=12)
            ax.tick_params(axis='y', length=0, pad=10)
            
            # ✨ 核心升级：遍历 Y 轴标签，单独赋予颜色和粗细
            yticklabels = ax.get_yticklabels()
            yticklabels[0].set_color(COLOR_OURS)
            yticklabels[0].set_fontweight('bold')
            yticklabels[1].set_color(COLOR_TRANSHLA)
            yticklabels[1].set_fontweight('bold')
            for i in range(2, len(yticklabels)):
                yticklabels[i].set_color('#666666') # 传统模型用深灰，不加粗
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            
# ==========================================
# 4. 图像拼装与全局图例
# ==========================================
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 9))

plot_cleveland_panel(axes[0], hla1_models, hla1_data, hla1_colors)
plot_cleveland_panel(axes[1], hla2_models, hla2_data, hla2_colors)

# ✨ 核心升级：添加全局精美图例 (Global Legend)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Epi-UQ (Allele-agnostic)', markerfacecolor=COLOR_OURS, markersize=12),
    Line2D([0], [0], marker='o', color='w', label='TransHLA (Allele-agnostic)', markerfacecolor=COLOR_TRANSHLA, markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Traditional Baselines (Allele-dependent)', markerfacecolor=COLOR_BASELINE, markersize=12)
]
# 修改 1：将 bbox_to_anchor 的 Y 值从 1.05 下调至 1.00，使图例更贴近顶部边缘
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.00),
           ncol=3, frameon=False, fontsize=13, prop={'weight': 'bold'})

# 修改 2：调整画布边缘
# top=0.90 (原0.88) -> 图表向上延伸
# hspace=0.28 (原0.35) -> 减少上下两行图表之间的空白，使整体更紧凑
plt.subplots_adjust(left=0.08, top=0.90, hspace=0.28)

# 修改 3：调整大标题位置，防止与下移的图例重叠
# 标题 A 下移至 0.94 (原0.92)
fig.text(0.01, 0.94, 'A. HLA-I Cross-source Generalization', fontsize=16, fontweight='bold')
# 标题 B 保持或微调，这里维持原样或稍作移动以保持视觉平衡
fig.text(0.01, 0.46, 'B. HLA-II Cross-source Generalization', fontsize=16, fontweight='bold')

# ... 后续的 savefig 和 show 代码保持不变 ...
plt.savefig("plot_External_1.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.savefig("plot_External_1.png", format='png', dpi=600, bbox_inches='tight')
plt.tight_layout() # 注意：tight_layout 有时会覆盖 subplots_adjust，如果在屏幕上显示不理想，可注释掉这行，主要依赖 savefig 的 bbox_inches='tight'
print("✅ 极简风增强版图表已生成！")
plt.show()