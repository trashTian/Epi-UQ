import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# ==========================================
# 1. 适配单栏的紧凑全局配置
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
# 进一步缩小字体，适配窄子图
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

# NPG 经典配色
COLORS = {
    "Epi-UQ": "#E64B35",
    "TransHLA": "#4DBBD5",
    "Mhcflurry": "#00A087",
    "Mhcnuggets": "#3C5488",
    "TransPHLA": "#F39B7F",
    "Anthem": "#8491B4",
    "NetMHCPan4.1b": "#91D1C2",
    "MixMHCpred": "#7E6148",
    "DeepSeqPanII": "#B09C85",
    "NetMHCIIpan4.3b": "#4E596F",
    "MixMHC2pred": "#D4A373"
}

# ==========================================
# 2. 实验数据输入
# ==========================================
hla1_data = {
    "Epi-UQ":      [0.643, 0.023, 0.911, 0.002, "Agnostic"],
    "TransHLA":       [0.5425, 0.0382, 0.8890, 0.0030, "Agnostic"],
    "Mhcflurry":      [0.4067, 0.0023, 0.9014, 0.0011, "Dependent"],
    "NetMHCPan4.1b":  [0.4446, 0.0036, 0.8414, 0.0014, "Dependent"],
    "MixMHCpred":     [0.2614, 0.0021, 0.8872, 0.0011, "Dependent"],
    "Anthem":         [0.2368, 0.0019, 0.8610, 0.0012, "Dependent"],
    "TransPHLA":      [0.2557, 0.0023, 0.5760, 0.0033, "Dependent"],
    "Mhcnuggets":     [0.2007, 0.0018, 0.6154, 0.0037, "Dependent"]
}
hla2_data = {
    "Epi-UQ":      [0.510, 0.038, 0.758, 0.008, "Agnostic"],
    "TransHLA":       [0.3358, 0.0126, 0.6884, 0.0145, "Agnostic"],
    "MixMHC2pred":    [0.2218, 0.0022, 0.6561, 0.0043, "Dependent"],
    "NetMHCIIpan4.3b":[0.2005, 0.0017, 0.5581, 0.0035, "Dependent"],
    "Mhcnuggets":     [0.1970, 0.0023, 0.5225, 0.0072, "Dependent"],
    "DeepSeqPanII":   [0.1969, 0.0027, 0.4905, 0.0059, "Dependent"]
}

# ==========================================
# 3. 文本避让排版字典（针对1行2列重新调整）
# ==========================================
TEXT_OFFSETS_HLA1 = {
    "Epi-UQ": (0, -12, 'center'),     # 正下方，避开右上角标题
    "TransHLA": (-5, -10, 'right'),      # 左下方，避开Our model
    "Mhcflurry": (8, 0, 'left'),         # 右侧
    "NetMHCPan4.1b": (8, -6, 'left'),    # 右下方
    "MixMHCpred": (8, 4, 'left'),        # 右上方
    "Anthem": (8, -4, 'left'),           # 右下方
    "TransPHLA": (8, 0, 'left'),
    "Mhcnuggets": (8, 0, 'left')
}
TEXT_OFFSETS_HLA2 = {
    "Epi-UQ": (0, -12, 'center'),     # 正下方，避开右上角标题
    "TransHLA": (8, 0, 'left'),
    "MixMHC2pred": (8, 0, 'left'),
    "NetMHCIIpan4.3b": (8, 0, 'left'),
    "Mhcnuggets": (8, 0, 'left'),
    "DeepSeqPanII": (8, 0, 'left')
}

# ==========================================
# 4. 核心绘图函数
# ==========================================
def plot_crosshair_scatter(ax, data, title, is_hla1=True):
    xlim = (0.15, 0.70) if is_hla1 else (0.15, 0.60)
    ylim = (0.55, 0.95) if is_hla1 else (0.45, 0.80)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    prec_safe_threshold = 0.50 if is_hla1 else 0.40

    # 陷阱区 (红色)
    rect_trap = patches.Rectangle(
        (xlim[0], ylim[0]), prec_safe_threshold - xlim[0], ylim[1] - ylim[0],
        linewidth=0, facecolor='#FF0000', alpha=0.06, zorder=0)
    ax.add_patch(rect_trap)
    # ★ 关键修改：文字换行 (\n)，字体缩小至8，位置微调
    ax.text(
        xlim[0] + (prec_safe_threshold - xlim[0]) / 2,
        ylim[1] - (ylim[1] - ylim[0]) * 0.005,
        "Low Precision\nRegime",
        ha='center', va='top', fontsize=8, color='#B22222',
        fontweight='bold', zorder=1, linespacing=1.2)

    # 安全区 (绿色)
    rect_safe = patches.Rectangle(
        (prec_safe_threshold, ylim[0]), xlim[1] - prec_safe_threshold, ylim[1] - ylim[0],
        linewidth=0, facecolor='#008000', alpha=0.06, zorder=0)
    ax.add_patch(rect_safe)
    ax.text(
        prec_safe_threshold + (xlim[1] - prec_safe_threshold) / 2,
        ylim[1] - (ylim[1] - ylim[0]) * 0.005,
        "High Precision\nRegime",
        ha='center', va='top', fontsize=8, color='#006400',
        fontweight='bold', zorder=1, linespacing=1.2)

    # 随机猜测线
    ax.axvline(x=0.2, color='#888888', linestyle='--', linewidth=1.2, zorder=1)
    ax.text(
        0.193, ylim[0] + (ylim[1] - ylim[0]) * 0.12,
        'Random Guess\nBaseline (0.2)',
        rotation=90, va='bottom', ha='right',
        color='#888888', fontsize=7, fontweight='bold', linespacing=1.2)

    offsets_dict = TEXT_OFFSETS_HLA1 if is_hla1 else TEXT_OFFSETS_HLA2

    for model, values in data.items():
        prec_m, prec_s, auc_m, auc_s, category = values
        marker_style = 'o' if category == "Agnostic" else 's'
        size = 100 if model == "Epi-UQ" else 60
        z_ord = 5 if model == "Epi-UQ" else 4

        ax.errorbar(
            prec_m, auc_m, xerr=prec_s, yerr=auc_s,
            fmt=marker_style, color=COLORS[model],
            ecolor=COLORS[model], elinewidth=1.2, capsize=0,
            markersize=size / 10,
            markeredgecolor='white', markeredgewidth=1.0,
            zorder=z_ord, label=model)

        x_off, y_off, h_align = offsets_dict[model]
        font_weight = 'bold' if model == "Epi-UQ" else 'medium'
        ax.annotate(
            model, (prec_m, auc_m),
            xytext=(x_off, y_off), textcoords='offset points',
            ha=h_align, va='center' if y_off == 0 else 'top',
            fontsize=7.5, fontweight=font_weight,
            color='#333333', zorder=6)

    ax.set_xlabel('Precision (Suppressing False Positives)', fontweight='bold', labelpad=4)
    ax.set_ylabel('AUC (Overall Discrimination)', fontweight='bold', labelpad=2)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==========================================
# 5. 图像拼装：1行2列，适配单栏宽度
# ==========================================
# Bioinformatics 单栏宽度 ≈ 18.3 cm ≈ 7.2 inches
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))

plot_crosshair_scatter(axes[0], hla1_data, "A. HLA-I", is_hla1=True)
plot_crosshair_scatter(axes[1], hla2_data, "B. HLA-II", is_hla1=False)

# 调整子图间距，防止标签重叠
fig.subplots_adjust(wspace=0.35, hspace=0.1, top=0.92, bottom=0.22)

# 图例：放在图下方，4列排布（更紧凑）
handles1, labels1 = axes[0].get_legend_handles_labels()
handles2, labels2 = axes[1].get_legend_handles_labels()
handles_all = handles1 + handles2
labels_all = labels1 + labels2
by_label = dict(zip(labels_all, handles_all))

order = [
    "Epi-UQ", "TransHLA", "Mhcflurry", "NetMHCPan4.1b",
    "MixMHCpred", "Anthem", "TransPHLA", "Mhcnuggets",
    "MixMHC2pred", "NetMHCIIpan4.3b", "DeepSeqPanII"
]
ordered_handles = [by_label[m] for m in order if m in by_label]
ordered_labels = [m for m in order if m in by_label]

fig.legend(
    ordered_handles, ordered_labels,
    loc='lower center', ncol=4,
    bbox_to_anchor=(0.5, 0.0),
    frameon=False, fontsize=7.5,
    columnspacing=1.2, handletextpad=0.4)

# 保存图像
plt.savefig("plot_External_4.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.savefig("plot_External_4.png", format='png', dpi=600, bbox_inches='tight')
print("1行2列布局完成！文字重叠已解决。")
plt.show()
