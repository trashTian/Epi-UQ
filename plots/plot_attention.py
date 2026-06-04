import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ==========================================
# 1. 路径与配置
# ==========================================
ESM_MODEL_PATH = "/data1/gpj/LLMModels/models/facebook/esm2_t33_650M_UR50D"

HLA_I_MODEL_PATH = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora/lora_cnn_attention_hlai_esm650_seed921.pth"
HLA_II_MODEL_PATH = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora/lora_cnn_attention_hlaii_esm650_seed921.pth"

HLA_I_CSV = "/data1/gpj/AIDD/hla_epitope/data/HLA_I_epitope_train_shuffle.csv"
HLA_II_CSV = "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_train_shuffle.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FIG = "./Figure5_Attention_Heatmaps_NatureStyle.pdf"

BATCH_SIZE = 4096  

# Nature 期刊全局字体配置
plt.rcParams.update({
    "font.family": "sans-serif", 
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13, 
    "axes.titlesize": 15, 
    "axes.labelsize": 13
})

# ==========================================
# 2. 模型架构 (保持不变)
# ==========================================
class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention_weights = nn.Linear(in_features, 1)
        
    def forward(self, x, mask):
        attn_scores = self.attention_weights(x).squeeze(-1).masked_fill(mask == 0, -1e4)
        return F.softmax(attn_scores, dim=-1)

class EpiAdvanced_Model(nn.Module):
    def __init__(self, esm_model_path, r, lora_alpha):
        super().__init__()
        base_esm = EsmModel.from_pretrained(esm_model_path, local_files_only=True)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=True, 
                                 r=r, lora_alpha=lora_alpha, target_modules=["query", "key", "value"], bias="none")
        self.esm = get_peft_model(base_esm, peft_config)
        self.conv1d = nn.Sequential(nn.Conv1d(1280, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2))
        self.attn_pool = AttentionPooling(512)
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        
    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids, attention_mask=attention_mask)
        seq = self.conv1d(out.last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
        attn_weights = self.attn_pool(seq, attention_mask)
        pooled_rep = torch.bmm(attn_weights.unsqueeze(1), seq).squeeze(1)
        logits = self.classifier(pooled_rep).squeeze(-1)
        return logits, attn_weights

# ==========================================
# 3. 抽取策略 (保持不变)
# ==========================================
def extract_optimized_attention(model_path, csv_path, tokenizer, r, alpha, target_len, num_samples=6, is_hla_i=True):
    print(f"\n-> Loading model for {target_len}-mers...")
    base_model = EpiAdvanced_Model(ESM_MODEL_PATH, r, alpha)
    base_model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model).to(DEVICE)
    else:
        model = base_model.to(DEVICE)
    model.eval()

    df = pd.read_csv(csv_path)
    pos_df = df[df['Label'] == 1].copy()
    pos_df['Len'] = pos_df['Epitope.1'].str.len()
    
    available_peptides = pos_df[pos_df['Len'] == target_len]['Epitope.1'].tolist()
    if len(available_peptides) < num_samples:
        target_len = int(pos_df['Len'].mode()[0])
        available_peptides = pos_df[pos_df['Len'] == target_len]['Epitope.1'].tolist()

    pep_scores, pep_attns = [], []
    
    print(f"-> Batch inferencing {len(available_peptides)} sequences...")
    for i in tqdm(range(0, len(available_peptides), BATCH_SIZE), desc="Inferencing", leave=False):
        batch_peps = available_peptides[i:i+BATCH_SIZE]
        enc = tokenizer(batch_peps, padding='max_length', truncation=True, max_length=target_len+2, return_tensors='pt')
        ids, mask = enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits, attn_w = model(ids, mask)
            scores = torch.sigmoid(logits).cpu().numpy()
            attns = attn_w.cpu().numpy()
            
        pep_scores.extend(scores)
        for j in range(len(batch_peps)):
            pep_attns.append(attns[j, 1:target_len+1]) 

    results = list(zip(available_peptides, pep_scores, pep_attns))
    results.sort(key=lambda x: x[1], reverse=True) 
    
    if is_hla_i:
        final_sort = results[:num_samples]
    else:
        top_50 = results[:50]
        def center_of_mass(attn):
            positions = np.arange(len(attn))
            return np.sum(positions * attn) / (np.sum(attn) + 1e-9)
        top_50.sort(key=lambda x: center_of_mass(x[2]))
        idx = np.linspace(0, len(top_50)-1, num_samples).astype(int)
        final_sort = [top_50[i] for i in idx]

    final_lbls = [x[0] for x in final_sort]
    final_mats = np.array([x[2] for x in final_sort])
            
    del model, base_model
    torch.cuda.empty_cache()
    
    return final_mats, final_lbls, target_len


# ==========================================
# 4. ★ 全新优化的顶级排版绘图流程 (完美解决遮挡版)
# ==========================================
def create_conditional_annotations(matrix, threshold=0.05):
    """低于阈值的权重显示为空白，极大地提升数据墨水比"""
    annot = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val >= threshold:
                annot[i, j] = f"{val:.2f}"
            else:
                annot[i, j] = ""
    return annot

def main():
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    
    mat_i, lbl_i, len_i = extract_optimized_attention(HLA_I_MODEL_PATH, HLA_I_CSV, tokenizer, r=32, alpha=64, target_len=9, is_hla_i=True)
    mat_ii, lbl_ii, len_ii = extract_optimized_attention(HLA_II_MODEL_PATH, HLA_II_CSV, tokenizer, r=32, alpha=64, target_len=15, is_hla_i=False)

    print("\n[*] Plotting Nature-style heatmaps...")
    
    fig = plt.figure(figsize=(22, 6.5))
    
    # [优化 1] wspace=0.45 进一步拉开两张图的安全距离
    gs = GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.45)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    annot_i = create_conditional_annotations(mat_i)
    annot_ii = create_conditional_annotations(mat_ii)

    # --- 绘制 HLA-I (左侧) ---
    # [优化 2] shrink=0.58 使 Colorbar 与热力图等高，pad=0.06 将其向右推移避开文字
    sns.heatmap(mat_i, annot=annot_i, fmt="", cmap="Reds", ax=ax1, 
                square=True, vmin=0.0, vmax=1.0, 
                linewidths=1.0, linecolor='white',
                cbar_kws={'label': 'Attention Weight', 'shrink': 0.58, 'pad': 0.06},
                annot_kws={"size": 11, "fontweight": "bold"})
    
    # [优化 3] 稍微降低字号为 17，增加 pad=25 让标题悬浮得更高
    ax1.set_title("A. HLA-I C-Terminal Anchor Dominance", fontweight="bold", fontsize=17, loc='left', pad=25)
    ax1.set_xlabel("Peptide Position", fontweight="bold", labelpad=10)
    ax1.set_yticklabels(lbl_i, rotation=0, fontsize=12)
    ax1.set_xticklabels([f"P{i}" for i in range(1, len_i + 1)], fontsize=12)
    
    ax1.tick_params(axis='both', length=0, pad=8)
    
    # 生物物理锚点高亮
    for tick_label in ax1.get_xticklabels():
        if tick_label.get_text() in ['P2', 'P9']:
            tick_label.set_color('#E64B35') 
            tick_label.set_fontweight('bold')

    # --- 绘制 HLA-II (右侧) ---
    # 同步应用 shrink=0.58 和 pad=0.06 保持视觉对称
    sns.heatmap(mat_ii, annot=annot_ii, fmt="", cmap="Blues", ax=ax2, 
                square=True, vmin=0.0, vmax=1.0, 
                linewidths=1.0, linecolor='white',
                cbar_kws={'label': 'Attention Weight', 'shrink': 0.58, 'pad': 0.06},
                annot_kws={"size": 11, "fontweight": "bold"})
    
    ax2.set_title("B. HLA-II Sliding Core Anchor Detection", fontweight="bold", fontsize=17, loc='left', pad=25)
    ax2.set_xlabel("Peptide Position", fontweight="bold", labelpad=10)
    ax2.set_yticklabels(lbl_ii, rotation=0, fontsize=12)
    ax2.set_xticklabels([f"P{i}" for i in range(1, len_ii + 1)], fontsize=12)
    
    ax2.tick_params(axis='both', length=0, pad=8)

    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(OUTPUT_FIG.replace(".pdf", ".png"), dpi=600, bbox_inches='tight', transparent=True)
    
    print(f"[✅] Perfect Heatmap saved to {OUTPUT_FIG}!")
    plt.show()

if __name__ == "__main__":
    main()
