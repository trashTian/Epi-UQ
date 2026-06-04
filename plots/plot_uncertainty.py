import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from tqdm import tqdm

# ==========================================
# 1. 路径与全局配置
# ==========================================
ESM_MODEL_PATH = "/data1/gpj/LLMModels/models/facebook/esm2_t33_650M_UR50D"
MODEL_WEIGHTS = [
    "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora/lora_cnn_attention_hlai_esm650_seed42.pth",
    "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora/lora_cnn_attention_hlai_esm650_seed921.pth",
    "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora/lora_cnn_attention_hlai_esm650_seed2026.pth"
]
TEST_CSV = "/data1/gpj/AIDD/hla_epitope/data/HLA_I_external_4_time_negative.csv"
OUTPUT_DIR = "./Figures_Final_Paper"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
MAX_LEN = 30

# ★ 修改：进一步缩小字体，防止标签重叠
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "axes.linewidth": 1.0,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

COLOR_MCC = '#3C5488'
COLOR_SPEC = '#4DBBD5'
COLOR_FP = '#E64B35'

# ==========================================
# 2. 数据集与模型定义 (保持原样)
# ==========================================
class EpitopeDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        pep = self.data.iloc[idx]['Epitope.1']
        lbl = float(self.data.iloc[idx]['Label'])
        enc = self.tokenizer(pep, add_special_tokens=True, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        return {'pep': pep, 'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'label': lbl}

class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention_weights = nn.Linear(in_features, 1)
    def forward(self, x, mask):
        attn_scores = self.attention_weights(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

class EpiAdvanced_Model(nn.Module):
    def __init__(self, esm_model_path):
        super().__init__()
        base_esm = EsmModel.from_pretrained(esm_model_path, local_files_only=True)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=True,
                                 r=32, lora_alpha=64, target_modules=["query", "key", "value"], bias="none")
        self.esm = get_peft_model(base_esm, peft_config)
        hidden_size = self.esm.config.hidden_size
        self.conv1d = nn.Sequential(nn.Conv1d(hidden_size, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2))
        self.attn_pool = AttentionPooling(512)
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        seq_features = self.conv1d(outputs.last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
        return self.classifier(self.attn_pool(seq_features, attention_mask)).squeeze(-1)

# ==========================================
# 3. 推理与集成预测 (保持原样)
# ==========================================
def run_ensemble_inference():
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    dataset = EpitopeDataset(TEST_CSV, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    all_probs = []
    labels = []
    for i, weight_path in enumerate(MODEL_WEIGHTS):
        print(f"[*] Running Inference for Model {i+1}/3")
        model = EpiAdvanced_Model(ESM_MODEL_PATH).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
        model.eval()
        preds, labels_current = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                ids, mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
                prob = torch.sigmoid(model(ids, mask)).cpu().numpy()
                preds.extend(prob)
                if i == 0: labels_current.extend(batch['label'].numpy())
        all_probs.append(preds)
        if i == 0: labels = labels_current
        del model
        torch.cuda.empty_cache()
        gc.collect()
    return pd.DataFrame({'Label': labels, 'Prob_Mean': np.mean(all_probs, axis=0), 'Uncertainty': np.std(all_probs, axis=0)})

# ==========================================
# 4. 终极绘图：解决重叠与图例问题
# ==========================================
def plot_uncertainty_panels(df_ens):
    # figsize 保持单栏宽度 7.2 inches
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    
    df_sorted_desc = df_ens.sort_values(by="Uncertainty", ascending=False).reset_index(drop=True)
    df_sorted_asc = df_ens.sort_values(by="Uncertainty", ascending=True).reset_index(drop=True)
    total_samples = len(df_ens)
    
    # ------------------------------------------
    # Panel A: Accuracy-Rejection Analysis (左图)
    # ------------------------------------------
    ax1 = axes[0]
    rejection_rates = np.arange(0, 51, 5)
    mccs, specs = [], []
    for rr in rejection_rates:
        drop_count = int(total_samples * (rr / 100.0))
        df_accepted = df_sorted_desc.iloc[drop_count:]
        y_true, y_pred = df_accepted['Label'].values, (df_accepted['Prob_Mean'].values > 0.5).astype(int)
        mccs.append(matthews_corrcoef(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        
    ax1.set_title("A. Accuracy-Rejection Analysis", loc='left', fontweight='bold', fontsize=10, pad=8)
    
    # ★ 修改：缩短X轴和Y轴标签，防止与B图重叠
    ax1.set_xlabel('Rejection Rate (%)', fontweight='bold')
    ax1.set_ylabel('MCC', color=COLOR_MCC, fontweight='bold')
    
    line1 = ax1.plot(rejection_rates, mccs, marker='o', color=COLOR_MCC, label='MCC', linewidth=2.0, markersize=5)
    ax1.tick_params(axis='y', labelcolor=COLOR_MCC)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Specificity', color=COLOR_SPEC, fontweight='bold') # 缩短标签
    line2 = ax2.plot(rejection_rates, specs, marker='s', color=COLOR_SPEC, label='Specificity', linewidth=2.0, markersize=5)
    ax2.tick_params(axis='y', labelcolor=COLOR_SPEC)
    
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # 图例放在右下角
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=False, fontsize=8)
    
    # ------------------------------------------
    # Panel B: Risk Stratification (右图)
    # ------------------------------------------
    ax3 = axes[1]
    percentiles = [100, 75, 50, 25]
    mccs_bar, fps_bar = [], []
    for p in percentiles:
        df_subset = df_sorted_asc.head(int(total_samples * (p / 100.0)))
        y_true, y_pred = df_subset['Label'].values, (df_subset['Prob_Mean'].values > 0.5).astype(int)
        mccs_bar.append(matthews_corrcoef(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fps_bar.append(fp)
        
    x = np.arange(len(percentiles))
    width = 0.35
    
    ax3.set_title("B. Uncertainty-Guided Risk Stratification", loc='left', fontweight='bold', fontsize=10, pad=8)
    
    # ★ 修改：缩短X轴标签
    ax3.set_xlabel("Confidence Cohort", fontweight='bold')
    ax3.set_ylabel('MCC', color=COLOR_MCC, fontweight='bold') # 缩短标签
    
    rects1 = ax3.bar(x - width/2, mccs_bar, width, label='MCC', color=COLOR_MCC, edgecolor='none')
    ax3.tick_params(axis='y', labelcolor=COLOR_MCC)
    ax3.set_ylim(min(mccs_bar)*0.85, max(mccs_bar)*1.15)
    
    ax4 = ax3.twinx()
    rects2 = ax4.bar(x + width/2, fps_bar, width, label='False Positives', color=COLOR_FP, edgecolor='none')
    ax4.set_ylabel('False Positives', color=COLOR_FP, fontweight='bold') # 缩短标签
    ax4.tick_params(axis='y', labelcolor=COLOR_FP)
    ax4.set_ylim(0, max(fps_bar)*1.15)
    
    # ★ 修改：X轴标签分行显示，防止拥挤
    xtick_labels = ["100%\n(Baseline)", "Top 75%\nConfident", "Top 50%\nConfident", "Top 25%\nConfident"]
    ax3.set_xticks(x)
    ax3.set_xticklabels(xtick_labels, fontsize=7.5) 
    
    ax3.bar_label(rects1, fmt='%.2f', padding=2, fontsize=7, color=COLOR_MCC, fontweight='bold')
    ax4.bar_label(rects2, labels=[f"{val:,}" for val in fps_bar], padding=2, fontsize=7, color=COLOR_FP, fontweight='bold')
    
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    
    # ★ 核心修改：将图例移入图内右上角，添加半透明背景防止遮挡数据
    lines_b1, labels_b1 = ax3.get_legend_handles_labels()
    lines_b2, labels_b2 = ax4.get_legend_handles_labels()
    ax3.legend(lines_b1 + lines_b2, labels_b1 + labels_b2, 
               loc='upper right', frameon=True, framealpha=0.8, 
               edgecolor='none', fontsize=8)
    
    # ★ 修改：增加 wspace 让左右图分开，减少底部边距
    fig.subplots_adjust(wspace=0.45, bottom=0.15, top=0.92)
    
    out_path = os.path.join(OUTPUT_DIR, "Fig5_Uncertainty_Analysis_1x2.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    print(f"[*] Saved Fixed Figure -> {out_path}")

# ==========================================
# 5. 主执行入口
# ==========================================
if __name__ == "__main__":
    print("Starting Pipeline...")
    df_ensemble = run_ensemble_inference()
    print("Generating Figure...")
    plot_uncertainty_panels(df_ensemble)
    print("Done!")
