import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm
import gc

# ==========================================
# 1. 路径与配置 
# ==========================================
ESM_MODEL_PATH = "/data1/gpj/LLMModels/models/facebook/esm2_t33_650M_UR50D"
ABLATION_DIR = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_ablation"
OURS_DIR = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora" 

TEST_CSV_DICT = {
    "HLA-I": "/data1/gpj/AIDD/hla_epitope/data/HLA_I_epitope_test.csv",
    "HLA-II": "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_test.csv"
}

SEEDS = [42, 921, 2026]
BATCH_SIZE = 512 # 多卡并行推理，Batch Size 可以放心调大
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 模型定义 (包含 GAP, MLP, CNN+Attn)
# ==========================================
class EpitopeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        pep = self.data.iloc[idx]['Epitope.1']
        lbl = float(self.data.iloc[idx]['Label'])
        enc = self.tokenizer(pep, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'label': lbl}

class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention_weights = nn.Linear(in_features, 1)
    def forward(self, x, mask):
        attn_scores = self.attention_weights(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return torch.bmm(F.softmax(attn_scores, dim=-1).unsqueeze(1), x).squeeze(1)

def get_ablation_model(config_name, esm_path, lora_r=32, lora_alpha=64):
    base_esm = EsmModel.from_pretrained(esm_path, local_files_only=True)
    hidden_size = base_esm.config.hidden_size 
    
    class AblationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config_name = config_name
            if "Frozen" in config_name:
                self.esm = base_esm
            else:
                peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=True, 
                                         r=lora_r, lora_alpha=lora_alpha, target_modules=["query", "key", "value"], bias="none")
                self.esm = get_peft_model(base_esm, peft_config)

            if "MLP" in config_name:
                self.classifier = nn.Sequential(nn.Linear(hidden_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 1))
            elif "GAP" in config_name:
                self.conv1d = nn.Sequential(nn.Conv1d(hidden_size, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2))
                # GAP 没有 Attention Pooling
                self.classifier = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
            else: # CNN + Attn
                self.conv1d = nn.Sequential(nn.Conv1d(hidden_size, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2))
                self.attn_pool = AttentionPooling(512)
                self.classifier = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

        def forward(self, input_ids, attention_mask):
            outputs = self.esm(input_ids, attention_mask=attention_mask)
            if "MLP" in self.config_name:
                return self.classifier(outputs.last_hidden_state[:, 0, :]).squeeze(-1)
            elif "GAP" in self.config_name:
                seq = self.conv1d(outputs.last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
                # --- Masked Global Average Pooling 逻辑 ---
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_seq = torch.sum(seq * mask_expanded, dim=1)
                sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
                pooled_rep = sum_seq / sum_mask
                return self.classifier(pooled_rep).squeeze(-1)
            else: # CNN + Attn
                seq = self.conv1d(outputs.last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
                return self.classifier(self.attn_pool(seq, attention_mask)).squeeze(-1)
                
    return AblationModel()

# ==========================================
# 3. 指标计算与主推断逻辑
# ==========================================
def calculate_metrics(labels, preds_prob):
    labels, preds_prob = np.array(labels), np.array(preds_prob)
    y_pred = (preds_prob > 0.5).astype(int)
    
    auc = roc_auc_score(labels, preds_prob)
    auprc = average_precision_score(labels, preds_prob)
    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, zero_division=0)
    rec = recall_score(labels, y_pred, zero_division=0)
    prec = precision_score(labels, y_pred, zero_division=0)
    mcc = matthews_corrcoef(labels, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return [auc, auprc, acc, f1, rec, prec, spec, mcc]

def main():
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    
    # 完整的 5 种消融配置 (包含全新的 GAP 基线和我们的最终架构)
    CONFIGS = [
        ("Frozen_MLP", ABLATION_DIR, "{task}_Frozen_MLP_seed{seed}.pth"),
        ("Frozen_CNN_Attn", ABLATION_DIR, "{task}_Frozen_CNN_Attn_seed{seed}.pth"),
        ("LoRA_MLP", ABLATION_DIR, "{task}_LoRA_MLP_seed{seed}.pth"),
        ("LoRA_CNN_GAP", ABLATION_DIR, "{task}_LoRA_CNN_GAP_seed{seed}.pth"),
        # ("LoRA_CNN_Attn", OURS_DIR, "lora_cnn_attention_{task_lower}_esm650_seed{seed}.pth") 
    ]
    
    all_results = []
    
    for task in ["HLA-I", "HLA-II"]:
        max_len = 30 if task == "HLA-I" else 35
        task_lower = task.lower().replace("-", "")
        test_csv = TEST_CSV_DICT[task]
        
        dataset = EpitopeDataset(test_csv, tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        
        for cfg_name, base_dir, path_template in CONFIGS:
            metrics_seeds = []
            for seed in SEEDS:
                # 确定模型权重路径
                weight_file = path_template.format(task=task, task_lower=task_lower, seed=seed)
                weight_path = os.path.join(base_dir, weight_file)
                
                # 兼容 Ours 特殊无 seed 后缀的情况
                if not os.path.exists(weight_path) and "LoRA_CNN_Attn" in cfg_name:
                    weight_path = os.path.join(base_dir, f"lora_cnn_attention_{task_lower}_esm650.pth")

                if not os.path.exists(weight_path):
                    print(f"⚠️ Missing: {weight_path}")
                    continue
                
                # 加载权重与多卡并行处理 (DataParallel)
                model = get_ablation_model(cfg_name, ESM_MODEL_PATH)
                state_dict = torch.load(weight_path, map_location="cpu")
                
                # 剔除可能存在的 'module.' 前缀 (针对多卡训练保存的权重)
                clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(clean_state_dict, strict=False)
                
                # 启用多卡推理
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(DEVICE)
                model.eval()
                
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=f"{task} | {cfg_name} | Seed {seed}", leave=False):
                        logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
                        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                        all_labels.extend(batch['label'].numpy())
                
                metrics = calculate_metrics(all_labels, all_preds)
                metrics_seeds.append(metrics)
                
                # 内存回收
                del model, state_dict, clean_state_dict
                torch.cuda.empty_cache()
                gc.collect()
            
            if metrics_seeds:
                # 计算 Mean 和 Std
                metrics_arr = np.array(metrics_seeds)
                mean_metrics = np.mean(metrics_arr, axis=0)
                std_metrics = np.std(metrics_arr, axis=0)
                
                res = {"Task": task, "Architecture": cfg_name}
                metric_names = ["AUC", "AUPRC", "ACC", "F1", "Recall", "Precision", "Specificity", "MCC"]
                for i, name in enumerate(metric_names):
                    res[name] = f"{mean_metrics[i]:.3f} ± {std_metrics[i]:.3f}"
                all_results.append(res)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    df.to_csv("ablation_results_test.csv", index=False)
    
    # 打印优美的 LaTeX 表格
    print("\n" + "="*80)
    print(" 🌟 LaTeX Table Code ")
    print("="*80)
    for task in ["HLA-I", "HLA-II"]:
        print(f"\n% Ablation Results for {task}")
        task_df = df[df['Task'] == task]
        for _, row in task_df.iterrows():
            arch_name = row['Architecture'].replace("_", "\\_")
            print(f"{arch_name:25} & {row['AUC']} & {row['AUPRC']} & {row['MCC']} & {row['Specificity']} & {row['Recall']} \\\\")

if __name__ == "__main__":
    main()
