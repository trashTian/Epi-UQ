import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import random
import gc

# ==========================================
# 1. 路径与全局配置
# ==========================================
ESM_MODEL_PATH = "/data1/gpj/LLMModels/models/facebook/esm2_t33_650M_UR50D"

# 数据集路径
DATASETS = {
    "HLA-I": {
        "train": "/data1/gpj/AIDD/hla_epitope/data/HLA_I_epitope_train_shuffle.csv",
        "val": "/data1/gpj/AIDD/hla_epitope/data/HLA_I_epitope_validation.csv",
        "max_len": 30, "lora_r": 32, "lora_alpha": 64
    },
    "HLA-II": {
        "train": "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_train_shuffle.csv",
        "val": "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_validation.csv",
        "max_len": 35, "lora_r": 32, "lora_alpha": 64
    }
}

SAVE_DIR = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_ablation"
os.makedirs(SAVE_DIR, exist_ok=True)

# 专属日志文件
LOG_FILE = os.path.join(SAVE_DIR, "training_ablation_LoRA_CNN_GAP.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'), logging.StreamHandler()])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
SEEDS = [42] # 依然跑3个种子以确保严谨921, 2026
LEARNING_RATE = 2e-4
BATCH_SIZE = 256 # LoRA 模型保守使用 256

# ==========================================
# 2. 随机种子与 Dataset
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class EpitopeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        pep = self.data.iloc[idx]['Epitope.1']
        lbl = float(self.data.iloc[idx]['Label'])
        enc = self.tokenizer(pep, add_special_tokens=True, padding='max_length',
                             truncation=True, max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'label': torch.tensor(lbl, dtype=torch.float)}

# ==========================================
# 3. 专属结构：LoRA ESM + CNN + Global Average Pooling
# ==========================================
def get_lora_cnn_gap_model(esm_path, task_info):
    base_esm = EsmModel.from_pretrained(esm_path, local_files_only=True)
    hidden_size = base_esm.config.hidden_size # 1280
    
    class AblationGAPModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 1. 注入 LoRA
            peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
                                     r=task_info['lora_r'], lora_alpha=task_info['lora_alpha'], 
                                     target_modules=["query", "key", "value"], lora_dropout=0.2, bias="none")
            self.esm = get_peft_model(base_esm, peft_config)

            # 2. 1D-CNN 特征提取
            self.conv1d = nn.Sequential(
                nn.Conv1d(hidden_size, 512, 3, padding=1), 
                nn.BatchNorm1d(512), 
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            
            # 3. 移除 Attention Pooling，直接接分类头
            self.classifier = nn.Sequential(
                nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.esm(input_ids, attention_mask=attention_mask)
            
            # Conv1d 提取: [Batch, Channels, SeqLen] -> [Batch, SeqLen, Channels]
            seq = self.conv1d(outputs.last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
            
            # --- 核心：Masked Global Average Pooling ---
            # 扩展 mask 的维度以匹配特征: [Batch, SeqLen, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            
            # 仅对非 padding 的氨基酸特征进行求和
            sum_seq = torch.sum(seq * mask_expanded, dim=1)
            
            # 统计实际有效的氨基酸长度 (防止除以 0)
            sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
            
            # 求得平均特征 (等价于没有权重的硬平均)
            pooled_rep = sum_seq / sum_mask
            
            return self.classifier(pooled_rep).squeeze(-1)
                
    return AblationGAPModel()

# ==========================================
# 4. 训练与验证引擎
# ==========================================
def evaluate(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels =[], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return roc_auc_score(all_labels, all_preds)

def train_model(task_name, seed):
    config_name = "LoRA_CNN_GAP"
    seed_everything(seed)
    logging.info(f"\n{'='*50}\n🚀 TASK: {task_name} | CONFIG: {config_name} | SEED: {seed}\n{'='*50}")
    
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    task_info = DATASETS[task_name]
    
    train_dl = DataLoader(EpitopeDataset(task_info['train'], tokenizer, task_info['max_len']), 
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_dl   = DataLoader(EpitopeDataset(task_info['val'], tokenizer, task_info['max_len']), 
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    # 修改点：初始化带有 GAP 的消融模型
    model = get_lora_cnn_gap_model(ESM_MODEL_PATH, task_info)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
    else:
        model = model.to(DEVICE)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    best_auc = 0.0
    model_save_path = os.path.join(SAVE_DIR, f"{task_name}_{config_name}_seed{seed}.pth")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loop = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch in loop:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            loss = criterion(logits, batch['label'].to(DEVICE))
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
        val_auc = evaluate(model, val_dl, criterion)
        logging.info(f"Epoch {epoch} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), model_save_path)
            logging.info(f"[✅] Checkpoint saved! Best AUC: {best_auc:.4f}")

    del model, optimizer, train_dl, val_dl
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 5. 自动启动
# ==========================================
if __name__ == "__main__":
    # 只跑这一次额外添加的消融实验  "HLA-I", 
    for task in ["HLA-II"]:
        for seed in SEEDS:
            train_model(task, seed)
            
    logging.info("\n🎉🎉🎉 GAP ABLATION TRAINING COMPLETED! 🎉🎉🎉")
