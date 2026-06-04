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
        "max_len": 30,
        "lora_r": 32, "lora_alpha": 64
    },
    "HLA-II": {
        "train": "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_train_shuffle.csv",
        "val": "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_validation.csv",
        "max_len": 35,
        "lora_r": 32, "lora_alpha": 64
    }
}

SAVE_DIR = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_ablation"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "training_ablation.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'), logging.StreamHandler()])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
SEEDS = [42, 921, 2026] # 3 个独立随机种子
LEARNING_RATE = 2e-4

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
# 3. 三种消融结构定义
# ==========================================
class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention_weights = nn.Linear(in_features, 1)
    def forward(self, x, mask):
        attn_scores = self.attention_weights(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return torch.bmm(F.softmax(attn_scores, dim=-1).unsqueeze(1), x).squeeze(1)

def get_ablation_model(config_name, esm_path, task_info):
    base_esm = EsmModel.from_pretrained(esm_path, local_files_only=True)
    hidden_size = base_esm.config.hidden_size # 1280
    
    # 核心分发逻辑
    class AblationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config_name = config_name
            
            # --- 1. 处理 ESM 躯干 (Frozen or LoRA) ---
            if "Frozen" in config_name:
                for param in base_esm.parameters():
                    param.requires_grad = False
                self.esm = base_esm
            else:
                peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
                                         r=task_info['lora_r'], lora_alpha=task_info['lora_alpha'], 
                                         target_modules=["query", "key", "value"], lora_dropout=0.2, bias="none")
                self.esm = get_peft_model(base_esm, peft_config)

            # --- 2. 处理分类头 (MLP or CNN+Attn) ---
            if "MLP" in config_name:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 1)
                )
            else:
                self.conv1d = nn.Sequential(
                    nn.Conv1d(hidden_size, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2)
                )
                self.attn_pool = AttentionPooling(512)
                self.classifier = nn.Sequential(
                    nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
                )

        def forward(self, input_ids, attention_mask):
            outputs = self.esm(input_ids, attention_mask=attention_mask)
            
            if "MLP" in self.config_name:
                # 仅使用 [CLS] 向量
                cls_rep = outputs.last_hidden_state[:, 0, :]
                return self.classifier(cls_rep).squeeze(-1)
            else:
                # 使用 CNN + Attention
                seq = self.conv1d(outputs.last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
                pooled_rep = self.attn_pool(seq, attention_mask)
                return self.classifier(pooled_rep).squeeze(-1)
                
    return AblationModel()

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

def train_model(task_name, config_name, seed):
    seed_everything(seed)
    logging.info(f"\n{'='*50}\n🚀 TASK: {task_name} | CONFIG: {config_name} | SEED: {seed}\n{'='*50}")
    
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    task_info = DATASETS[task_name]
    
    # 动态 Batch Size：Frozen 模型不占显存，狂飙 1024；LoRA 模型求稳用 256
    batch_size = 1024 if "Frozen" in config_name else 256
    
    train_dl = DataLoader(EpitopeDataset(task_info['train'], tokenizer, task_info['max_len']), 
                          batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl   = DataLoader(EpitopeDataset(task_info['val'], tokenizer, task_info['max_len']), 
                          batch_size=batch_size, shuffle=False, num_workers=8)
    
    model = get_ablation_model(config_name, ESM_MODEL_PATH, task_info)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
    else:
        model = model.to(DEVICE)
        
    criterion = nn.BCEWithLogitsLoss()
    # 智能拾取可训练参数 (Frozen 模式下 ESM 的参数不会被放进优化器)
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
            # 保存时兼容 DataParallel
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), model_save_path)
            logging.info(f"[✅] Checkpoint saved! Best AUC: {best_auc:.4f}")

    # 清理内存，防止 18 个循环下来显存爆炸
    del model, optimizer, train_dl, val_dl
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 5. 自动化轰炸启动
# ==========================================
if __name__ == "__main__":
    # 需要跑的 3 个消融配置
    CONFIGS = [
        "Frozen_MLP",      # 1. 冻结ESM + [CLS] MLP
        "Frozen_CNN_Attn", # 2. 冻结ESM + CNN & Attn
        "LoRA_MLP"         # 3. LoRA ESM + [CLS] MLP
    ]
    
    # 嵌套循环执行 18 次独立训练
    for task in ["HLA-I", "HLA-II"]:
        for config in CONFIGS:
            for seed in SEEDS:
                train_model(task, config, seed)
                
    logging.info("\n🎉🎉🎉 ALL ABLATION TRAINING MISSIONS COMPLETED! 🎉🎉🎉")
