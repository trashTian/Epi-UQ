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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm
import random

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 1. 路径与超参数配置
# ==========================================
TRAIN_CSV = "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_train_shuffle.csv"
VAL_CSV   = "/data1/gpj/AIDD/hla_epitope/data/HLA_II_epitope_validation.csv"
SAVE_DIR  = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "training_lora_cnn_attention_hlaii_esm650_seed921.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'), logging.StreamHandler()])

ESM_MODEL_PATH = "/data1/gpj/LLMModels/models/facebook/esm2_t33_650M_UR50D"
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 2e-4
LORA_R = 32
LORA_ALPHA = 64

logging.info(f"Using device: {DEVICE}")

# ==========================================
# 2. Dataset
# ==========================================
class EpitopeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=30):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        peptide = self.data.iloc[idx]['Epitope.1']
        label = float(self.data.iloc[idx]['Label'])
        encoding = self.tokenizer(peptide, add_special_tokens=True, padding='max_length',
                                  truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ==========================================
# 3. 终极架构：ESM2(LoRA) + 1D-CNN + Attention Pooling
# ==========================================
class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(in_features, 1)

    def forward(self, x, mask):
        # x shape: [Batch, Seq_Len, Features]
        # mask shape:[Batch, Seq_Len]
        attn_scores = self.attention_weights(x).squeeze(-1) # [Batch, Seq_Len]
        
        # 将 padding 的部分注意力置为极小值
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1) # [Batch, Seq_Len]
        
        # 加权求和
        pooled_output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) #[Batch, Features]
        return pooled_output

class EpiAdvanced_Model(nn.Module):
    def __init__(self, esm_model_path):
        super(EpiAdvanced_Model, self).__init__()
        base_esm = EsmModel.from_pretrained(esm_model_path, local_files_only=True)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
                                 r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["query", "key", "value"], 
                                 lora_dropout=0.2, bias="none")
        self.esm = get_peft_model(base_esm, peft_config)
        
        hidden_size = self.esm.config.hidden_size # 1280
        
        # 1D-CNN 用于捕获局部 Motif 模式 (如三联氨基酸特征)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 自动注意力池化，寻找锚定位点
        self.attn_pool = AttentionPooling(in_features=512)
        
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        # 获取完整的序列输出[Batch, Seq_Len, 1280]
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 调整维度以适应 Conv1d: [Batch, Channels, Seq_Len]
        conv_input = sequence_output.permute(0, 2, 1)
        conv_output = self.conv1d(conv_input)
        
        # 调整回[Batch, Seq_Len, Channels] 以送入 Attention Pooling
        seq_features = conv_output.permute(0, 2, 1)
        
        # 利用 Attention 聚合全局特征
        pooled_rep = self.attn_pool(seq_features, attention_mask)
        
        # 预测
        logits = self.classifier(pooled_rep)
        return logits.squeeze(-1)

# ==========================================
# 4. 指标与验证
# ==========================================
def calculate_metrics(labels, preds_prob):
    labels = np.array(labels)
    preds_prob = np.array(preds_prob)
    y_pred = (preds_prob > 0.5).astype(int)
    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, zero_division=0)
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    mcc = matthews_corrcoef(labels, y_pred)
    auroc = roc_auc_score(labels, preds_prob)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, f1, recall, mcc, precision, spec, auroc

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [],[]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            loss = criterion(logits, batch['label'].to(DEVICE))
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return total_loss / len(dataloader), calculate_metrics(all_labels, all_preds)

# ==========================================
# 5. 训练引擎
# ==========================================
def main():
    logging.info("Starting Advanced Architecture (CNN+Attn) Training...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    model = EpiAdvanced_Model(ESM_MODEL_PATH).to(DEVICE)
    
    train_dataset = EpitopeDataset(TRAIN_CSV, tokenizer)
    val_dataset   = EpitopeDataset(VAL_CSV, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_auroc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        train_preds, train_labels = [],[]
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch in loop:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            loss = criterion(logits, batch['label'].to(DEVICE))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                train_preds.extend(torch.sigmoid(logits).cpu().numpy())
                train_labels.extend(batch['label'].numpy())
            loop.set_postfix(loss=loss.item())
            
        tr_acc, tr_f1, tr_rec, tr_mcc, tr_prec, tr_spec, tr_auc = calculate_metrics(train_labels, train_preds)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)
        v_acc, v_f1, v_rec, v_mcc, v_prec, v_spec, v_auc = val_metrics
        
        logging.info(f"========== EPOCH {epoch} ==========")
        logging.info(f"AUC  | Train: {tr_auc:.4f} | Val: {v_auc:.4f}")
        logging.info(f"ACC  | Train: {tr_acc:.4f} | Val: {v_acc:.4f}")
        logging.info(f"F1   | Train: {tr_f1:.4f} | Val: {v_f1:.4f}")
        logging.info(f"Rec  | Train: {tr_rec:.4f} | Val: {v_rec:.4f}")
        logging.info(f"Spec | Train: {tr_spec:.4f} | Val: {v_spec:.4f}")
        logging.info(f"MCC  | Train: {tr_mcc:.4f} | Val: {v_mcc:.4f}")
        
        if v_auc > best_auroc:
            best_auroc = v_auc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lora_cnn_attention_hlaii_esm650_seed921.pth"))
            logging.info(f"[🚀] New SOTA Model saved! (AUC: {best_auroc:.4f})")

if __name__ == "__main__":
    seed_everything(921)
    main()
