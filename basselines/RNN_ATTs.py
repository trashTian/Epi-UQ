import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, recall_score, precision_score, confusion_matrix
import random
import argparse
from tqdm import tqdm  # 进度条
import sys

# =========================================================
# 全局设置
# =========================================================
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# 开启 CuDNN 加速
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# ================= 工具函数 =================

def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True 

def calculate_metrics(y_true, y_pred, y_probs):
    """计算所有指标"""
    acc = 100 * (y_pred == y_true).sum() / len(y_true)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.0

    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'ACC': acc, 'F1': f1, 'Recall': recall, 'MCC': mcc, 
        'Precision': precision, 'Specificity': specificity, 'AUC': auc
    }

# ================= Tokenizer =================

class AminoAcidTokenizer:
    def __init__(self, max_len=14):
        # 基础氨基酸字典
        self.aa_dict = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8,
            'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 
            'X': 21, 'B': 22, 'Z': 23, 'J': 24
        }
        self.pad_index = 0
        self.max_len = max_len
        # 图片要求 Vocabulary Size: 40
        # 我们的字典其实只有25个左右，但为了完全符合超参数设置，我们在模型层设置 vocab=40
        self.vocab_size = 40 

    def encode(self, seq):
        seq = seq.upper().strip()
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        ids = [self.aa_dict.get(aa, 21) for aa in seq] 
        
        if len(ids) < self.max_len:
            ids = ids + [self.pad_index] * (self.max_len - len(ids))
            
        return ids

    def batch_encode(self, seq_list):
        return [self.encode(s) for s in seq_list]

def load_data_and_tokenize(path, tokenizer, is_train=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")

    print(f"Loading data from: {path}")
    df = pd.read_csv(path, header=0)
    
    if is_train:
        df = df.drop_duplicates(keep='first')
    
    seqs = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].values
    
    input_ids = tokenizer.batch_encode(seqs)
    
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    return input_tensor, label_tensor

# ================= RNN_ATTs 模型定义 =================

class Attention(nn.Module):
    """
    Attention Mechanism
    Assigns weights to LSTM outputs.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 简单的 Attention: 学习一个权重向量将 hidden_dim 映射到 attention score
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs: [Batch, Seq_Len, Hidden_Dim * Num_Directions]
        
        # 1. 计算 Attention Score
        # weights: [Batch, Seq_Len, 1]
        weights = torch.tanh(self.attn(rnn_outputs)) 
        
        # 2. Softmax 归一化
        weights = F.softmax(weights, dim=1)
        
        # 3. 加权求和得到 Context Vector
        # context: [Batch, Hidden_Dim * Num_Directions]
        context = torch.sum(weights * rnn_outputs, dim=1)
        
        return context

class RNN_ATTs(nn.Module):
    def __init__(self, config):
        super(RNN_ATTs, self).__init__()
        
        # Hyperparameters from image
        self.vocab_size = 40
        self.embed_dim = 256
        self.hidden_dim = 128
        self.n_layers = 2
        self.bidirectional = True
        self.dropout_rate = 0.2
        self.padding_idx = 0
        self.second_hidden_size = 64
        self.output_dim = 2
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            self.vocab_size, 
            self.embed_dim, 
            padding_idx=self.padding_idx
        )
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout_rate # PyTorch adds dropout between LSTM layers
        )
        
        # Determine dimension after LSTM
        # Bidirectional means output dimension is hidden_dim * 2
        self.lstm_out_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        # 3. Attention Mechanism
        self.attention = Attention(self.lstm_out_dim)
        
        # 4. Fully Connected Layers
        # Image: "passed through a fully connected layer with ReLU activation"
        self.fc1 = nn.Linear(self.lstm_out_dim, self.second_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate) # Add dropout before final layer
        
        # 5. Final Classification Layer
        self.fc2 = nn.Linear(self.second_hidden_size, self.output_dim)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        
        # Embedding
        embedded = self.embedding(x) # [Batch, Seq_Len, Embed_Dim]
        
        # LSTM
        # lstm_out: [Batch, Seq_Len, Hidden_Dim * 2]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Attention
        # context: [Batch, Hidden_Dim * 2]
        context = self.attention(lstm_out)
        
        # Second Hidden Layer + ReLU + Dropout
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Final Output
        logits = self.fc2(out)
        return logits

# ================= 训练与评估流程 =================

def evaluate(data_loader, device, model):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            
            probs = F.softmax(logits, dim=1)[:, 1]
            _, predicted = torch.max(logits, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    return calculate_metrics(all_labels, all_preds, all_probs)

def training(args, seed, train_loader, val_loader, test_loader, device):
    set_seed(seed)
    
    # 初始化模型
    model = RNN_ATTs(args).to(device)
    
    # 使用 Adam 优化器 (通常用于 RNN)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n>>> Seed {seed} Training Start...")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # === 这里添加了进度条 ===
        with tqdm(train_loader, desc=f"Seed {seed} | Epoch {epoch+1}/{args.epochs}", unit="batch", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                # 实时显示 Loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # 验证
        val_metrics = evaluate(val_loader, device, model)
        val_acc = val_metrics['ACC']
        
        # 保存最佳
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}. Best Val ACC: {best_acc:.2f}")
            break
            
    # 加载最佳模型并在测试集评估
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(test_loader, device, model)
    
    print(f"Seed {seed} Finished. Best Test ACC: {test_metrics['ACC']:.2f}, AUC: {test_metrics['AUC']:.4f}")
    
    # 保存模型权重
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(best_model_state, os.path.join(args.model_path, f"{args.model_name}_seed{seed}.pt"))
    
    return test_metrics

# ================= 主函数 =================

def get_args():
    parser = argparse.ArgumentParser(description="RNN_ATTs Baseline for HLA")
    
    # 数据路径
    parser.add_argument('--train_path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--validation_path', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test CSV')
    
    # 保存路径
    parser.add_argument('--model_path', type=str, default='./models_rnn_att', help='Folder to save models')
    parser.add_argument('--model_name', type=str, default='rnn_att_baseline', help='Model name prefix')
    
    # 模型超参数 (大部分已在类中根据截图硬编码)
    parser.add_argument('-max_len', type=int, default=14, help='Max sequence length')
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=1280)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-patience', type=int, default=10)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. 准备 Tokenizer
    tokenizer = AminoAcidTokenizer(max_len=args.max_len)
    
    # 2. 加载数据
    train_x, train_y = load_data_and_tokenize(args.train_path, tokenizer, is_train=True)
    val_x, val_y = load_data_and_tokenize(args.validation_path, tokenizer, is_train=False)
    test_x, test_y = load_data_and_tokenize(args.test_path, tokenizer, is_train=False)
    
    # 3. 创建 DataLoader
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    test_ds = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 4. 5次随机种子实验
    SEEDS = [1, 2, 3, 4, 5]
    all_results = []
    
    print(f"{'='*20} Start 5-Fold Random Seed Experiments {'='*20}")
    
    for seed in SEEDS:
        metrics = training(args, seed, train_loader, val_loader, test_loader, device)
        all_results.append(metrics)
        
    # 5. 结果汇总
    df_res = pd.DataFrame(all_results)
    
    print(f"\n{'='*20} RNN_ATTs Final Results (5 Runs) {'='*20}")
    print(df_res)
    print("-" * 60)
    
    metrics_list = ['ACC', 'F1', 'Recall', 'MCC', 'Precision', 'Specificity', 'AUC']
    print(f"{'Metric':<15} | {'Mean ± Std':<25}")
    print("-" * 40)
    
    for metric in metrics_list:
        if metric in df_res.columns:
            mean_val = df_res[metric].mean()
            std_val = df_res[metric].std()
            print(f"{metric:<15} | {mean_val:.4f} ± {std_val:.4f}")
    print("-" * 40)
    
    # 保存结果到文件
    res_path = os.path.join(args.model_path, 'rnn_att_results.csv')
    df_res.to_csv(res_path, index=False)
    print(f"Detailed results saved to {res_path}")

if __name__ == "__main__":
    main()


    """
    python /HARD-DATA/GPJ/AIDD/ecml2026/baselines/RNN_ATTs.py \
  --train_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_I_epitope_train_shuffle.csv \
  --validation_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_I_epitope_validation.csv \
  --test_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_I_epitope_test.csv \
  --model_path models_rnn_att \
  -max_len 14 --model_name rnn_att_baseline_i
    
  
  python /HARD-DATA/GPJ/AIDD/ecml2026/baselines/RNN_ATTs.py \
  --train_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_II_epitope_train_shuffle.csv \
  --validation_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_II_epitope_validation.csv \
  --test_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_II_epitope_test.csv \
  --model_path models_rnn_att \
  -max_len 21 --model_name rnn_att_baseline_ii
    """