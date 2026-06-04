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
from tqdm import tqdm
import sys

# =========================================================
# 全局设置
# =========================================================
# 可以在这里指定显卡，或者在命令行运行前指定
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

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
    """计算所有指标: ACC, F1, Recall, MCC, Precision, Specificity, AUC"""
    acc = 100 * (y_pred == y_true).sum() / len(y_true)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.0

    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'ACC': acc, 
        'F1': f1, 
        'Recall': recall, 
        'MCC': mcc, 
        'Precision': precision, 
        'Specificity': specificity, 
        'AUC': auc
    }

# ================= 数据预处理 (Tokenizer) =================

class AminoAcidTokenizer:
    def __init__(self, max_len=14):
        # 20种常见氨基酸 + 填充(PAD)
        self.aa_dict = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8,
            'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 
            'X': 21, 'B': 22, 'Z': 23, 'J': 24 # 处理非常规氨基酸
        }
        self.pad_index = 0
        self.max_len = max_len
        self.vocab_size = len(self.aa_dict) + 1 # +1 for padding

    def encode(self, seq):
        seq = seq.upper().strip()
        # 截断
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        # 编码
        ids = [self.aa_dict.get(aa, 21) for aa in seq] # 未知字符映射到 'X' (21)
        
        # 填充
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
    
    # 训练集去重，测试集不去重
    if is_train:
        df = df.drop_duplicates(keep='first')
    
    seqs = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].values
    
    # Tokenize
    input_ids = tokenizer.batch_encode(seqs)
    
    # 转 Tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    return input_tensor, label_tensor

# ================= DPCNN 模型定义 =================

class DPCNN(nn.Module):
    def __init__(self, config):
        super(DPCNN, self).__init__()
        
        # Hyperparameters from image
        self.channel_size = 256
        self.kernel_size = 3
        self.pooling_size = 2
        self.vocab_size = config.vocab_size
        self.embedding_dim = 256
        self.num_classes = 2
        self.padding_index = 0
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            self.vocab_size, 
            self.embedding_dim, 
            padding_idx=self.padding_index
        )
        
        # 2. Region Embedding (Initial Feature Extraction)
        # 通常是一个无padding的卷积，或者padding后保持维度的卷积
        self.region_conv = nn.Conv1d(
            self.embedding_dim, 
            self.channel_size, 
            kernel_size=self.kernel_size, 
            padding=1
        )
        
        # 3. First Convolutional Block (Pre-activation)
        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=1)
        )
        
        # 4. Pooling & Pyramid Structure
        self.pooling = nn.MaxPool1d(kernel_size=self.pooling_size, stride=2)
        
        # Residual Conv Block (ReLU -> Conv -> ReLU -> Conv)
        # Note: Padding=1 ensures input/output length matches before pooling
        self.padding_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=1)
        )
        
        # 5. Output Layer
        self.fc = nn.Linear(self.channel_size, self.num_classes)

    def _block(self, x):
        # Pyramid Logic: Pooling -> ConvBlock -> Residual Add
        x = self.pooling(x)
        x_res = self.padding_conv(x)
        return x + x_res # Residual connection

    def forward(self, x):
        # x: [Batch, Seq_Len]
        
        # Embedding: [Batch, Seq_Len, Dim] -> [Batch, Dim, Seq_Len] (for Conv1d)
        x = self.embedding(x)
        x = x.permute(0, 2, 1) 
        
        # Region Embedding
        x = self.region_conv(x)
        
        # First Conv Block
        x = self.conv_block(x)
        
        # Pyramid Loop
        # 只要序列长度允许，就继续进行下采样
        while x.size(2) >= 2:
            x = self._block(x)
        
        # Global Max Pooling (if any length remains) or Flatten
        if x.size(2) > 1:
            x = F.max_pool1d(x, kernel_size=x.size(2))
            
        x = x.squeeze(2) # Flatten: [Batch, Channel]
        
        logits = self.fc(x)
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
    model = DPCNN(args).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n>>> Seed {seed} Training Start...")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # 训练
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # 可选：在进度条右边实时显示当前的 loss
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
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # 加载最佳模型并在测试集评估
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(test_loader, device, model)
    
    print(f"Seed {seed} Best Test ACC: {test_metrics['ACC']:.2f}, AUC: {test_metrics['AUC']:.4f}")
    
    # 也可以保存模型权重
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(best_model_state, os.path.join(args.model_path, f"{args.model_name}_seed{seed}.pt"))
    
    return test_metrics

# ================= 主函数 =================

def get_args():
    parser = argparse.ArgumentParser(description="DPCNN Baseline for HLA")
    
    # 数据路径
    parser.add_argument('--train_path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--validation_path', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test CSV')
    
    # 保存路径
    parser.add_argument('--model_path', type=str, default='./models_dpcnn', help='Folder to save models')
    parser.add_argument('--model_name', type=str, default='dpcnn_baseline', help='Model name prefix')
    
    # 模型超参数 (大部分已在类中硬编码，这里留一些可变的)
    parser.add_argument('-max_len', type=int, default=14, help='Max sequence length (14 for HLA-I, 21 for HLA-II)')
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=12800)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-patience', type=int, default=10)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. 准备 Tokenizer
    tokenizer = AminoAcidTokenizer(max_len=args.max_len)
    args.vocab_size = tokenizer.vocab_size # 将 vocab_size 传递给模型参数
    
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
    
    for seed in SEEDS:
        metrics = training(args, seed, train_loader, val_loader, test_loader, device)
        all_results.append(metrics)
        
    # 5. 结果汇总
    df_res = pd.DataFrame(all_results)
    
    print(f"\n{'='*20} DPCNN Final Results (5 Runs) {'='*20}")
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
    res_path = os.path.join(args.model_path, 'dpcnn_results.csv')
    df_res.to_csv(res_path, index=False)
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()


    """
    python /HARD-DATA/GPJ/AIDD/ecml2026/baselines/DPCNN.py \
  --train_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_I_epitope_train_shuffle.csv \
  --validation_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_I_epitope_validation.csv \
  --test_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_I_epitope_test.csv \
  --model_path models_dpcnn \
  -max_len 14

  python /HARD-DATA/GPJ/AIDD/ecml2026/baselines/DPCNN.py \
  --train_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_II_epitope_train_shuffle.csv \
  --validation_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_II_epitope_validation.csv \
  --test_path /HARD-DATA/GPJ/AIDD/ecml2026/MulHLA/data/HLA_II_epitope_test.csv \
  --model_path models_dpcnn \
  -max_len 21 --model_name dpcnn_baseline_hlaii
    """