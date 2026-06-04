import os
# =========================================================
# 1. 指定使用的显卡 (物理编号 0-7)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, recall_score, precision_score, confusion_matrix
import argparse
from tqdm import tqdm
import sys

# 开启 CuDNN 加速
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from model import TransHLA

# ================= 工具函数 =================

def addbatch(data, label, batchsize, shuffle=False):
    """构建推理用的 DataLoader"""
    data = TensorDataset(data, label)
    # num_workers=16 加速数据读取
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=shuffle, num_workers=16, pin_memory=True)
    return data_loader

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

def evaluate(data_loader, device, model):
    """单次推理过程"""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 模型前向传播
            Result, _ = model(inputs)
            
            # 获取正类概率 (假设 index 1 是正类)
            probs = F.softmax(Result, dim=1)[:, 1]
            _, predicted = torch.max(Result, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    return calculate_metrics(all_labels, all_preds, all_probs)

def load_data_raw(path, max_len=None):
    if path:
        path = path.strip()
    
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")

    print(f"Loading test data from: '{path}'")
    df = pd.read_csv(path, header=0)
    
    # === 数据清洗：移除超过 max_len 的序列 ===
    if max_len is not None:
        original_count = len(df)
        # 确保是字符串并计算长度
        df['seq_len'] = df.iloc[:, 0].astype(str).apply(len)
        # 过滤
        df_filtered = df[df['seq_len'] <= max_len]
        
        filtered_count = len(df_filtered)
        dropped = original_count - filtered_count
        
        if dropped > 0:
            print(f"[Warning] Dropped {dropped} sequences longer than {max_len} AA. Remaining: {filtered_count}")
            df = df_filtered
        else:
            print(f"All {original_count} sequences are valid length (<={max_len}).")
            
    seqs = list(zip(range(len(df)), df.iloc[:, 0]))
    labels = torch.tensor(np.array(df.iloc[:, 1], dtype='int64'))
    return seqs, labels

def get_args():
    parser = argparse.ArgumentParser(description="Inference Script for TransHLA")
    
    # === 模型配置 (必须与训练时保持一致) ===
    parser.add_argument('-max_len',type = int, default = 14, help="14 for HLA-I, 21 for HLA-II")
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-d_model', type=int, default=1280)
    parser.add_argument('-dim-feedforward', type=int, default=64)
    parser.add_argument('-cnn_num_channel', type=int, default=256)
    parser.add_argument('-cnn_kernel_size', type=int, default=3)
    parser.add_argument('-cnn_padding_size', type=int, default=1)
    parser.add_argument('-cnn_stride', type=int, default=1)
    parser.add_argument('-pooling_size', type=int, default=2)
    parser.add_argument('-region_embedding_size',type=int,default = 3)
    
    # === 路径配置 ===
    parser.add_argument('--test_path', type=str, required=True, help="Path to the new test CSV")
    parser.add_argument('--model_path', type=str, required=True, help="Directory where .pt files are saved")
    parser.add_argument('--model_name', type=str, required=True, help="Prefix of the saved models (e.g., transhla_hlai)")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. 设备设置
    gpu_count = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {gpu_count} GPUs.")
    
    # 推理时显存占用小，Batch Size 可以很大
    BATCH_SIZE = 2048 * gpu_count 
    print(f"Inference Batch Size: {BATCH_SIZE}")

    # 2. 准备 tokenizer (不需要加载 ESM 模型权重)
    print("Loading ESM-2 tokenizer...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    del esm_model # 释放显存
    
    # 3. 加载并编码数据 (带过滤)
    test_seqs, test_y = load_data_raw(args.test_path, max_len=args.max_len)
    
    if len(test_seqs) == 0:
        print("Error: No data left after length filtering.")
        return

    print("Encoding sequences (this may take a while)...")
    _, _, test_x = batch_converter(test_seqs)
    
    # 构建 DataLoader
    test_loader = addbatch(test_x, test_y, BATCH_SIZE, shuffle=False)

    # 4. 循环加载 5 个 Seed 的模型
    SEEDS = [1, 2, 3, 4, 5]
    results_list = []
    
    print(f"\n{'='*20} Start Evaluation on 5 Seeds {'='*20}")
    
    for seed in SEEDS:
        # 构造模型文件名
        model_filename = f"{args.model_name}_seed{seed}.pt"
        model_full_path = os.path.join(args.model_path, model_filename)
        
        if not os.path.exists(model_full_path):
            print(f"Warning: Model file not found: {model_full_path}. Skipping...")
            continue
            
        print(f"\n>>> Loading Model: {model_filename}")
        
        # 初始化模型结构
        model = TransHLA(args)
        
        # 加载权重
        state_dict = torch.load(model_full_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        # 推理时也可以用多卡加速
        if gpu_count > 1:
            model = nn.DataParallel(model)
            
        # 评估
        metrics = evaluate(test_loader, device, model)
        results_list.append(metrics)
        
        print(f"Seed {seed} Results: ACC={metrics['ACC']:.2f}, AUC={metrics['AUC']:.4f}")

    # 5. 统计汇总
    if not results_list:
        print("No results generated.")
        return

    df_res = pd.DataFrame(results_list)
    
    print(f"\n{'='*20} Final Aggregated Report {'='*20}")
    print(f"Test Set: {args.test_path}")
    print("-" * 60)
    print(df_res)
    print("-" * 60)
    
    metrics_names = ['ACC', 'F1', 'Recall', 'MCC', 'Precision', 'Specificity', 'AUC']
    print(f"{'Metric':<15} | {'Mean ± Std':<25}")
    print("-" * 40)
    
    for metric in metrics_names:
        if metric in df_res.columns:
            mean_val = df_res[metric].mean()
            std_val = df_res[metric].std()
            print(f"{metric:<15} | {mean_val:.4f} ± {std_val:.4f}")
    print("-" * 40)

    # 保存结果到CSV
    save_csv_name = f"eval_result_{args.model_name}_{os.path.basename(args.test_path)}"
    df_res.to_csv(save_csv_name, index=False)
    print(f"Results saved to {save_csv_name}")

if __name__ == "__main__":
    main()


    """
    进入到TransHLA-main这一级目录，运行训推代码。
    python model_train_test/predict.py    --test_path data/NEPDB_I.csv  --model_path models/   --model_name transhla_hlai

    python model_train_test/predict.py    --test_path data/HLA_II_external_1_time_negative.csv  --model_path models/   --model_name transhla_hlaii -max_len 21
    """