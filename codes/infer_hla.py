import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. 路径与全局配置
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ESM_MODEL_PATH = "/data1/gpj/LLMModels/models/facebook/esm2_t33_650M_UR50D"

# 假设你的 3 个模型的权重保存在这个文件夹下，且命名为 hlai_esm650_seed1.pth, seed2.pth, seed3.pth
# 请根据实际情况修改 SEEDS 和 CHECKPOINT_DIR
SEEDS = [1, 2, 3] 
CHECKPOINT_DIR = "/data1/gpj/AIDD/hla_epitope/Advanced_baselines/checkpoints_lora"

MAX_LEN = 30
BATCH_SIZE = 128

# 需要评估的 4 个测试集
TEST_SETS = {
    "Test_Set": "/data1/gpj/AIDD/hla_epitope/data/HLA_I_epitope_test.csv",
    "Ext_1_Time": "/data1/gpj/AIDD/hla_epitope/data/HLA_I_external_1_time_negative.csv",
    "Ext_4_Time": "/data1/gpj/AIDD/hla_epitope/data/HLA_I_external_4_time_negative.csv",
    "NEPDB": "/data1/gpj/AIDD/hla_epitope/data/NEPDB_I.csv"
}

# ==========================================
# 2. Dataset
# ==========================================
class EpitopeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=30):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 兼容不同的列名 (比如有些数据集叫 Epitope.1，有些叫 Peptide)
        self.seq_col = 'Epitope.1' if 'Epitope.1' in self.data.columns else 'Peptide'

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        peptide = self.data.iloc[idx][self.seq_col]
        label = float(self.data.iloc[idx]['Label'])
        encoding = self.tokenizer(peptide, add_special_tokens=True, padding='max_length',
                                  truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ==========================================
# 3. 模型架构定义 (直接复用你训练时的代码)
# ==========================================
class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(in_features, 1)

    def forward(self, x, mask):
        attn_scores = self.attention_weights(x).squeeze(-1) 
        # 将 -1e9 改为 -1e4，完美兼容 FP16 混合精度
        attn_scores = attn_scores.masked_fill(mask == 0, -1e4) 
        attn_weights = F.softmax(attn_scores, dim=-1) 
        pooled_output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled_output

class EpiAdvanced_Model(nn.Module):
    def __init__(self, esm_model_path):
        super(EpiAdvanced_Model, self).__init__()
        base_esm = EsmModel.from_pretrained(esm_model_path, local_files_only=True)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=True, 
                                 r=32, lora_alpha=64, target_modules=["query", "key", "value"], bias="none")
        self.esm = get_peft_model(base_esm, peft_config)
        hidden_size = self.esm.config.hidden_size 
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attn_pool = AttentionPooling(in_features=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        conv_input = sequence_output.permute(0, 2, 1)
        conv_output = self.conv1d(conv_input)
        seq_features = conv_output.permute(0, 2, 1)
        pooled_rep = self.attn_pool(seq_features, attention_mask)
        logits = self.classifier(pooled_rep)
        return logits.squeeze(-1)

# ==========================================
# 4. 指标计算与评估函数
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
    auprc = average_precision_score(labels, preds_prob)
    
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return [acc, f1, recall, mcc, precision, spec, auroc, auprc]

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing", leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # 使用 AMP 混合精度加速推理
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def format_mean_std(values):
    """格式化为 Mean ± Std 形式 (这里统一采用百分制，若不需要可去除 *100)"""
    mean_val = np.mean(values) * 100
    std_val = np.std(values) * 100
    return f"{mean_val:.2f} ± {std_val:.2f}"

# ==========================================
# 5. 主程序逻辑
# ==========================================
def main():
    print(f"🚀 正在初始化 Tokenizer 与 模型 (Device: {DEVICE})...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    
    # 模型只实例化一次，极大地节省时间和显存
    model = EpiAdvanced_Model(ESM_MODEL_PATH).to(DEVICE)
    
    # 存放所有测试集最终结果的字典
    results_dict = {
        "Test Dataset": [],
        "ACC": [], "F1": [], "Recall": [], "MCC": [], 
        "Precision": [], "Specificity": [], "AUC": [], "AUPRC": []
    }

    # 外层循环：遍历 4 个测试集
    for ds_name, csv_path in TEST_SETS.items():
        print(f"\n" + "="*60)
        print(f"🎯 正在评估测试集: {ds_name}")
        print(f"📁 路径: {csv_path}")
        print("="*60)
        
        if not os.path.exists(csv_path):
            print(f"⚠️ 找不到数据集 {csv_path}，已跳过！")
            continue
            
        dataset = EpitopeDataset(csv_path, tokenizer, max_len=MAX_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # 临时记录当前测试集下，3个 seed 模型的所有指标
        metrics_all_seeds = {k: [] for k in ["ACC", "F1", "Recall", "MCC", "Precision", "Specificity", "AUC", "AUPRC"]}
        
        # 内层循环：遍历 3 个 seed
        for seed in SEEDS:
            weight_path = os.path.join(CHECKPOINT_DIR, f"hlai_esm650_seed{seed}.pth")
            
            if not os.path.exists(weight_path):
                print(f"⚠️ 找不到权重文件 {weight_path}，跳过该 seed！")
                continue
                
            # 加载当前 seed 的权重
            state_dict = torch.load(weight_path, map_location=DEVICE)
            # 处理部分情况下由于 DataParallel 导致的多余 "module." 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            
            # 运行推理并计算指标
            y_true, y_prob = evaluate_model(model, dataloader)
            res = calculate_metrics(y_true, y_prob)
            
            # 将该 seed 的成绩对应添加到列表中
            for i, key in enumerate(metrics_all_seeds.keys()):
                metrics_all_seeds[key].append(res[i])
                
            print(f"   ✓ Seed {seed} 评测完毕. AUC: {res[6]*100:.2f}%, MCC: {res[3]*100:.2f}%")

        # 汇总当前测试集的均值和方差
        if len(metrics_all_seeds["ACC"]) > 0:
            results_dict["Test Dataset"].append(ds_name)
            for key in metrics_all_seeds.keys():
                results_dict[key].append(format_mean_std(metrics_all_seeds[key]))

    # ==========================================
    # 6. 生成论文专用表格并导出
    # ==========================================
    print("\n✅ 所有测试集推理完毕，正在生成论文统计表...")
    df_results = pd.DataFrame(results_dict)
    
    print("\n" + "="*110)
    print(df_results.to_string(index=False))
    print("="*110)
    
    csv_out_path = "multi_seed_inference_results_hlai.csv"
    df_results.to_csv(csv_out_path, index=False)
    print(f"\n🎉 完美收工！结果已自动保存至: {csv_out_path} (可直接复制到 Excel/论文中)")

if __name__ == "__main__":
    main()

