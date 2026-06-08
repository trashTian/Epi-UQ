# MHC 结合预测 Baseline Bootstrap 评估脚本

包含一组用于 **MHC 肽段结合预测** baseline 的评估脚本。每个脚本对测试集做推理，再通过 **bootstrap 重采样** 计算二分类指标（ACC、F1、Recall、MCC、Precision、Specificity、AUC），并输出均值 ± 标准差。

## 通用流程

```text
test.csv  →  模型推理  →  bootstrap × N  →  per_run.csv + summary.csv + summary.txt
```

### 输入 CSV

| 列 | 说明 |
|---|---|
| 肽段列 | 自动识别 `Epitope.1` / `peptide` / `Peptide` / `epitope` / `sequence`，也可用 `--peptide_col` 指定 |
| 标签列 | 自动识别 `Label` / `label` / `binder` / `target` / `class` / `y`，也可用 `--label_col` 指定；取值须为 `0` 或 `1` |
| 等位基因列 | 部分脚本需要；自动识别 `Allele` / `allele` / `mhc` / `HLA` / `hla` 等，也可用 `--allele_col` 指定 |

肽段会自动转为大写，并过滤含非标准氨基酸（`ACDEFGHIKLMNPQRSTVWY` 以外字符）的行。

### 三种运行模式

| 模式 | 条件 | 行为 |
|---|---|---|
| **逐行等位基因** | CSV 含等位基因列 | 每行用自己的等位基因预测 |
| **peptide_only** | `--peptide_only` 或 CSV 无等位基因列 | 所有行使用 `--default_allele` |
| **panel** | 提供 `--allele_panel_csv` / `--allele_panel_txt` | 对每个等位基因分别预测，再跨等位基因聚合 |

### 输出文件

每个脚本在 `--out_dir` 下生成三个文件，前缀因工具而异：

| 文件 | 内容 |
|---|---|
| `{Tool}_bootstrap_per_run.csv` | 每次 bootstrap 的各指标 |
| `{Tool}_bootstrap_summary.csv` | 各指标的 Mean、Std、Mean±Std |
| `{Tool}_bootstrap_summary.txt` | 纯文本版 summary |

指标顺序：ACC、F1、Recall、MCC、Precision、Specificity、AUC。

### 公共参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--test_csv` | （必填） | 测试集路径 |
| `--out_dir` | （必填） | 输出目录 |
| `--n_bootstrap` | `5` | bootstrap 次数 |
| `--seed` | `42` | 随机种子 |
| `--max_alleles` | `0` | `>0` 时 panel 模式仅使用前 N 个等位基因（调试用） |

---


## 脚本一览

### MHC I 类

| 脚本 | 外部依赖 | 默认等位基因 | 肽长过滤 |
|---|---|---|---|
| `run_bootstrap_mhcflurry.py` | [MHCflurry](https://github.com/openvax/mhcflurry) | `HLA-A*02:01` | 8–15 |
| `run_bootstrap_mhcnuggetsI.py` | [MHCnuggets](https://github.com/KarchinLab/mhcnuggets) | `HLA-A02:01` | 8–15 |
| `run_bootstrap_netmhcpan41.py` | [NetMHCpan-4.1](https://services.healthtech.dtu.dk/service.php?NetMHCpan-4.1) | `HLA-A*02:01` | 8–15 |
| `run_bootstrap_mixmhcpred.py` | [MixMHCpred 3.0](https://github.com/GfellerLab/MixMHCpred) | `HLA-A*02:01` | 8–15 |
| `run_bootstrap_anthem.py` | [Anthem](https://github.com/IEDB-Anthem/anthem) | （仅 panel） | 8–15 |

### MHC II 类

| 脚本 | 外部依赖 | 默认等位基因 | 肽长过滤 |
|---|---|---|---|
| `run_bootstrap_mhcnuggetsII.py` | MHCnuggets（`-c II`） | `HLA-DRA0101-DRB10101` | 8–40（可调） |
| `run_bootstrap_netmhciipan43.py` | [NetMHCIIpan-4.3](https://services.healthtech.dtu.dk/service.php?NetMHCIIpan-4.3) | `DRB1_0101` | 9–40（可调） |
| `run_bootstrap_mixmhc2pred.py` | [MixMHC2pred 2.0](https://github.com/GfellerLab/MixMHC2pred) | `DRB1_01_01` | 8–40（可调） |
| `run_bootstrap_deepseqpan.py` | [DeepSeqPanII](https://github.com/pcpLiu/DeepSeqPanII) | `DRA*01:01-DRB1*01:01` | 5–25 |

### 辅助脚本

| 脚本 | 说明 |
|---|---|
| `deepseqpanii_predict_allele.py` | DeepSeqPanII 单等位基因对批量推理 worker，由 `run_bootstrap_deepseqpan.py` 以子进程调用，一般无需单独运行 |

---

## 使用示例
### MHCnuggets I

```bash
# Class I
python run_bootstrap_mhcnuggetsI.py \
  --test_csv HLA_I_epitope_test.csv \
  --out_dir results/mhcnuggets_i \
  --n_bootstrap 5 \
  --seed 42 \
  --peptide_col Epitope.1 \
  --label_col Label \
  --allele_panel_csv MhcnuggetsI_alleles.csv \
  --aggregate max
```

各工具额外依赖：

| 工具 | 安装方式 |
|---|---|
| MHCflurry | `pip install mhcflurry`；首次使用会自动下载模型，或通过 `--mhcflurry_models_dir` 指定 |
| MHCnuggets | `pip install mhcnuggets`（或从源码安装）；需 `mhcnuggets-predict` 在 PATH 中 |
| NetMHCpan / NetMHCIIpan | 从 DTU 下载并解压对应平台包，**不可 pip 安装** |
| MixMHCpred / MixMHC2pred | 从 GitHub 克隆并编译，将可执行文件路径传给 `--mixmhcpred_dir` / `--mixmhc2pred_dir` |
| DeepSeqPanII | 克隆官方仓库；推理 worker 需 PyTorch（通常旧版 0.4.x） |
| Anthem | 克隆官方仓库；需 Java 与内置 Weka |

---
