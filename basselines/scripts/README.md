# MHC Binding Prediction Baseline Bootstrap Evaluation Scripts

A collection of scripts for evaluating **MHC peptide binding prediction** baselines. Each script runs inference on a test set, then computes binary classification metrics (ACC, F1, Recall, MCC, Precision, Specificity, AUC) via **bootstrap resampling**, reporting mean ± standard deviation.

## General workflow

```text
test.csv  →  model inference  →  bootstrap × N  →  per_run.csv + summary.csv + summary.txt
```

### Input CSV

| Column | Description |
|---|---|
| Peptide | Auto-detected: `Epitope.1` / `peptide` / `Peptide` / `epitope` / `sequence`; override with `--peptide_col` |
| Label | Auto-detected: `Label` / `label` / `binder` / `target` / `class` / `y`; override with `--label_col`; values must be `0` or `1` |
| Allele | Required by some scripts; auto-detected: `Allele` / `allele` / `mhc` / `HLA` / `hla`; override with `--allele_col` |

Peptides are uppercased. Most scripts drop rows containing non-standard amino acids (outside `ACDEFGHIKLMNPQRSTVWY`).

### Run modes

| Mode | Condition | Behavior |
|---|---|---|
| **Per-row allele** | Test CSV has an allele column | Predict each row with its own allele |
| **peptide_only** | `--peptide_only`, or no allele column (and no panel) | Use `--default_allele` for all rows |
| **panel** | `--allele_panel_csv` and/or `--allele_panel_txt` (tool-specific) | Predict per allele in the panel, then aggregate across alleles |

### Output files

Each script writes three files under `--out_dir` (prefix varies by tool):

| File | Content |
|---|---|
| `{Tool}_bootstrap_per_run.csv` | Metrics for each bootstrap replicate |
| `{Tool}_bootstrap_summary.csv` | Mean, Std, and Mean±Std per metric |
| `{Tool}_bootstrap_summary.txt` | Plain-text summary |

Metric order: ACC, F1, Recall, MCC, Precision, Specificity, AUC.

### Common arguments

| Argument | Default | Description |
|---|---|---|
| `--test_csv` | (required) | Path to test set |
| `--out_dir` | (required) | Output directory |
| `--n_bootstrap` | `5` | Number of bootstrap replicates |
| `--seed` | `42` | Random seed |
| `--max_alleles` | `0` | If `>0`, use only the first N alleles in panel mode (debug / speed-up) |

---

## Script overview

### MHC class I

| Script | External dependency | Default allele |
|---|---|---|
| `run_bootstrap_mhcflurry.py` | [MHCflurry](https://github.com/openvax/mhcflurry) | `HLA-A*02:01` |
| `run_bootstrap_mhcnuggetsI.py` | [MHCnuggets](https://github.com/KarchinLab/mhcnuggets) | `HLA-A02:01` |
| `run_bootstrap_netmhcpan41.py` | [NetMHCpan-4.1](https://services.healthtech.dtu.dk/service.php?NetMHCpan-4.1) | `HLA-A*02:01` |
| `run_bootstrap_mixmhcpred.py` | [MixMHCpred 3.0](https://github.com/GfellerLab/MixMHCpred) | `HLA-A*02:01` |
| `run_bootstrap_anthem.py` | [Anthem](https://github.com/17shutao/Anthem) | (panel only) |

### MHC class II

| Script | External dependency | Default allele |
|---|---|---|
| `run_bootstrap_mhcnuggetsII.py` | MHCnuggets (`-c II`) | `HLA-DRA0101-DRB10101` |
| `run_bootstrap_netmhciipan43.py` | [NetMHCIIpan-4.3](https://services.healthtech.dtu.dk/service.php?NetMHCIIpan-4.3) | `DRB1_0101` |
| `run_bootstrap_mixmhc2pred.py` | [MixMHC2pred 2.0](https://github.com/GfellerLab/MixMHC2pred) | `DRB1_01_01` |
| `run_bootstrap_deepseqpan.py` | [DeepSeqPanII](https://github.com/pcpLiu/DeepSeqPanII) | `DRA*01:01-DRB1*01:01` |

### Helper script

| Script | Description |
|---|---|
| `deepseqpanii_predict_allele.py` | DeepSeqPanII batch inference worker for one allele pair; invoked as a subprocess by `run_bootstrap_deepseqpan.py` (usually not run standalone) |

---

## Usage example

### MHCnuggets class I

```bash
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

## Tool-specific dependencies

| Tool | Installation |
|---|---|
| MHCflurry | `pip install mhcflurry`; models download on first use, or set `--mhcflurry_models_dir` |
| MHCnuggets | `pip install mhcnuggets` (or install from source); `mhcnuggets-predict` must be on `PATH` |
| NetMHCpan / NetMHCIIpan | Download and unpack the DTU platform bundle; **not** installable via pip |
| MixMHCpred / MixMHC2pred | Clone from GitHub and build; pass `--mixmhcpred_dir` / `--mixmhc2pred_dir` |
| DeepSeqPanII | Clone the official repo; inference worker needs PyTorch (often legacy 0.4.x) |
| Anthem | Clone the official repo; requires Java and bundled Weka |

---
