# Epi-UQ: Allele-Agnostic Prediction of HLA-Presented Epitopes via Parameter-Efficient Representation Learning and Ensemble Uncertainty Quantification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org/)

**Epi-UQ** is an allele-agnostic framework for peptide-HLA binding prediction that integrates anchor-aware representation learning with ensemble-based epistemic uncertainty quantification. By enforcing localized physicochemical constraints and quantifying prediction variance, Epi-UQ effectively mitigates shortcut learning and systematically filters false positives in out-of-distribution (OOD) and highly imbalanced clinical screening scenarios.

## 🔥 Key Features

- **Allele-Agnostic Prediction**: Predict peptide presentation without requiring explicit HLA allele inputs
- **Parameter-Efficient Architecture**: Leverages LoRA-adapted ESM-2 with localized 1D-CNN and attention pooling
- **Uncertainty Quantification**: Ensemble-based epistemic UQ reduces false-positive predictions by **91.4%** in imbalanced screening
- **Robust OOD Generalization**: Maintains high predictive fidelity on novel mutational landscapes and cross-source cohorts
- **Clinical Translation**: Provides calibrated risk-stratification for efficient neoantigen prioritization

## 📦 Installation

### Environment Setup

```bash
# Create conda environment
conda create -n immunellm python=3.10 -y
conda activate immunellm

# Install PyArrow
pip install pyarrow==17.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install core dependencies
pip install pandas==2.0.3 numpy==1.24.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install transformers and ESM
pip install transformers==4.57.3 fair-esm==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install datasets and fine-tuning libraries
pip install datasets==2.19.1 accelerate==1.12.0 peft==0.18.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install scikit-learn
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🚀 Quick Start

### 1. Model Training

We train the Epi-UQ model using HLA-I peptide sequences. The training script (`train_hla_i.py`) utilizes the ESM-2 (650M) protein language model fine-tuned with LoRA, followed by a 1D-CNN and Attention Pooling module.

#### Data Format Requirements
The training script expects input data in **CSV format** with the following columns:
- `Epitope.1`: The peptide amino acid sequence (e.g., `KLQPETQRY`).
- `Label`: Binary label (1 for presented binder, 0 for non-binder).

#### Single Model Training
To train a single model, configure the data paths and run the script. By default, the model is trained on `cuda:1` with a fixed random seed.

```bash
cd codes

# Run training script
python train_hla_i.py
```

### 2. Model Inference & Evaluation

The inference script (`infer_hla.py`) is designed to evaluate the trained Epi-UQ ensemble across multiple independent and out-of-distribution (OOD) test sets. It automatically calculates comprehensive metrics and formats the results as `Mean ± Std` for publication.

#### Data Format Requirements
The test datasets must be in **CSV format** and contain the following columns:
- **Sequence Column**: Must be named either `Epitope.1` or `Peptide`.
- **Label Column**: Must be named `Label` (1 for positive binders, 0 for negatives).

#### Configuration & Running
Before running the script, open `infer_hla.py` and update the global configuration variables at the top of the file to match your local paths:

```python
# 1. Update model and checkpoint paths
ESM_MODEL_PATH = "/path/to/facebook/esm2_t33_650M_UR50D"
CHECKPOINT_DIR = "/path/to/Advanced_baselines/checkpoints_lora"

# 2. Define the random seeds used for ensemble inference (Default: 3 models)
SEEDS = [1, 2, 3]

# 3. Define the test datasets you want to evaluate
TEST_SETS = {
    "Test_Set": "/path/to/data/HLA_I_epitope_test.csv",
    "Ext_1_Time": "/path/to/data/HLA_I_external_1_time_negative.csv",
    "Ext_4_Time": "/path/to/data/HLA_I_external_4_time_negative.csv",
    "NEPDB": "/path/to/data/NEPDB_I.csv"
}
```
