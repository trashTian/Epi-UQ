# Epi-UQ: Parameter-Efficient, Allele-Agnostic Epitope Prediction with Uncertainty Quantification

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

Model Training
