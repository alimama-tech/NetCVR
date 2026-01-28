# Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Insights and Solutions

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)]()

Here is the official implenmentation of our WWW 2026 paper [*Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Insights and Solutions*](./NetCVR-TESLA-paper.pdf).
This repository provides a comprehensive benchmark and open-source toolkit for **Modeling Cascaded Delay Feedback** in online **Net Conversion Rate (NetCVR)** prediction.

In this work, we present systematic insights into the cascading nature of delayed feedback signals and propose effective modeling solutions. This codebase includes datasets, models, training pipelines, and evaluation tools to support future research in delay feedback modeling.

---


## 📦 Dataset

The experiments are based on a large-scale industrial dataset from Alibaba, capturing multi-stage user behaviors including click, add-to-cart, payment, and refund, with precise timestamps for modeling delay dynamics.

👉 **Dataset Information**:  
[CASCADE Dataset]() *(Coming Soon)*

> 🔒 Note: Due to data privacy policies, the full raw dataset cannot be publicly released. A processed benchmark version with anonymized features will be made available for research purposes.

📁 Data structure includes:
- User/item/Related Features
- Timestamps for each conversion stage (`click_time`, `pay_time`, `refund_time`)

data source should be placed under `data/CASCADE/`.

---


## 🧪 Baseline Models

Below are the baseline models included in this benchmark, along with their original paper references and corresponding implementation scripts.      


| Model Name                                                                 | Model Reference Script                   |
|----------------------------------------------------------------------------|------------------------------------------|
| [ESDFM](https://arxiv.org/abs/2012.03245)                                  | `ali_reesdfm_stream_pretrain.py`            |
| [MISS](https://dl.acm.org/doi/10.1609/aaai.v38i8.28726)                     | `ali_remiss_stream_train.py`             |
| [DFSN](https://dl.acm.org/doi/10.1145/3539618.3591747)                      | `ali_redfsn_stream_train.py`             |
| Oracle                                                                     | `ali_reoracle_stream_train.py`              |
| [FNW](https://dl.acm.org/doi/10.1145/3298689.3347002)                       | `ali_refnw_stream_train.py`              |
| [FNC]()                                 | `ali_refnc_stream_train.py`              |
| [Defuse](https://dl.acm.org/doi/10.1145/3485447.3511965)                    | `ali_redefuse_stream_train.py`           |
| [Defer](https://arxiv.org/abs/2012.03245)                                   | `ali_redefer_stream_train.py`            |
| [DDFM](https://dl.acm.org/doi/10.1145/3583780.3614856)                      | `ali_reddfm_stream_train.py`             |
| [TESLA (Ours)](./NetCVR-TESLA-paperpdf)                             | `ali_TESLA_stream_train.py`              |



## 📁 Project Structure

```bash
AirBench4OpenSource/
├── data/               # Raw and metadata files
├── dataloader/         # Custom data loading modules
├── datasets/           # Dataset classes and preprocessing scripts
├── log/                # Training logs and evaluation outputs
├── models/             # Model architectures (e.g., CascadeNet, ESDFM)
├── mx_utils/           # Utility functions: metrics, config, logging, etc.
├── trainers/           # Training and evaluation logic
├── examples/           # Example scripts for quick start
├── requirements.txt    # Required Python packages
├── README.md           # This file
└── LICENSE             # MIT License

```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone git@gitlab.alibaba-inc.com:CASCADE/AirBench4OpenSource.git
cd AirBench4OpenSource

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. processing data

Download [CASCADE Dataset]() and process it by using scripts below or directly download the processed version from the [link]() above and place it under `data/CASCADE/`. (coming soon)
```bash
# to process data
python process_CASCADE_with_MappingDict.py
```

### 3. run an example script
To run the main training script for our model, use:
```bash
# to direct run our model 
python AirBench4OpenSource/ali_TESLA_stream_train.py
```
Specifically, you need to run the following pre-training scripts in advance:
```bash
# Step 1: Pre-train the base model
python AirBench4OpenSource/ali_esdfmRf_PLE_pretrain.py

# Step 2: Pre-train the inw-tn-pay delay feedback model
python AirBench4OpenSource/ali_esdfmRF_inw_tn_pay_pretrain.py

# Step 3: Pre-train the inw-tn-refund delay feedback model
python AirBench4OpenSource/ali_esdfmRF_inw_tn_refund_pretrain.py

```
These scripts will generate the necessary checkpoint files (model weights), which are then loaded by ali_TESLA_stream_train.py during training.
More usage examples can be found in the scripts under the examples/ directory.
