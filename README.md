# Deep LDSC

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-1.10+-orange.svg)

Tools for Deep LDSC training, evaluation, and inference. Currently supports MLP models with LDSC loss.

---

## Installation

To get started, clone the repository and install the required dependencies. It is highly recommended to use a virtual environment.

```bash
# 1. Clone the repository
git clone [https://github.com/your_username/your_repository.git](https://github.com/your_username/your_repository.git)
cd your_repository

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install the required packages
pip install -r requirements.txt
```

---

## Configuration

All aspects of the training process are controlled by the `config.yaml` file. This includes data paths, model architecture, hyperparameters, and logging settings.

Before running, review and edit `config.yaml` to match your setup.

**Example `config.yaml` snippet:**

```yaml
# Configuration for D-LDSC Training

# --- Model Configuration ---
model:
  model: ["MLP_Borzoi", "MLP_Baseline"] # Choose model based on input annotations.
  features: "all_features.txt" # List of annotations to include in model.

# --- Training Configuration ---
training:
  epochs: 100 
  learning_rate: 0.001
  weight_decay: 0.01
  learning_rate_intercept: 0.01 # LDSC intercept parameter
  weight_decay_intercept: 0.0
  dropout_rate: 0.0
  early_stopping: False
  patience: 10
  gradient_clipping: True
  max_norm: 1.0
  pin_memory: True
  shuffle: True
  num_workers: 4
  t_max: 100  # Total epochs for Cosine Annealing scheduler    
  eta_min: 0.000001 # Minimum learning rate for scheduler
  train_chr: [1,3,5,7,9,11,13,15,17,19,21]
  val_chr: [2,4,6,8,10,12,14,16,18,20,22]
  
# --- Data Configuration ---
data:
  chisq: "trait.parquet" # See below for formats
  annotation: "annotation.meta.tsv"
  R2: "ld_matrix.zarr"
  batch_id: "R2.meta.tsv"
  index: "BMI_MAF:0.01_X2:80_INFO:0.6_SQ.idx"
  disk_cache: "/tmp/BMI_MAF:0.01_X2:80_INFO:0.6_SQ" # Cache the LD matrices as npy arrays on a faster disk.

# --- Output ---
output:
  run_id: "BMI_MAF:0.01_X2:80_INFO:0.6_SQ_ALL"
  model: "output/models"
  logs: "output/logs"

# --- Logging ---
wandb:
  project: "D-LDSC"
  entity: "davidwang758-calico"
  mode: "online" # [online, offline, disabled]

hydra:
  run:
    dir: logs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```
---

## Data

The following data is mandatory to train a D-LDSC model.

- **GWAS CHISQ Summary Statistics:** Parquet file with columns ["CHR", "BP", "A1", "A2", "MAF", "CHISQ", "INFO"]. 
These are chromosome, position, reference allele, alternate allele, minor allele frequency, Chisq summary statistic, and INFO score. 
- **Variant Annotations:** One Parquet file per chromosome with columns ["CHR", "BP", "A1", "A2"] and 1 named column for each annotation. 
Meta data is stored annotation.meta.tsv with columns ["chr", "parquet"]. These are the chromsome, and full path to each parquet file.
- **LD Matrices:** One Zarr file. Each LD matrix is a square matrix that can be accssed by a unique ID. 
Meta data is stored in R2.meta.tsv with columns ["chr", "id", "file", "start", "end"].
These are chromosome, LD matrix id, path to sparse array from which the Zarr file is generated, window start, window end.

Utility functions to convert UKBB raw data into these formats can be found in /data/ukbb/process_ukbb.py.

Given these datasets, first build a index file using dataloader.build_index before running the training script.

---

## Usage

### Training

To start a new training run, execute the main training script:

```bash
python dldsc.py
```

The script will automatically load the settings from `config.yaml`.

#### Overriding Configuration from Command Line

You can override any parameter from the `config.yaml` file directly from the command line. This is useful for running sweeps or quick experiments.

```bash
# Example: Run with a different learning rate
python dldsc.py training.learning_rate=0.01
```

### Inference

TODO


