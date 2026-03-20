# Deep-LDSC

![Python Version](https://img.shields.io/badge/python-3.13+-brightgreen.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.7+-orange.svg)

Deep-LDSC is a end to end model for multi-trait, genome-wide, functionally informed heritability estimation, fine-mapping, and co-localization using functional variant annotations. It can be trained on biobank scale data with thousands of GWAS traits and cell-type specific annotations derived from variant effect prediction models such as Borzoi. The main tasks our model was design for include:

1. Joint heritability estimation across multiple traits using functional annotations. 
2. Functionally informed fine-mapping across multiple complex (GWAS) and molecular (QTL) traits.
3. Functionally informed co-localization.
4. Quantifying annotation importance and functional enrichment. 

A typical D-LDSC workflow invovles:

1. Training the D-LDSC model on functional annotations and GWAS $\chi^2$ statistics to predict per-SNP heritability. The model is fit using a multitask S-LDSC objective which requires squared LD matricies to evaluate.
2. Fine-tuning the D-LDSC model on functional annotations and GWAS z-scores using a SVEM algorithm (stochastic variational expectation maximization) to predict the prior causal probability of a variant. The model is fit using multitask fine-mapping ELBO objectives (SuSiE/FINEMAP) which require signed LD matricies to evaluate. 
3. Using DeepSHAP and enrichment metrics to interpret causal variants. Our goal is to understand which annotations/cell-types are enriched for heritability and drive causal variant fine-mapping.

---

## Installation

The installation requires conda.

```bash
# 1. Clone the repository
git clone https://github.com/davidwang758/dldsc.git
cd dldsc

# 2. Create and activate a virtual environment 
conda create -n dldsc python=3.13
source activate dldsc 

# 3. Install the required packages
pip install -r requirements.txt
```

This will install PyTorch which can take a few minutes.

---

## Configuration

All of the D-LDSC training and inference settings are controlled by the `config.yaml` file. This includes model architecture, hyperparameters, data paths, and logging settings. **DO NOT** edit this config file. The parameters should all be set as command line arguments (see bash scripts in examples folder). Parameters that are already filled in are suggested defaults. They can be overidden in using command line arguments. 

Notes:
- Currently the "Finemap" objective is not supported for multitask training (only for 1 trait at a time).
- The ResNet paramters are for on-going experiments. Only use with "ResNet" as the model.

```yaml
# Configuration for D-LDSC Training

mode: # <REQUIRED> ["Train", "Inference"]: Whether to run the model in training or inference mode.

# --- Model Configuration ---
model:
  model: # <REQUIRED> ["MLP_Baseline", "MLP_Borzoi", "ResNet", "Non_Functional"]: Model for preddiction tasks from annotation.
  loss: # <REQUIRED> ["LDSC", "Susie", "Finemap"]: Loss for heritiability (LDSC) or fine-mapping tasks (Susie/Finemap).
  traits: # <REQUIRED>: Trait file containing the trait IDs the model will be trained on.
  features: # <REQUIRED>: Annotation file containing the annotation IDs the model will be trained on.

# --- Training Configuration ---
training:
  train_chr: # <REQUIRED>: Training chromosomes. Not needed for inference. For example: [1,3,5,7,9,11,13,15,17,19,21]
  val_chr: # <REQUIRED>: Validation chromosomes. Not needed for inference. For example: [2,4,6,8,10,12,14,16,18,20,22]
  epochs: 100 # Number of training epochs.
  learning_rate: 0.0001 # Learning rate.
  weight_decay: 0.01 # Weight decay (regularization).
  learning_rate_intercept: 0.0001 # Learning rate for LDSC intercept.
  weight_decay_intercept: 0.01 # Weight decay for LDSC intercept.
  dropout_rate: 0.0 # Dropout rate (regularization). Always set to 0.0 for inference.
  early_stopping: True # Stop training the model if the validation loss does not improve after "patience" epochs.
  patience: 10 # Used with "early_stopping" described above.
  gradient_clipping: True # Whether or not gradients should be clipped.
  max_norm: 1.0 # Used with "gradient_clipping" described above.
  pin_memory: True # Improve efficiency of dataloaders.
  shuffle: True # Whether batch order should be randomized.
  num_workers: 4 # Number of workers for dataloaders.
  t_max: 100 # Cosine annealing scheduler parameter.     
  eta_min: 0.000001 # Cosine annealing scheduler parameter.
  sweep: False # Set to true for hyperparameter search (see sweep.yaml)
  smart_init: True # Use posterior paramters from previous epoch to initialize the next (speeds up convergence).

finetune: 
  pretrained_model: # <OPTIONAL>: Pretrained model path. Leave blank if no model is pretrained and weights use default initializations.
  L: 10 # Max number of causal variants for SuSiE.
  min_prior_var: 1e-3 # Minimum prior variance for SuSiE. Defaults are a reasonable range of UKBB complex traits.
  max_prior_var: 1e-6 # Maximum prior variance for SuSiE. Defaults are a reasonable range of UKBB complex traits.
  max_iter: 300 # Maximum number of SuSiE iterations.
  coloc: False # Whether to share causal indicator variable across traits (shared = co-localization).
  init_threshold: 0.01 # "smart_init" threshold. Lowering this value reduces converence time but increases memory usage.

inference:
  model_list: # Trained model paths tsv file. See examples for schema.
  
# --- Data Configuration ---
data:
  N: # <REQUIRED>: Number of individuals in GWAS study or effective sample size. This is a required sufficient statistic (y^Ty). If unknown, use a large number.
  sum_stat: # <REQUIRED>: TSV with paths to summary statistic parquet files OR the summary statistic parquet file. This is a required sufficient statistic (X^Ty). See below for details.
  annotation: # <REQUIRED>: TSV with paths to annotation parquet files. See below for details.
  LD: # <REQUIRED>: LD matrix zarr file. This is a required sufficient statistic (X^TX). See below for details.
  batch_id: # <REQUIRED>: TSV with LD matrix IDs (e.g. "chr1_1000001_1300001"). Each LD matrix is one training "batch". Only IDs in this file will be used for training. 
  index: # <REQUIRED>: Index file generated through data/create_index.py script.
  disk_cache: # <REQUIRED>: Empty directory for caching LD matrices as numpy arrays. This speeds up I/O when reloading the LD matricies.

# --- Output ---
output:
  run_id: # <REQUIRED>: Output name prefix.
  dir: # <REQUIRED>: Directory to save output.

# --- Logging ---
wandb:
  project: # WANDB project (e.g. "D-LDSC"). Leave blank if not using WANDB.
  entity: # WANDB account name (e.g. "davidwang758-calico"). Leave blank if not using WANDB.
  mode: "disabled" # ["online", "offline", "disabled"]: WANDB setting. Use "disabled" if not using WANDB. 

hydra:
  run:
    dir: # <REQUIRED>: Directory to save logs.
```
---

## Data

The following data is mandatory to train a D-LDSC model.

- **Sample Size:** A scalar value for the sample size of the GWAS or QTL study. For large sample size studies, D-LDSC is not sensitive to the exact value. If the exact value is unknown but you know the sample size is large (e.g. UKBB), you can approximate this value with a large number. 
- **Summary Statistics:** Parquet file with columns ["CHR", "BP", "A1", "A2", "MAF", "INFO", "TRAIT_ID"]. These columns correspond to chromosome, variant position, reference allele, alternate allele, minor allele frequency, imputation quality score, and trait. The number of traits is only limited by memory. For a rough estimate on the limit, we trained D-LDSC on the Big40 Brain IDP dataest from Smith et al. 2021 which included 3935 traits on L4 GPU nodes with 128GB memory. The summary statistics for each trait must be $\chi^2$ for LDSC and z-scores for fine-mapping. The model assumes the individual level phenotypes and genotypes were standardized (mean 0, variance 1). $\chi^2$ is just the squared z-score. Please take care use the correct summary statistics for each task. If they are incorrect (e.g. $\chi^2$ used for fine-mapping), the model will run without error but your results will not make sense! The MAF and INFO score columns are only used for filtering, not model training. If you don't know these values, just set them to 1.0 and all variants will be used. The user can optionally supply a TSV file which contains paths to one parquet file per chromosome instead of a single parquet file. An example of the file schema is in the examples folder. 
- **Variant Annotations:** Parquet file per chromosome with columns ["CHR", "BP", "A1", "A2", "ANNOT_ID"]. These columns correspond to chromosome, variant position, reference allele, alternate allele, and annotation. The number of annotations is only limited by memory. For a rough estimate on the limit, we trained D-LDSC on ~6500 cell-type specific variant effect scores predicted by the Borzoi model. The annotation set also includes 187 baseline annotations commonly used in S-LDSC which includes measures like conservation, allele frequency, selection, etc. The user must create one parquet file per chromosome and supply a TSV with paths to each parquet file. An example of the file schema is in the examples folder. 
- **LD Matrices:** LD matricies should be provided as sparse scipy arrays (".npz") and metadata saved as compressed dataframes (".gz"). Each LD matrix is a square matrix that can be accssed by a unique ID (e.g. "chr1_1000001_1300001"). We suggest using 1-3Mb windows. The metadata columns are ["rsid", "chromosome", "position", "allele1", "allele2"]. These files must be processed into a single zarr file which is provided as input to the model. The scripts to perform this file conversion are in the data folder. The user must also supply a TSV file with columns ["chr", "id", "file", "start", "end"]. These columns are chromosome, LD matrix ID, path to sparse scipy array from which the zarr file is generated, window start, window end. During training and inference, each LD matrix is treated as a "batch". Only LD matricies which are in this TSV file will be used. 
- **Index File:** An index file is required to match variants between the 3 inputs above. Parquet is column-wise format so selection of rows is handled by this index. It is generated from the metadata in the above inputs using scripts in the data folder. Consequently, it is not required that the variants are sorted or matched across the inputs, the indexing will handle this. Variants must appear in all 3 inputs (intersection) to be used. Variants can be filtered by MAF and INFO score.

---

## Usage

### Training

Example D-LDSC training script. Set variables to input paths.

```bash
python dldsc.py mode='Train' \
                model.model='MLP_Baseline' \
                model.loss='LDSC' \
                model.traits='examples/all_traits.txt' \
                model.features='examples/scores.baseline.txt' \
                training.train_chr=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] \
                training.val_chr=[1,22] \
                data.N=$N \
                data.sum_stat=$sum_stat \
                data.annotation=$annotation \
                data.LD=$LD \
                data.batch_id=$batch_id \
                data.index=$index \
                data.disk_cache=$disk_cache \
                output.run_id='example_train' \
                output.dir=$out_dir \
                hydra.run.dir="${out_dir}/logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
```

Example D-LDSC fine-tuning script. Set variables to input paths.

```bash
python dldsc.py mode='Train' \
                model.model='MLP_Baseline' \
                model.loss='Susie' \
                model.traits='examples/all_traits.txt' \
                model.features='examples/scores.baseline.txt' \
                training.train_chr=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] \
                training.val_chr=[1,22] \
                finetune.pretrained_model="${out_dir}/example_train.best.pth" \
                data.N=$N \
                data.sum_stat=$sum_stat \
                data.annotation=$annotation \
                data.LD=$LD \
                data.batch_id=$batch_id \
                data.index=$index \
                data.disk_cache=$disk_cache \
                output.run_id='example_finetune' \
                output.dir=$out_dir \
                hydra.run.dir="${out_dir}/logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
```
---

### Inference

Example D-LDSC inference script for heritability estimation. Set variables to input paths.

```bash
python dldsc.py mode='Inference' \
                model.model='MLP_Baseline' \
                model.loss='LDSC' \
                model.traits='examples/all_traits.txt' \
                model.features='examples/scores.baseline.txt' \
                data.N=$N \
                data.sum_stat=$sum_stat \
                data.annotation=$annotation \
                data.LD=$LD \
                data.batch_id=$batch_id \
                data.index=$index \
                data.disk_cache=$disk_cache \
                inference.model_list=$model_list \
                output.dir=$out_dir \
                hydra.run.dir="${out_dir}/logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
```

Example D-LDSC inference script for fine-mapping. Set variables to input paths.

```bash
python dldsc.py mode='Inference' \
                model.model='MLP_Baseline' \
                model.loss='Susie' \
                model.traits='examples/all_traits.txt' \
                model.features='examples/scores.baseline.txt' \
                data.N=$N \
                data.sum_stat=$sum_stat \
                data.annotation=$annotation \
                data.LD=$LD \
                data.batch_id=$batch_id \
                data.index=$index \
                data.disk_cache=$disk_cache \
                inference.model_list=$model_list \
                output.dir=$out_dir \
                hydra.run.dir="${out_dir}/logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
```

Alternatively, you can also do non-functional fine-mapping. Simply set "MLP_Baseline" to "Non_Functional". The list of trained models "model_list" is NOT required for non-functional fine-mapping. Under the hood, this essentially does 1 forward pass of the model with uniform priors. 

Notes:
- The outputs of non-functional fine-mapping should be near identical to susie_rss. D-LDSC does NOT fit the residual and prior variances of the SuSiE model. Instead, the residual variance is fixed at 1.0. The prior variances for each single effect component is set to the a value chosen unifromly from a grid on the range [min_prior_var, max_prior_var]. Fitting these parameters is very expensive, but SuSiE is not sensitive to the precise values for these parameters. They just need to be within a reasonable range. The default values we provide are sufficient for UKBB traits. If you dataset has very, very, very large effect variants (a single variant explains > 5% heritability or the effect size is extremely large such that prior variance is > 0.01), you might notice small, often negligible, shifts in PIPs compared to susise_rss.   

---

### Output

- Running D-LDSC in "Training" mode will output 2 model weight files. They are "run_id.best.pth" and "run_id.final.pth". Best corresponds to the model weights at the epoch with the lowest validation loss. Final corresponds to the model weights after the last epoch.

- Running D-LDSC in "Inference" mode will output 1 parquet file per "batch" which corresponds to a window defined by a LD matrix. It will also output a "loss.tsv" file which records the model loss for each batch. If the loss is "LDSC", the model will perform heritabliity inference. Each column will be a trait containing per-SNP heritability estimates. The total heritability of the trait is the sum of the per-SNP heritabilities. If the loss is "Susie", the model will perform fine-mapping. Each column will be a trait with suffix "_PIP", "_mu", and "_prior" and "_CS" containing PIP, posterior effect size, prior causal proability, and credible set. 

### Analysis

1. Fine-mapping + DeepSHAP.

Examples to plot fine-mapping results and interpret variants using DeepSHAP are in notebooks in the example folder.

2. Heritability + functional enrichment. 

Examples to plot heritability enrichment for binary and continous annotations are in notebooks in the examples folder.

## TODO:

- Documentation for data processing scripts.
- Documentation for sweeps for hyperparamter tuning.
- Co-localization examples and documentation. It works but needs more evaluations on our end.
- Move trained models and some data to GCP buckets for easier sharing.



