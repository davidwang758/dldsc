#!/bin/bash

### CHANGE ME ###
out_dir='/scratch4/davidwang/examples/output'

### Example UKBB Data ###
N=337000
sum_stat='/scratch4/davidwang/datasets/ukbb/gwas/all_traits.parquet'
annotation='/scratch4/davidwang/datasets/ukbb/annotation/all/scores.meta.tsv'
LD='/scratch4/davidwang/datasets/ukbb/ld_matrix/ukbb.zarr'
batch_id='/scratch4/davidwang/datasets/ukbb/ld_matrix/LD_example.meta.tsv'
index='/scratch4/davidwang/datasets/ukbb/index/ALL_MAF:0.01_X2:80_INFO:0.6_SQ.idx'
disk_cache="${out_dir}/ALL_MAF:0.01_X2:80_INFO:0.6_SQ"
model_list="/scratch4/davidwang/datasets/ukbb/models/ALL_MAF:0.01_X2:80_INFO:0.6_SQ_BASELINE_LR:1e-4.tsv"

cd ..

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