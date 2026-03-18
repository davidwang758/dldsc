import numpy as np
import pandas as pd
import torch
import zarr 
import pickle
import fsspec
import logging

from omegaconf import OmegaConf

from src.data.dataloader import GWAS_Dataset, GWAS_Dataset_Chr, Annotation_Dataset, DLDSC_DataLoader
from src.model.mlp import MLP_Baseline, MLP_Borzoi
from src.train.cavi import finemap_cavi
from src.train.ibss import multitask_susie_ibss
from src.model.ldsc_loss import LDSC_Loss
from src.model.finemapping_loss import multitask_susie_elbo

class MLP_Identity(torch.nn.Module):
    def __init__(self, output_dim):
        super(MLP_Identity, self).__init__()
        self.output_dim = output_dim

    def forward(self, x):
        return torch.ones((x.shape[0], self.output_dim)).to(x.device) / x.shape[0]

def to_device(x, y, R2, w, device, non_blocking=False):
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)
    R2 = R2.to(device, non_blocking=non_blocking)
    w = w.to(device, non_blocking=non_blocking)
    return x, y, R2, w

def _run_finemapping_inference(cfg, model_list, dataloader_list, criterion, device, log):
    tau2 = torch.linspace(cfg.finetune.max_prior_var,cfg.finetune.min_prior_var,cfg.finetune.L, device=device)
    loss_list = []
    batch_list = []

    log.info("\nStarting finemapping inference...")
    for model, dataloader in zip(model_list, dataloader_list):
        model.eval()
        with torch.no_grad():
            for x, y, R, w, b, m in dataloader: 
                log.info(f"Batch: {b}")
                x, y, R, w = to_device(x, y, R, w, device, non_blocking=cfg.training.pin_memory)

                pi = model(x) 

                # Fit posterior to convergence (E step)
                mu, s2, alpha = criterion[1](y, R, cfg.data.N, tau2, pi, max_iter=cfg.finetune.max_iter, device=device, L=cfg.finetune.L)
                loss, _, _, _ = criterion[0](y, R, cfg.data.N, mu, s2, tau2, alpha, pi)
            
                loss_list.append(loss.item())
                batch_list.append(b)
                pip = 1 - torch.prod(1 - alpha,axis=0)
                post_mu = torch.sum(mu * alpha, axis=0)
                pip = pd.DataFrame(pip.half().detach().cpu().numpy())
                post_mu = pd.DataFrame(post_mu.half().detach().cpu().numpy())
                pip.columns = dataloader.traits + "_PIP"
                post_mu.columns = dataloader.traits + "_mu"
                prior = pd.DataFrame(pi.half().detach().cpu().numpy())
                prior.columns = dataloader.traits + "_prior"
                m = pd.concat([m, pip, post_mu, prior], axis=1)
                m.to_parquet(f"{cfg.output.dir}/{b}.parquet", engine='pyarrow', index=False)

            loss_df = pd.DataFrame({"Batch": batch_list, "Loss": loss_list})
            loss_df.to_csv(f"{cfg.output.dir}/loss.tsv", sep="\t", index=False)

def _run_dldsc_inference(cfg, model_list, dataloader_list, criterion, device, log):
    loss_list = []
    batch_list = []
    log.info("\nStarting heritability inference...")
    for model, dataloader in zip(model_list, dataloader_list):
        model.eval()
        with torch.no_grad():
            for x, y, R2, w, b, m in dataloader: 
                log.info(f"Batch: {b}")
                x, y, R2, w = to_device(x, y, R2, w, device, non_blocking=cfg.training.pin_memory)

                yhat = model(x) 

                loss = criterion(yhat, y, R2, w, model.sigma2, trait_specific=False)
            
                loss_list.append(loss.item())
                batch_list.append(b)
                h2 = yhat # This should be divided by the number of variants.
                h2 = pd.DataFrame(h2.half().detach().cpu().numpy())
                h2.columns = dataloader.traits
                m = pd.concat([m, h2], axis=1)
                m.to_parquet(f"{cfg.output.dir}/{b}.parquet", engine='pyarrow', index=False)

            loss_df = pd.DataFrame({"Batch": batch_list, "Loss": loss_list})
            loss_df.to_csv(f"{cfg.output.dir}/loss.tsv", sep="\t", index=False)

def predict(cfg):
    log = logging.getLogger(__name__)

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    traits = pd.read_csv(cfg.model.traits, header=None, dtype="str")[0].values
    features = pd.read_csv(cfg.model.features, header=None, sep="\t")[0].values

    if cfg.data.sum_stat.endswith(".parquet"): 
        gwas_data = GWAS_Dataset(cfg.data.sum_stat, traits=traits)
    else:
        gwas_data = GWAS_Dataset_Chr(pd.read_csv(cfg.data.sum_stat, sep="\t"), traits=traits)
    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features)

    input_size = len(features)
    output_size = len(traits) 

    R2 = zarr.open(cfg.data.LD, mode="r")
    with fsspec.open(cfg.data.index, "rb") as f:
        index = pickle.load(f)

    batch_id = pd.read_csv(cfg.data.batch_id, sep="\t")
    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]

    if cfg.model.model == "Non_Functional":
        model_list = [MLP_Identity(output_size)]
        batch_list = [batch_id]
    elif cfg.model.model == "MLP_Baseline":
        model_file = pd.read_csv(cfg.inference.model_list, sep="\t")
        model_list = []
        model_chr = [[int(x) for x in chrom.split(",")] for chrom in model_file.CHR]
        activation = "softmax" if cfg.model.loss == "Susie" else "softplus"
        for model_path in model_file.MODEL:
            model = MLP_Baseline(input_size, output_size, p=cfg.training.dropout_rate, output_activation=activation).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model_list.append(model)
        batch_list = [batch_id.loc[[i in j for i in batch_id.chr],] for j in model_chr]
    else:
       raise ValueError(f"{cfg.model.model} is an invalid model.")

    # Initialize dataloaders
    dataloader_list =  [DLDSC_DataLoader(gwas_data, annot_data, R2, batch, index, 
                                         weights = None, 
                                         shuffle=cfg.training.shuffle, 
                                         num_workers=cfg.training.num_workers, 
                                         disk_cache=cfg.data.disk_cache, 
                                         pin_memory=cfg.training.pin_memory, meta=True) 
                                         for batch in batch_list]

    if cfg.model.loss == "LDSC":
        criterion = LDSC_Loss()
    elif cfg.model.loss == "Finemap":
        criterion = finemap_cavi
    elif cfg.model.loss == "Susie":
        criterion = (multitask_susie_elbo, multitask_susie_ibss)
    else:
        raise ValueError(f"{cfg.model.loss} is an invalid loss.")
    
    # Train model
    if cfg.model.loss == "LDSC":
        _run_dldsc_inference(cfg, model_list, dataloader_list, criterion, device, log)
    elif cfg.model.loss == "Susie" or cfg.model.loss == "Finemap":
        _run_finemapping_inference(cfg, model_list, dataloader_list, criterion, device, log) 
    else:
        raise ValueError(f"{cfg.model.loss} is an invalid loss.")
    
    