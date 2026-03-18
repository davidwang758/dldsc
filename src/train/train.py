import numpy as np
import pandas as pd
import os
import zarr 
import pickle
import fsspec
import logging

import torch
import torch.optim as optim

import wandb
from omegaconf import OmegaConf

from src.model.mlp import MLP_Baseline, MLP_Borzoi
from src.model.resnet import TabularResNet
from src.model.ldsc_loss import LDSC_Loss
from src.data.dataloader import GWAS_Dataset, GWAS_Dataset_Chr, Annotation_Dataset, DLDSC_DataLoader
from src.train.cavi import finemap_cavi
from src.model.finemapping_loss import finemap_elbo
from src.train.ibss import multitask_susie_ibss
from src.model.finemapping_loss import multitask_susie_elbo

def to_device(x, y, R2, w, device, non_blocking=False):
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)
    R2 = R2.to(device, non_blocking=non_blocking)
    w = w.to(device, non_blocking=non_blocking)
    return x, y, R2, w

def optimizer_params(cfg, model):
    group_params = [
        {'params': model.layers.parameters(), 'weight_decay': cfg.training.weight_decay, 'lr': cfg.training.learning_rate},
        {'params': model.sigma2, 'weight_decay': cfg.training.weight_decay_intercept, 'lr': cfg.training.learning_rate_intercept}
    ]
    return group_params

def _get_init(x, threshold):
    diff = x.abs().max(dim=1)[0]
    l_init, v_init = torch.where(diff >= threshold)
    return l_init, v_init, x[l_init, :, v_init]

def _finetune_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device, log):
    # The loaded model should be the finetunable model. 
    optimizer = optim.AdamW(model.parameters(), 
                            lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=cfg.training.t_max,
                                                     eta_min=cfg.training.eta_min)
    
    wandb.watch(model, log='all')
    log.info("\nStarting finetuning...")

    best_val_loss = float("inf")
    patience_counter = 0
    batch_id = np.concatenate([train_dataloader.batch_id.id.values, val_dataloader.batch_id.id.values])
    n_train_batches = len(train_dataloader.batch_id.id.values)
    n_val_batches = len(val_dataloader.batch_id.id.values)
    Eb_init = {key: None for key in batch_id}
    tau2 = torch.linspace(cfg.finetune.max_prior_var,cfg.finetune.min_prior_var,cfg.finetune.L, device=device)
    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        for x, y, R, w, b in train_dataloader:
            x, y, R, w = to_device(x, y, R, w, device, non_blocking=cfg.training.pin_memory)

            pi = model(x)

            # Fit posterior to convergence (E step)
            with torch.no_grad(): 
                mu, s2, alpha = criterion[1](y, R, cfg.data.N, tau2, pi, Eb_init=Eb_init[b], max_iter=cfg.finetune.max_iter, coloc=cfg.finetune.coloc, L=cfg.finetune.L, device=device)
                if cfg.training.smart_init:
                    Eb = (mu * alpha).cpu()
                    l_init, v_init, val = _get_init(Eb, cfg.finetune.init_threshold)
                    Eb_init[b] = (l_init, v_init, val.cpu().half())
 
            # Fit priors (M step)
            loss, _, _, _ = criterion[0](y, R, cfg.data.N, mu, s2, tau2, alpha, pi) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() 

        # Validation
        model.eval() 
        epoch_val_loss = 0.0
        with torch.no_grad():  
            for x, y, R, w, b in val_dataloader:
                x, y, R, w = to_device(x, y, R, w, device, non_blocking=cfg.training.pin_memory)

                pi = model(x)

                with torch.no_grad():
                    mu, s2, alpha = criterion[1](y, R, cfg.data.N, tau2, pi, Eb_init=Eb_init[b], max_iter=cfg.finetune.max_iter, coloc=cfg.finetune.coloc, L=cfg.finetune.L, device=device)
                    if cfg.training.smart_init:
                            Eb = mu * alpha
                            l_init, v_init, val = _get_init(Eb, cfg.finetune.init_threshold)
                            Eb_init[b] = (l_init.cpu(), v_init.cpu(), val.cpu().half())

                loss, _, _, _ = criterion[0](y, R, cfg.data.N, mu, s2, tau2, alpha, pi) 
                epoch_val_loss += loss.item()

        epoch_train_loss = epoch_train_loss / n_train_batches # this should divide by number of batches 
        epoch_val_loss = epoch_val_loss / n_val_batches

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        log_dict = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": current_lr
        }

        wandb.log(log_dict)

        log.info(f"Epoch [{epoch+1}/{cfg.training.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            best_model_state = model.state_dict() 
            log.info(f'Validation loss improved. Saving model state.')
        elif cfg.training.early_stopping:
            patience_counter += 1 
            log.info(f'Validation loss did not improve. Patience: [{patience_counter}/{cfg.training.patience}]')
            if patience_counter >= cfg.training.patience:
                log.info(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break
        else:
            log.info(f'Validation loss did not improve.')
    
    log.info("Training finished.")
    return best_model_state, model.state_dict()

def _train_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device, log):
    optimizer = optim.AdamW(model.parameters(), 
                            lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=cfg.training.t_max,
                                                     eta_min=cfg.training.eta_min)

    wandb.watch(model, log='all')

    log.info("\nStarting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        epoch_train_size = 0.0
        for x, y, R2, w, _ in train_dataloader: 
            if x.shape[0] < 2:
                continue

            x, y, R2, w = to_device(x, y, R2, w, device, non_blocking=cfg.training.pin_memory)
            
            # Forward pass
            yhat = model(x)
            loss = criterion(yhat, y, R2, w, model.sigma2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if cfg.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_norm)
            optimizer.step()

            epoch_train_loss += loss.item() * y.size(0) 
            epoch_train_size += y.size(0)

        # Validation
        model.eval()  
        epoch_val_loss = 0.0
        epoch_val_size = 0.0
        epoch_val_trait_loss = torch.zeros(len(val_dataloader.traits)).to(device) # Possibly keep this off device

        with torch.no_grad():  
            for x, y, R2, w, _ in val_dataloader:
                if x.shape[0] < 2:
                    continue
                x, y, R2, w = to_device(x, y, R2, w, device, non_blocking=cfg.training.pin_memory)

                yhat = model(x)

                loss, trait_loss = criterion(yhat, y, R2, w, model.sigma2, trait_specific=True)

                epoch_val_trait_loss += trait_loss * y.size(0)
                epoch_val_loss += loss.item() * y.size(0)
                epoch_val_size += y.size(0)

        epoch_train_loss = epoch_train_loss / epoch_train_size
        epoch_val_loss = epoch_val_loss / epoch_val_size
        epoch_val_trait_loss = epoch_val_trait_loss / epoch_val_size

        scheduler.step() 
        current_lr = scheduler.get_last_lr()[0]

        log_dict = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": current_lr
        }
        for i,t in enumerate(val_dataloader.traits):
            log_dict[t] = epoch_val_trait_loss[i].item()

        wandb.log(log_dict)

        log.info(f"Epoch [{epoch+1}/{cfg.training.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            best_model_state = model.state_dict() 
            log.info(f'Validation loss improved. Saving model state.')
        elif cfg.training.early_stopping:
            patience_counter += 1 
            log.info(f'Validation loss did not improve. Patience: [{patience_counter}/{cfg.training.patience}]')
            if patience_counter >= cfg.training.patience:
                log.info(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break
        else:
            log.info(f'Validation loss did not improve.')
        
    log.info("Training finished.")
    return best_model_state, model.state_dict()

def get_model(cfg, input_size, output_size):
    activation = "softmax" if cfg.model.loss == "Susie" else "softplus"
    if cfg.model.model == "MLP_Borzoi":
        model = MLP_Borzoi(input_size, output_size, p=cfg.training.dropout_rate, output_activation=activation)
    elif cfg.model.model == "MLP_Baseline":
        model = MLP_Baseline(input_size, output_size, p=cfg.training.dropout_rate, output_activation=activation)
    elif cfg.model.model == "ResNet":
        model = TabularResNet(input_size, cfg.resnet.hidden_dim, cfg.resnet.num_blocks, output_size, dropout_rate=cfg.training.dropout_rate)
    else:
        raise ValueError(f"{cfg.model.model} is an invalid model.")
        
    if cfg.finetune.pretrained_model is not None:
        checkpoint = torch.load(cfg.finetune.pretrained_model)
        model.load_state_dict(checkpoint) 

    return model

def get_loss(cfg):
    if cfg.model.loss == "LDSC":
        criterion = LDSC_Loss()
    elif cfg.model.loss == "Finemap":
        # Currently does not work for multitask
        criterion = (finemap_elbo, finemap_cavi)
    elif cfg.model.loss == "Susie": 
        criterion = (multitask_susie_elbo, multitask_susie_ibss)
    else:
        raise ValueError(f"{cfg.model.loss} is an invalid loss function.")
    return criterion

def train(cfg):
    # Set up logging and configs
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb.mode,
        name=cfg.output.run_id
    )

    if cfg.training.sweep:
        custom_name = f"WD:{cfg.training.weight_decay}_DO:{cfg.training.dropout_rate}"
        wandb.run.name = custom_name
    
    log = logging.getLogger(__name__)

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    traits = pd.read_csv(cfg.model.traits, header=None, dtype="str")[0].values
    features = pd.read_csv(cfg.model.features, header=None, sep="\t")[0].values

    # Check if using old or new GWAS format
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
    train_batch = batch_id.loc[[i in set(cfg.training.train_chr) for i in batch_id.chr],]
    val_batch = batch_id.loc[[i in set(cfg.training.val_chr) for i in batch_id.chr],]

    # Initialize dataloaders
    if not os.path.exists(cfg.data.disk_cache):
        os.makedirs(cfg.data.disk_cache)

    train_dataloader =  DLDSC_DataLoader(gwas_data, annot_data, R2, train_batch, index, 
                                         weights = None, 
                                         shuffle=cfg.training.shuffle, 
                                         num_workers=cfg.training.num_workers, 
                                         disk_cache=cfg.data.disk_cache, 
                                         pin_memory=cfg.training.pin_memory)
    val_dataloader = DLDSC_DataLoader(gwas_data, annot_data, R2, val_batch, index, 
                                      weights = None, 
                                      shuffle=cfg.training.shuffle, 
                                      num_workers=cfg.training.num_workers, 
                                      disk_cache=cfg.data.disk_cache, 
                                      pin_memory=cfg.training.pin_memory)


    # Choose model and loss
    model = get_model(cfg, input_size, output_size).to(device)
    criterion = get_loss(cfg)

    # Train model
    if cfg.model.loss == "LDSC":
        best_model, final_model = _train_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device, log)
    elif cfg.model.loss == "Susie" or cfg.model.loss == "Finemap":
        best_model, final_model = _finetune_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device, log)
    else:
        raise ValueError(f"{cfg.model.loss} is an invalid loss.")
    
    # Save results
    torch.save(best_model, f"{cfg.output.dir}/{cfg.output.run_id}.best.pth")
    torch.save(final_model, f"{cfg.output.dir}/{cfg.output.run_id}.final.pth")
    wandb.finish()
