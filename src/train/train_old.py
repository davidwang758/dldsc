import torch.optim as optim
import torch
import wandb
import hydra
from omegaconf import OmegaConf
import zarr 
import pickle
import pandas as pd
from model.mlp import MLP_Baseline, MLP_Borzoi
from model.ldsc_loss import SLDSC_Loss, SLDSC_Loss_No_R2
from data.dataloader2 import GWAS_Dataset_Chr, Annotation_Dataset, DLDSC_DataLoader, AE_DataLoader
from model.autoencoder import Autoencoder
import torch.nn as nn
#import schedulefree
from model.moe import DLDSC
import numpy as np
from train.cavi import finemap_cavi
from model.finemapping_loss import finemap_elbo
from train.ibss import susie_ibss
from model.finemapping_loss import susie_elbo
from train.ibss import multitask_susie_ibss
from model.finemapping_loss import multitask_susie_elbo
import torch.nn.functional as F
from model.finemapping import get_finetuning_model

#from torch.utils.data import DataLoader
#from torch.profiler import profile, record_function, ProfilerActivity

class MLP_Finetune(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(MLP_Finetune, self).__init__()

        original_layers = list(pretrained_model.layers.children())
        self.features = torch.nn.Sequential(*original_layers[:-1])
        self.classifier = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #x = torch.sigmoid(x)
        return x

def get_model(model_checkpoint, device="cuda"):
    pretrained_model = MLP_Baseline(187, 49).to(device)
    state_dict = torch.load(model_checkpoint, map_location=device)
    pretrained_model.load_state_dict(state_dict)
    model = MLP_Finetune(pretrained_model).to(device)
    with torch.no_grad():
        torch.nn.init.zeros_(model.classifier.weight)
        torch.nn.init.constant_(model.classifier.bias, -5.0)
    return model

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

def _finetune_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device):
    # The loaded model should be the finetunable model. 
    optimizer = optim.AdamW(model.parameters(), 
                            lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=cfg.training.t_max,
                                                     eta_min=cfg.training.eta_min)
    
    wandb.watch(model, log='all')
    print("\nStarting finetuning...")

    best_val_loss = float("inf")
    patience_counter = 0
    batch_id = np.concatenate([train_dataloader.batch_id.id.values, val_dataloader.batch_id.id.values])
    mu_init = {key: None for key in batch_id}
    alpha_init = {key: None for key in batch_id}
    tau2 = torch.tensor(np.linspace(1e-4,1e-6,10)).to(device)
    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        for x, y, R, w, _, b in train_dataloader: 
            x, y, R, w = to_device(x, y, R, w, device, non_blocking=cfg.training.pin_memory)
            #y = y.squeeze(1) # For the multitask, this doesn't have to be squeezed.

            # Get priors
            #tau2 = torch.ones(len(y)).to(device) * 1e-4
            # Use the grid strategy here.
            #pi = F.softmax(model(x).squeeze(1), dim=0)
            #pi = model(x).squeeze(1) # Same with this.
            pi = model(x)

            # Fit posterior to convergence (E step)
            with torch.no_grad(): # should add LBF output
                mu, s2, alpha = criterion[1](y, R, cfg.finemap.N, tau2, pi, mu_init=mu_init[b], alpha_init=alpha_init[b], max_iter=200, coloc=cfg.finemap.coloc)
                if cfg.training.smart_init:
                    mu_init[b] = mu.cpu()
                    alpha_init[b] = alpha.cpu()
 
            # Fit priors (M step)
            # pi should be the correct dimension
            # tau2 dimension check
            loss, _, _, _ = criterion[0](y, R, cfg.finemap.N, mu, s2, tau2, alpha, pi) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() 

        # Validation
        model.eval() 
        epoch_val_loss = 0.0
        with torch.no_grad():  
            for x, y, R, w, _, b in val_dataloader:
                x, y, R, w = to_device(x, y, R, w, device, non_blocking=cfg.training.pin_memory)
                #y = y.squeeze(1)

                #tau2 = torch.ones(len(y)).to(device) * 1e-3
                #pi = F.softmax(model(x).squeeze(1), dim=0)
                #pi = model(x).squeeze(1)
                pi = model(x)

                with torch.no_grad():
                    mu, s2, alpha = criterion[1](y, R, cfg.finemap.N, tau2, pi, mu_init=mu_init[b], alpha_init=alpha_init[b], max_iter=200, coloc=cfg.finemap.coloc)
                    if cfg.training.smart_init:
                        mu_init[b] = mu.cpu()
                        alpha_init[b] = alpha.cpu()

                loss, _, _, _ = criterion[0](y, R, cfg.finemap.N, mu, s2, tau2, alpha, pi) 
                epoch_val_loss += loss.item()

        epoch_train_loss = epoch_train_loss / cfg.training.epochs # this should divide by number of batches 
        epoch_val_loss = epoch_val_loss / cfg.training.epochs

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        log_dict = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": current_lr
        }

        wandb.log(log_dict)

        print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            best_model_state = model.state_dict() 
            print(f'Validation loss improved. Saving model state.')
        elif cfg.training.early_stopping:
            patience_counter += 1 
            print(f'Validation loss did not improve. Patience: [{patience_counter}/{cfg.training.patience}]')
            if patience_counter >= cfg.training.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break
        else:
            print(f'Validation loss did not improve.')
    
    print("Training finished.")
    return best_model_state, model.state_dict()

def _train_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device):
    #optimizer = optim.AdamW(optimizer_params(cfg, model))
    #optimizer = schedulefree.AdamWScheduleFree(optimizer_params(cfg, model))
    #optimizer = schedulefree.AdamWScheduleFree(model.parameters(), 
    #                                           lr=cfg.training.learning_rate,
    #                                           weight_decay=cfg.training.weight_decay)
    optimizer = optim.AdamW(model.parameters(), 
                            lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=cfg.training.t_max,
                                                     eta_min=cfg.training.eta_min)

    wandb.watch(model, log='all')

    print("\nStarting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        #optimizer.train() # For schedule free only
        epoch_train_loss = 0.0
        epoch_train_size = 0.0
        for x, y, R2, w in train_dataloader: 
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
        #optimizer.eval() # For schedule free only
        epoch_val_loss = 0.0
        epoch_val_size = 0.0
        epoch_val_trait_loss = torch.zeros(len(val_dataloader.traits)).to(device)

        with torch.no_grad():  
            for x, y, R2, w in val_dataloader:
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
        #for param_group in optimizer.param_groups:
        #    current_lr = param_group['lr']

        log_dict = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": current_lr
        }
        for i,t in enumerate(val_dataloader.traits):
            log_dict[t] = epoch_val_trait_loss[i].item()

        wandb.log(log_dict)

        print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            best_model_state = model.state_dict() 
            print(f'Validation loss improved. Saving model state.')
        elif cfg.training.early_stopping:
            patience_counter += 1 
            print(f'Validation loss did not improve. Patience: [{patience_counter}/{cfg.training.patience}]')
            if patience_counter >= cfg.training.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break
        else:
            print(f'Validation loss did not improve.')
        
    print("Training finished.")
    return best_model_state, model.state_dict()

def _train_autoencoder(cfg, model, criterion, train_dataloader, val_dataloader, device):
    optimizer = optim.AdamW(model.parameters(), weight_decay=cfg.training.weight_decay, lr=cfg.training.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=cfg.training.t_max,
                                                     eta_min=cfg.training.eta_min)
    wandb.watch(model, log='all')

    print("\nStarting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        epoch_train_size = 0.0
        for x in train_dataloader: 
            x = x.to(device, non_blocking=cfg.training.pin_memory)

            # Forward pass
            yhat = model(x)
            loss = criterion(yhat, x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if cfg.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_norm)
            optimizer.step()

            epoch_train_loss += loss.item() * x.size(0) 
            epoch_train_size += x.size(0)

        # Validation
        model.eval()  
        epoch_val_loss = 0.0
        epoch_val_size = 0.0

        with torch.no_grad():  
            for x in val_dataloader:
                x = x.to(device, non_blocking=cfg.training.pin_memory)

                yhat = model(x)
                loss = criterion(yhat, x)

                epoch_val_loss += loss.item() * x.size(0)
                epoch_val_size += x.size(0)

        epoch_train_loss = epoch_train_loss / epoch_train_size
        epoch_val_loss = epoch_val_loss / epoch_val_size

        scheduler.step() 
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": current_lr
        })

        print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            best_model_state = model.state_dict() 
            print(f'Validation loss improved. Saving model state.')
        elif cfg.training.early_stopping:
            patience_counter += 1 
            print(f'Validation loss did not improve. Patience: [{patience_counter}/{cfg.training.patience}]')
            if patience_counter >= cfg.training.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break
        else:
            print(f'Validation loss did not improve.')
        
    print("Training finished.")
    return best_model_state

def train_dldsc(cfg):
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
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    traits = pd.read_csv(cfg.model.traits, header=None, dtype="str")[0].values
    features = pd.read_csv(cfg.model.features, header=None, sep="\t")[0].values

    gwas_data = GWAS_Dataset_Chr(cfg.data.chisq, traits=traits)
    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features)

    input_size = len(features)
    output_size = len(traits)   
    
    R2 = zarr.open(cfg.data.R2, mode="r")
    with open(cfg.data.index, "rb") as f:
        index = pickle.load(f)

    batch_id = pd.read_csv(cfg.data.batch_id, sep="\t")
    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]
    train_batch = batch_id.loc[[i in set(cfg.training.train_chr) for i in batch_id.chr],]
    val_batch = batch_id.loc[[i in set(cfg.training.val_chr) for i in batch_id.chr],]

    # Set mode
    if cfg.training.mode == "LDSC":
        return_meta = False
    else:
        return_meta = True
    
    # Initialize dataloaders
    train_dataloader =  DLDSC_DataLoader(gwas_data, annot_data, R2, train_batch, index, 
                                         weights = None, 
                                         shuffle=cfg.training.shuffle, 
                                         num_workers=cfg.training.num_workers, 
                                         disk_cache=cfg.data.disk_cache, 
                                         pin_memory=cfg.training.pin_memory, meta=return_meta)
    val_dataloader = DLDSC_DataLoader(gwas_data, annot_data, R2, val_batch, index, 
                                      weights = None, 
                                      shuffle=cfg.training.shuffle, 
                                      num_workers=cfg.training.num_workers, 
                                      disk_cache=cfg.data.disk_cache, 
                                      pin_memory=cfg.training.pin_memory, meta=return_meta)

    # Initialize models and loss
    if cfg.model.model == "MLP_Borzoi" and cfg.training.mode == "LDSC":
        model = MLP_Borzoi(input_size, output_size, p=cfg.training.dropout_rate).to(device)
    elif cfg.model.model == "MLP_Baseline" and cfg.training.mode == "LDSC":
        model = MLP_Baseline(input_size, output_size, p=cfg.training.dropout_rate).to(device)
    elif cfg.model.model == "MoE":
        feature_splits = pd.read_csv(cfg.model.features, header=None, sep="\t")
        feature_splits = feature_splits.groupby(1)
        feature_splits = [np.array(indices.tolist()) for group, indices in feature_splits.groups.items()]
        model = DLDSC(feature_splits, output_size, no_mix_weights=cfg.training.no_mix).to(device)
    # The pretrained model needs to be adjusted for multitask.
    elif cfg.training.mode == "Finemap" and cfg.finemap.pretrained_model is not None:
        #ModelClass = get_finetuning_model(cfg.model.model, cfg.model.loss)
        #model = ModelClass(input_size, p=cfg.training.dropout_rate, trait_index=cfg.finemap.trait_index).to(device)
        #checkpoint = torch.load(cfg.finemap.pretrained_model)
        #model.load_state_dict(checkpoint) 
        # Above is old single task code, below is multitask.
        if cfg.model.model == "MLP_Baseline":
            model = MLP_Baseline(input_size, output_size, p=cfg.training.dropout_rate).to(device)
            # The rows need to sum to 1.
        elif cfg.model.model == "MLP_Borzoi":
            model = MLP_Borzoi(input_size, output_size, p=cfg.training.dropout_rate).to(device)
        else:
            raise Exception(f"{cfg.model.model} is an invalid model.")
        model.transform = lambda x: F.softmax(x, dim=0)
        checkpoint = torch.load(cfg.finemap.pretrained_model)
        model.load_state_dict(checkpoint) 
    elif cfg.training.mode == "Finemap" and cfg.finemap.pretrained_model is None:
        #ModelClass = get_finetuning_model(cfg.model.model, cfg.model.loss)
        #model = ModelClass(input_size, p=cfg.training.dropout_rate).to(device)
        if cfg.model.model == "MLP_Baseline":
            model = MLP_Baseline(input_size, output_size, p=cfg.training.dropout_rate).to(device)
        elif cfg.model.model == "MLP_Borzoi":
            model = MLP_Borzoi(input_size, output_size, p=cfg.training.dropout_rate).to(device)
        else:
            raise Exception(f"{cfg.model.model} is an invalid model.")
        model.transform = lambda x: F.softmax(x, dim=0)
    else:
        raise Exception(f"{cfg.model.model} is an invalid model.")

    if cfg.model.loss == "SLDSC":
        criterion = SLDSC_Loss()
    elif cfg.model.loss == "SLDSC_No_R2":
        criterion = SLDSC_Loss_No_R2()
    elif cfg.model.loss == "FINEMAP":
        criterion = (finemap_elbo, finemap_cavi)
    elif cfg.model.loss == "Susie": # Need to add a Susie Multitask ibss, does elbo need a multitask?
        #criterion = (susie_elbo, susie_ibss)
        criterion = (multitask_susie_elbo, multitask_susie_ibss)
    else:
        raise Exception(f"{cfg.model.lossl} is an invalid loss function.")

    # Train model
    if return_meta:
        best_model, final_model = _finetune_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device)
    else:
        best_model, final_model = _train_dldsc(cfg, model, criterion, train_dataloader, val_dataloader, device)

    # Save results
    torch.save(best_model, f"{cfg.output.model}/{wandb.run.name}.best.pth")
    torch.save(final_model, f"{cfg.output.model}/{wandb.run.name}.final.pth")
    wandb.finish()

def train_autoencoder(cfg):
    # Set up logging and configs
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb.mode,
        name=cfg.output.run_id
    )
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    features = pd.read_csv(cfg.model.features, header=None)[0].values
    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features)
    input_size = len(features)

    with open(cfg.data.index, "rb") as f:
        index = pickle.load(f)

    batch_id = pd.read_csv(cfg.data.batch_id, sep="\t")
    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]
    train_batch = batch_id.loc[[i in set(cfg.training.train_chr) for i in batch_id.chr],]
    val_batch = batch_id.loc[[i in set(cfg.training.val_chr) for i in batch_id.chr],]

    # Initialize dataloaders
    train_dataloader =  AE_DataLoader(annot_data, train_batch, index,
                                         shuffle=cfg.training.shuffle, 
                                         num_workers=cfg.training.num_workers,  
                                         pin_memory=cfg.training.pin_memory)
    val_dataloader =  AE_DataLoader(annot_data, val_batch, index,
                                         shuffle=cfg.training.shuffle, 
                                         num_workers=cfg.training.num_workers,  
                                         pin_memory=cfg.training.pin_memory)

    # Initialize models and loss
    model = Autoencoder(input_size, input_size, p=cfg.training.dropout_rate).to(device)

    criterion = nn.MSELoss()

    # Train model
    best_model, final_model = _train_autoencoder(cfg, model, criterion, train_dataloader, val_dataloader, device)

    # Save results
    torch.save(best_model, f"{cfg.output.model}/{wandb.run.name}.best.pth")
    torch.save(final_model, f"{cfg.output.model}/{wandb.run.name}.final.pth")
    wandb.finish()

    """
def train_autoencoder(cfg):
    # Set up logging and configs
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb.mode,
        name=cfg.output.run_id
    )
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    features = pd.read_csv(cfg.model.features, header=None)[0].values
    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features)
    input_size = len(features)

    with open(cfg.data.index, "rb") as f:
        index = pickle.load(f)

    batch_id = pd.read_csv(cfg.data.batch_id, sep="\t")
    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]
    train_batch = batch_id.loc[[i in set(cfg.training.train_chr) for i in batch_id.chr],]
    val_batch = batch_id.loc[[i in set(cfg.training.val_chr) for i in batch_id.chr],]

    # Initialize dataloaders
    train_dataloader =  AE_DataLoader(annot_data, train_batch, index,
                                         shuffle=cfg.training.shuffle, 
                                         num_workers=cfg.training.num_workers,  
                                         pin_memory=cfg.training.pin_memory)
    val_dataloader =  AE_DataLoader(annot_data, val_batch, index,
                                         shuffle=cfg.training.shuffle, 
                                         num_workers=cfg.training.num_workers,  
                                         pin_memory=cfg.training.pin_memory)

    # Initialize models and loss
    model = Autoencoder(input_size, input_size, p=cfg.training.dropout_rate).to(device)

    criterion = nn.MSELoss()

    # Train model
    best_model, final_model = _train_autoencoder(cfg, model, criterion, train_dataloader, val_dataloader, device)

    # Save results
    torch.save(best_model, f"{cfg.output.model}/{wandb.run.name}.best.pth")
    torch.save(final_model, f"{cfg.output.model}/{wandb.run.name}.final.pth")
    wandb.finish()
"""

#from torch.utils.data import DataLoader
#from torch.profiler import profile, record_function, ProfilerActivity
"""
class MLP_Finetune(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(MLP_Finetune, self).__init__()

        original_layers = list(pretrained_model.layers.children())
        self.features = torch.nn.Sequential(*original_layers[:-1])
        self.classifier = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #x = torch.sigmoid(x)
        return x

def get_model(model_checkpoint, device="cuda"):
    pretrained_model = MLP_Baseline(187, 49).to(device)
    state_dict = torch.load(model_checkpoint, map_location=device)
    pretrained_model.load_state_dict(state_dict)
    model = MLP_Finetune(pretrained_model).to(device)
    with torch.no_grad():
        torch.nn.init.zeros_(model.classifier.weight)
        torch.nn.init.constant_(model.classifier.bias, -5.0)
    return model
"""

"""
def _train_autoencoder(cfg, model, criterion, train_dataloader, val_dataloader, device):
    optimizer = optim.AdamW(model.parameters(), weight_decay=cfg.training.weight_decay, lr=cfg.training.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=cfg.training.t_max,
                                                     eta_min=cfg.training.eta_min)
    wandb.watch(model, log='all')

    print("\nStarting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        epoch_train_size = 0.0
        for x in train_dataloader: 
            x = x.to(device, non_blocking=cfg.training.pin_memory)

            # Forward pass
            yhat = model(x)
            loss = criterion(yhat, x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if cfg.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_norm)
            optimizer.step()

            epoch_train_loss += loss.item() * x.size(0) 
            epoch_train_size += x.size(0)

        # Validation
        model.eval()  
        epoch_val_loss = 0.0
        epoch_val_size = 0.0

        with torch.no_grad():  
            for x in val_dataloader:
                x = x.to(device, non_blocking=cfg.training.pin_memory)

                yhat = model(x)
                loss = criterion(yhat, x)

                epoch_val_loss += loss.item() * x.size(0)
                epoch_val_size += x.size(0)

        epoch_train_loss = epoch_train_loss / epoch_train_size
        epoch_val_loss = epoch_val_loss / epoch_val_size

        scheduler.step() 
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": current_lr
        })

        print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            best_model_state = model.state_dict() 
            print(f'Validation loss improved. Saving model state.')
        elif cfg.training.early_stopping:
            patience_counter += 1 
            print(f'Validation loss did not improve. Patience: [{patience_counter}/{cfg.training.patience}]')
            if patience_counter >= cfg.training.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break
        else:
            print(f'Validation loss did not improve.')
        
    print("Training finished.")
    return best_model_state
"""

"""
      elif cfg.model.model == "TabNet":
            #model = DLDSCTabNet(input_size, output_size)
            model = MLP_Baseline(input_size, output_size, p=cfg.training.dropout_rate)
        elif cfg.model.model == "MoE":
            feature_splits = pd.read_csv(cfg.model.features, header=None, sep="\t")
            feature_splits = feature_splits.groupby(1)
            feature_splits = [np.array(indices.tolist()) for group, indices in feature_splits.groups.items()]
            model = DLDSC(feature_splits, output_size, no_mix_weights=cfg.training.no_mix)
"""