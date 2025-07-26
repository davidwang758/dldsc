import torch.optim as optim
import torch
import wandb
import hydra
from omegaconf import OmegaConf
import zarr 
import pickle
import pandas as pd
from model.mlp import MLP_Baseline, MLP_Borzoi
from model.ldsc_loss import SLDSC_Loss
from data.dataloader import GWAS_Dataset, Annotation_Dataset, DLDSC_DataLoader
#from torch.utils.data import DataLoader
#from torch.profiler import profile, record_function, ProfilerActivity

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

# Need to make parameter groups.
def train(cfg, model, criterion, train_dataloader, val_dataloader, device):
    optimizer = optim.AdamW(optimizer_params(cfg, model))
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
        epoch_val_loss = 0.0
        epoch_val_size = 0.0

        with torch.no_grad():  
            for x, y, R2, w in val_dataloader:
                x, y, R2, w = to_device(x, y, R2, w, device, non_blocking=cfg.training.pin_memory)

                yhat = model(x)
                loss = criterion(yhat, y, R2, w, model.sigma2)

                epoch_val_loss += loss.item() * y.size(0)
                epoch_val_size += y.size(0)

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
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    gwas_data = GWAS_Dataset(cfg.data.chisq)
    features = pd.read_csv(cfg.model.features, header=None)[0].values
    input_size = len(features)
    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features)
    
    R2 = zarr.open(cfg.data.R2, mode="r")
    with open(cfg.data.index, "rb") as f:
        index = pickle.load(f)
    batch_id = pd.read_csv(cfg.data.batch_id, sep="\t")
    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]
    train_batch = batch_id.loc[[i in set(cfg.training.train_chr) for i in batch_id.chr],]
    val_batch = batch_id.loc[[i in set(cfg.training.val_chr) for i in batch_id.chr],]

    # Initialize dataloaders
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

    # Initialize models and loss
    if cfg.model.model == "MLP_Borzoi":
        model = MLP_Borzoi(input_size, 1, p=cfg.training.dropout_rate).to(device)
    elif cfg.model.model == "MLP_Baseline":
        model = MLP_Baseline(input_size, 1, p=cfg.training.dropout_rate).to(device)
    else:
        raise Exception(f"{cfg.model.model} is an invalid model.")

    criterion = SLDSC_Loss()

    # Train model
    trained_model = train(cfg, model, criterion, train_dataloader, val_dataloader, device)

    # Save results
    torch.save(trained_model, f"{cfg.output.model}/{cfg.output.run_id}.pth")
    wandb.save(f"{cfg.output.logs}/{cfg.output.run_id}.wandb.pth")
    wandb.finish()
