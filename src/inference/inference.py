import numpy as np
import pandas as pd
import torch
import zarr 
import pickle
import fsspec
import logging
from crick import TDigest

from omegaconf import OmegaConf

from src.data.dataloader import GWAS_Dataset, GWAS_Dataset_Chr, Annotation_Dataset, DLDSC_DataLoader, Annot_DataLoader
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

def compute_enrichment_metric(h2, annot):
    h2c = annot.T @ h2 
    asum = torch.sum(annot, axis=0).unsqueeze(1)
    return [h2c, asum]

def get_cs_table(alpha, R, coverage=0.95, ld_threshold=0.99, purity_threshold=0.5):
    # Compute CS
    sorted_indicies = torch.argsort(alpha, dim=1, descending=True)
    sorted_probs = torch.gather(alpha, dim=1, index=sorted_indicies)
    cum_sums = torch.cumsum(sorted_probs, dim=1)
    shifted_cum_sums = cum_sums - sorted_probs
    mask = shifted_cum_sums < coverage
    cs = mask.nonzero(as_tuple=False)
    cs[:,1] = sorted_indicies[mask]

    # Filter duplicate CS
    df = pd.DataFrame(cs.cpu().numpy(), columns=["L","V","T"])
    grouped = df.groupby(["L","T"])["V"].apply(frozenset).reset_index()
    grouped = grouped.drop_duplicates(subset=["T","V"])

    # Extend cs by high ld variants and compute purity
    R_abs = torch.abs(R) 
    
    new_cs = []
    purities = []
    
    for i, cs_frozenset in enumerate(grouped.V):
        curr_indices = torch.tensor(list(cs_frozenset), device=R.device, dtype=torch.long)
        
        max_ld_per_variant, _ = torch.max(R_abs[curr_indices, :], dim=0)
        extended_indices = torch.where(max_ld_per_variant > ld_threshold)[0]
        
        sub_R = R_abs[extended_indices][:, extended_indices]
        purity = torch.min(sub_R).item()
            
        new_cs.append(extended_indices.cpu().numpy().tolist())
        purities.append(purity)

    # Drop low purity cs
    grouped["V"] = new_cs
    grouped["purity"] = purities
    grouped = grouped.loc[grouped["purity"] >= purity_threshold]

    # Reformat for output. 
    exploded = grouped[['L', 'T', 'V']].explode('V')
    exploded['cs_id'] = exploded['L'] + 1

    matrix_df = exploded.pivot_table(index='V', 
                                 columns='T', 
                                 values='cs_id', 
                                 aggfunc='last').fillna(0).astype(int)
    
    full_idx = np.arange(alpha.shape[1])
    full_cols = np.arange(alpha.shape[2])
    
    matrix_df = matrix_df.reindex(index=full_idx, columns=full_cols, fill_value=0)

    return(matrix_df)

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

                # Get credible sets
                cs_table = get_cs_table(alpha, R)
                cs_table.columns = dataloader.traits + "_CS"
            
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
                m = pd.concat([m, pip, cs_table, post_mu, prior], axis=1)
                m.to_parquet(f"{cfg.output.dir}/{b}.parquet", engine='pyarrow', index=False)

            loss_df = pd.DataFrame({"Batch": batch_list, "Loss": loss_list})
            loss_df.to_csv(f"{cfg.output.dir}/loss.tsv", sep="\t", index=False)

def merge_duplicate_columns_vectorized(matrix, feature_names):
    """
    matrix: Tensor of shape (..., Z)
    feature_names: List or array of strings (already stripped)
    """
    # 1. Convert names to unique integer IDs
    # np.unique with return_inverse gives us an ID for every column
    unique_names, inverse_indices = np.unique(feature_names, return_inverse=True)
    num_unique = len(unique_names)
    
    # Move inverse_indices to the same device as the matrix
    inverse_indices = torch.from_numpy(inverse_indices).to(matrix.device)
    
    # 2. Reshape matrix to 2D for easier scattering: (Batch, Z)
    orig_shape = matrix.shape
    Z = orig_shape[-1]
    matrix_2d = matrix.reshape(-1, Z)
    batch_size = matrix_2d.shape[0]
    
    # 3. Initialize the output tensor: (Batch, Num_Unique)
    output = torch.zeros((batch_size, num_unique), 
                         dtype=matrix.dtype, 
                         device=matrix.device)
    
    # 4. Perform the vectorized sum using scatter_add
    # We expand inverse_indices to match the batch size
    expanded_indices = inverse_indices.expand(batch_size, Z)
    output.scatter_add_(1, expanded_indices, matrix_2d)
    
    # 5. Restore original leading dimensions: (x, y, num_unique)
    final_shape = list(orig_shape[:-1]) + [num_unique]
    return output.reshape(final_shape)

def get_quantile_binary_matrix(subset_matrix, thresholds, target_quantile):
    lower_bounds = thresholds[target_quantile]
    upper_bounds = thresholds[target_quantile + 1]
    
    if target_quantile == thresholds.shape[0] - 2:  # Last bin
        binary_mask = (subset_matrix >= lower_bounds) & (subset_matrix <= upper_bounds)
    else:
        binary_mask = (subset_matrix >= lower_bounds) & (subset_matrix < upper_bounds)
        
    return binary_mask.to(torch.float32)

def _run_dldsc_inference(cfg, model_list, dataloader_list, bin_annots, num_quantile, traits, features, device, log):
    # Get unique features after stripping lowfreq and common suffix if applicable.
    features = np.array([f.replace("_lowfreq", "").replace("_common", "").replace("_frequent", "") for f in features])

    log.info("Starting heritability inference.")
    num_bin = len(np.unique(features[bin_annots]))
    num_cont = len(np.unique(features[~bin_annots]))
    log.info(f"{num_bin} unique binary annotations detected.")
    log.info(f"{num_cont} unique continuous annotations detected.")

    if num_cont > 0:
        log.info("Precomputing quantiles for continous annotations.")
        log.info(f"Using {num_quantile} quantiles.")
        digests = [TDigest() for _ in range(num_cont)]
        for dataloader in dataloader_list:
            for x in dataloader:
                x = merge_duplicate_columns_vectorized(x[:,~bin_annots], features[~bin_annots])
                for i in range(num_cont):
                    digests[i].update(x[:,i])
        percentiles = np.linspace(0, 1, num_quantile + 1)
        thresholds = np.zeros((num_quantile + 1, num_cont))
        for i in range(num_cont):
            thresholds[:,i] = digests[i].quantile(percentiles)
        thresholds = torch.tensor(thresholds, device=device)

    log.info("Computing heritability and enrichment.")
    h2c_bin = torch.zeros(num_bin, len(traits)).to(device)
    asum_bin = torch.zeros(num_bin, 1).to(device)
    h2c_cont = torch.zeros((num_quantile, num_cont, len(traits))).to(device)
    asum_cont = torch.zeros((num_quantile, num_cont, 1)).to(device)
    h2g = torch.zeros(1, len(traits)).to(device)
    m = 0  
    for model, dataloader in zip(model_list, dataloader_list):
        model.eval()
        with torch.no_grad():
            for x in dataloader: 
                x = x.to(device, non_blocking=cfg.training.pin_memory)

                yhat = model(x) 
                h2g += torch.sum(yhat, axis=0).unsqueeze(0)
                m += yhat.shape[0]     

                if num_bin > 0:
                    x_bin = merge_duplicate_columns_vectorized(x[:,bin_annots], features[bin_annots])
                    terms_bin = compute_enrichment_metric(yhat, x_bin)
                    h2c_bin += terms_bin[0]
                    asum_bin += terms_bin[1]
                if num_cont > 0:
                    x_cont = merge_duplicate_columns_vectorized(x[:,~bin_annots], features[~bin_annots])
                    for q in range(num_quantile):
                        terms_cont  = compute_enrichment_metric(yhat, get_quantile_binary_matrix(x_cont, thresholds, q))
                        h2c_cont[q,:,:] += terms_cont[0]
                        asum_cont[q,:,:] += terms_cont[1]

    h2_final = h2g / cfg.data.N 
    h2_final = pd.DataFrame({"h2": h2_final.squeeze().cpu().detach().numpy()})
    h2_final.index = traits
    h2_final.to_csv(f"{cfg.output.dir}/h2.tsv", header=None, sep="\t")
    if num_bin > 0:
        enrichment_bin = (h2c_bin / h2g) / (asum_bin / m)
        enrichment_bin = pd.DataFrame(enrichment_bin.cpu().detach().numpy())
        enrichment_bin.columns = traits
        enrichment_bin.index = np.unique(features[bin_annots])
        enrichment_bin.to_csv(f"{cfg.output.dir}/binary_enrichment.tsv", sep="\t")
    if num_cont > 0:
        enrichment_cont = (h2c_cont / h2g) / (asum_cont / m)
        enrichment_cont = enrichment_cont.reshape(-1, len(traits))
        enrichment_cont = pd.DataFrame(enrichment_cont.cpu().detach().numpy())
        enrichment_cont.columns = traits 
        names = np.unique(features[~bin_annots])
        repeated_names = np.tile(names, num_quantile) 
        suffixes = np.repeat(np.char.add("_q", np.arange(num_quantile).astype(str)), num_cont)
        enrichment_cont.index = np.char.add(repeated_names, suffixes)
        enrichment_cont.to_csv(f"{cfg.output.dir}/continuous_enrichment.tsv", sep="\t")

def predict(cfg):
    log = logging.getLogger(__name__)

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    traits = pd.read_csv(cfg.model.traits, header=None, dtype="str")[0].values
    features = pd.read_csv(cfg.model.features, header=None, sep="\t")[0].values
    bin_annots = pd.read_csv(cfg.model.features, header=None, sep="\t")[1].values.astype(bool)

    if cfg.model.loss != "LDSC":
        if cfg.data.sum_stat.endswith(".parquet"): 
            gwas_data = GWAS_Dataset(cfg.data.sum_stat, traits=traits)
        else:
            gwas_data = GWAS_Dataset_Chr(pd.read_csv(cfg.data.sum_stat, sep="\t"), traits=traits)

        R2 = zarr.open(cfg.data.LD, mode="r")

    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features)

    input_size = len(features)
    output_size = len(traits) 

    with fsspec.open(cfg.data.index, "rb") as f:
        index = pickle.load(f)

    batch_id = pd.read_csv(cfg.data.batch_id, sep="\t")
    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]

    if cfg.model.model == "Non_Functional":
        if cfg.model.loss == "LDSC":
            ValueError(f"Non-Functional only valid for fine-mapping.")
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
    if cfg.model.loss == "LDSC":
        dataloader_list = [Annot_DataLoader(annot_data, batch, index) for batch in batch_list]
    elif (cfg.model.loss == "Finemap") | (cfg.model.loss == "Susie"):
        dataloader_list =  [DLDSC_DataLoader(gwas_data, annot_data, R2, batch, index, 
                                            weights = None, 
                                            shuffle=cfg.training.shuffle, 
                                            num_workers=cfg.training.num_workers, 
                                            disk_cache=cfg.data.disk_cache, 
                                            pin_memory=cfg.training.pin_memory, meta=True) 
                                            for batch in batch_list]
    else:
        raise ValueError(f"{cfg.model.loss} is an invalid loss.")

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
        _run_dldsc_inference(cfg, model_list, dataloader_list, bin_annots, cfg.inference.n_quantile, traits, features, device, log)
    elif cfg.model.loss == "Susie" or cfg.model.loss == "Finemap":
        _run_finemapping_inference(cfg, model_list, dataloader_list, criterion, device, log) 
    else:
        raise ValueError(f"{cfg.model.loss} is an invalid loss.")
    
    