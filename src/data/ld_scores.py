# Load data through dataloader.
# Compute LD score.
# Compute weight. W is sum of columns. L score without annotations is other score.
import pandas as pd
from data.dataloader import GWAS_Dataset, Annotation_Dataset, DLDSC_DataLoader
import zarr 
import pickle 
import torch
import sys


def compute_ld_scores(dataloader):
    path = "/scratch4/davidwang/datasets/ukbb/ld_scores/ALL_MAF:0.01_X2:80_INFO:0.6_SQ_BASELINE"
    weight = []
    ld_score = []
    m = []
    meta = []
    annots = []
    cur_chrom = None

    for x, y, R2, w, mt in dataloader:
        batch_chrom = mt.CHR.values[0]
        # todo
        # Add filtering for rect matrix.
        if cur_chrom != batch_chrom:
            if cur_chrom is not None:
                meta = pd.concat(meta,axis=0).reset_index(drop=True)

                annots = pd.DataFrame(torch.concat(annots, axis=0).numpy())
                annots.columns = dataloader.features

                ld_score = pd.DataFrame(torch.concat(ld_score, axis=0).numpy())
                ld_score.columns = dataloader.features

                weight = pd.DataFrame(torch.concat(weight, axis=0).numpy())
                weight.columns = ["L2"]

                m = pd.DataFrame(torch.sum(torch.stack(m,dim=0),axis=0).numpy().reshape(1, -1))
                
                pd.concat([meta, ld_score],axis=1).to_parquet(f"{path}/annotations.{cur_chrom}.l2.ldscore.parquet")
                pd.concat([meta, annots],axis=1).to_parquet(f"{path}/annotations.{cur_chrom}.annot.parquet")
                pd.concat([meta, weight],axis=1).to_parquet(f"{path}/weights.{cur_chrom}.l2.ldscore.parquet")
                m.to_csv(f"{path}/annotations.{cur_chrom}.l2.M",sep=" ", index=None, header=None)

                ld_score = []
                weight = []
                m = []
                meta = []
                annots = []

            cur_chrom = batch_chrom
        weight.append(torch.sum(R2, axis=1))
        ld_score.append(R2 @ x)
        annots.append(x)
        m.append(torch.sum(x, axis=0))
        meta.append(mt)

    meta = pd.concat(meta,axis=0).reset_index(drop=True)

    annots = pd.DataFrame(torch.concat(annots, axis=0).numpy())
    annots.columns = dataloader.features

    ld_score = pd.DataFrame(torch.concat(ld_score, axis=0).numpy())
    ld_score.columns = dataloader.features

    weight = pd.DataFrame(torch.concat(weight, axis=0).numpy())
    weight.columns = ["L2"]

    m = pd.DataFrame(torch.sum(torch.stack(m,dim=0),axis=0).numpy().reshape(1, -1))
    
    pd.concat([meta, ld_score],axis=1).to_parquet(f"{path}/annotations.{cur_chrom}.l2.ldscore.parquet")
    pd.concat([meta, annots],axis=1).to_parquet(f"{path}/annotations.{cur_chrom}.annot.parquet")
    pd.concat([meta, weight],axis=1).to_parquet(f"{path}/weights.{cur_chrom}.l2.ldscore.parquet")
    m.to_csv(f"{path}/annotations.{cur_chrom}.l2.M",sep=" ", index=None, header=None)

def get_ld_scores(cfg):
    traits = pd.read_csv(cfg.model.traits, header=None)[0].values
    features = pd.read_csv(cfg.model.features, header=None)[0].values

    gwas_data = GWAS_Dataset(cfg.data.chisq, traits=traits)
    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features) 
    
    R2 = zarr.open(cfg.data.R2, mode="r")
    with open(cfg.data.index, "rb") as f:
        index = pickle.load(f)

    batch_id = pd.read_csv(cfg.data.batch_id, sep="\t")
    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]
    #batch_id = batch_id.loc[batch_id.chr == 22,:] # fix this later
    #print(batch_id.loc[batch_id.chr == 22,:])
    #sys.exit(0)

    # Initialize dataloaders
    dataloader =  DLDSC_DataLoader(gwas_data, annot_data, R2, batch_id, index, 
                                         weights = None, 
                                         shuffle=False, 
                                         num_workers=cfg.training.num_workers, 
                                         disk_cache=cfg.data.disk_cache, 
                                         pin_memory=cfg.training.pin_memory, meta=True)

    compute_ld_scores(dataloader)
