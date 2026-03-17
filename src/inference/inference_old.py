import pandas as pd
import torch
from pathlib import Path
from data.dataloader import Annotation_Dataset
from model.mlp import MLP_Baseline, MLP_Borzoi
import polars as pl

def score_dldsc(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    traits = pd.read_csv(cfg.model.traits, header=None)[0].values
    features = pd.read_csv(cfg.model.features, header=None)[0].values
    annot_data = Annotation_Dataset(pd.read_csv(cfg.data.annotation, sep="\t"), features=features)
    input_size = len(features)
    output_size = len(traits)

    model_checkpoint = cfg.model.checkpoint
    if cfg.model.model == "MLP_Baseline":
        model = MLP_Baseline(input_size, output_size).to(device)
    elif cfg.model.model == "MLP_Borzoi":
        model = MLP_Borzoi(input_size, output_size).to(device)
    else:
        raise Exception(f"{cfg.model.model} is an invalid model.")
    print(model_checkpoint)
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    output_prefix = Path(cfg.model.checkpoint).stem
    output_dir = Path(f"{cfg.output.priors}/{output_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for chr in annot_data.chr:
        print(f"Scoring variant on chromosome {chr}.")
        meta = annot_data.get_meta(str(chr))
        annot = torch.from_numpy(annot_data.get_annot(str(chr))).to(device)
        with torch.no_grad(): 
            yhat = model(annot).cpu().numpy()
        result = pl.DataFrame(yhat)
        result.columns = traits
        result = pl.concat([meta, result], how="horizontal")
        result.write_parquet(f"{cfg.output.priors}/{output_prefix}/{str(chr)}.parquet")



