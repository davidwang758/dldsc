import os
import zarr
import scipy.sparse as sp
import numpy as np
import pandas as pd

def sparse_R_to_zarr_R2(path, out_prefix):
    all_files = [filename[:-4] for filename in os.listdir(path) if filename.endswith(".npz")]
    for f in all_files:
        print(f)
        ld_mat = sp.load_npz(os.path.join(path, f"{f}.npz")).toarray()
        ld_mat += ld_mat.T
        np.power(ld_mat, 2, out=ld_mat)
        zarr.create_array(store=f"{out_prefix}.zarr", name=f, data=ld_mat, overwrite=True)

def annot_parquet_to_zarr(path_prefix, out_prefix):
    file_name = os.path.basename(path_prefix)
    META_COLUMNS = ['SNP', 'CHR', 'BP', 'A1', 'A2'] 
    for i in range(1,23):
        print(i)
        annot = pd.read_parquet(f"{path_prefix}.{i}.annot.parquet")
        annot.drop(META_COLUMNS, axis=1, inplace=True)
        zarr.create_array(store=f"{out_prefix}.zarr", name=f"{i}", data=annot.values.astype(np.float32), overwrite=True)

def gwas_gz_to_parquet(path, out_prefix):
    gwas_ss = pd.read_csv(path, sep="\t")
    gwas_ss = gwas_ss[["SNP", "CHR", "BP", "ALLELE1", "ALLELE0", "A1FREQ", "INFO", "CHISQ_BOLT_LMM"]]
    gwas_ss["A1FREQ"] = np.minimum(gwas_ss["A1FREQ"], 1-gwas_ss["A1FREQ"])
    gwas_ss.columns = ["SNP", "CHR", "BP", "A1", "A2", "MAF", "INFO", "CHISQ"]
    gwas_ss.to_parquet(f"{out_prefix}.parquet")

def gwas_parquet_to_zarr(path, out_prefix):
    gwas_ss = pd.read_parquet(path)
    chisq = gwas_ss.CHISQ.values.reshape(-1,1)
    zarr.create_array(store=f"{out_prefix}.zarr", name=f"chisq", data=chisq.astype(np.float32), overwrite=True)

def make_R2_table(path, out_prefix):
    all_files = [filename[:-4] for filename in os.listdir(path) if filename.endswith(".npz")]
    splits = [f.split("_") for f in all_files]
    files = [os.path.join(path,f"{f}.gz") for f in all_files]
    chrom = [x[0].removeprefix("chr") for x in splits]
    start = [x[1] for x in splits]
    end = [x[2] for x in splits]
    df = pd.DataFrame({"chr": chrom, "id": all_files, "file":files, "start": start, "end": end})
    df.to_csv(f"{out_prefix}.tsv", sep="\t", index=None)

if __name__ == '__main__':
    #gwas_gz_to_parquet("/home/divyanshi/projects/finemapping/data/broad_alkesgroup/337k_british_unrelated/bolt_337K_unrelStringentBrit_MAF0.001_v3.body_BMIz.bgen.stats.gz",
    #                   "/scratch4/davidwang/datasets/ukbb/gwas/BMI")

    #sparse_R_to_zarr_R2("/home/divyanshi/projects/finemapping/data/broad_alkesgroup/ld_scores", 
    #                    "/scratch4/davidwang/datasets/ukbb/ld_matrix/ukbb")

    #annot_parquet_to_zarr("/home/divyanshi/projects/finemapping/baseline_annots/e_baselineLF2.2.UKB/baselineLF2.2.UKB",
    #                      "/scratch4/davidwang/datasets/ukbb/annotation/ukbb_baseline")

    #gwas_parquet_to_zarr("/scratch4/davidwang/datasets/ukbb/gwas/BMI.parquet",
    #                     "/scratch4/davidwang/datasets/ukbb/gwas/BMI")

    #make_R2_table("/home/divyanshi/projects/finemapping/data/broad_alkesgroup/ld_scores",
    #              "/home/davidwang/datasets/UKBB/gwas_ss/polyfun_torch_test/R2_table")

