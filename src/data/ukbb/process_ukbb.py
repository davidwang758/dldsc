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

def sparse_R_to_zarr_R(path, out_prefix):
    all_files = [filename[:-4] for filename in os.listdir(path) if filename.endswith(".npz")]
    for f in all_files:
        print(f)
        ld_mat = sp.load_npz(os.path.join(path, f"{f}.npz")).toarray()
        ld_mat += ld_mat.T
        zarr.create_array(store=f"{out_prefix}.zarr", name=f, data=ld_mat, overwrite=True)

def annot_parquet_to_zarr(path_prefix, out_prefix):
    file_name = os.path.basename(path_prefix)
    META_COLUMNS = ['SNP', 'CHR', 'BP', 'A1', 'A2'] 
    for i in range(1,23):
        print(i)
        annot = pd.read_parquet(f"{path_prefix}.{i}.annot.parquet")
        annot.drop(META_COLUMNS, axis=1, inplace=True)
        zarr.create_array(store=f"{out_prefix}.zarr", name=f"{i}", data=annot.values.astype(np.float32), overwrite=True)

def annot_parquet_to_float32(path_prefix, out_prefix):
    chroms = np.arange(22) + 1
    meta = pd.DataFrame({"chr": chroms, "parquet": [f"{out_prefix}.{c}.annot.parquet" for c in chroms]})
    meta.to_csv(f"{out_prefix}.meta.tsv",index=None , sep="\t")
    for i in range(1,23):
        print(i)
        baseline_df = pd.read_parquet(f"{path_prefix}.{i}.annot.parquet")
        casting_dict = {col: 'float32' for col in baseline_df.columns[5:]}
        baseline_df = baseline_df.astype(casting_dict)
        baseline_features = pd.DataFrame({"id": baseline_df.columns[5:]})
        baseline_features.to_csv(f"{out_prefix}.baseline.txt",index=None, header=None)
        baseline_df.to_parquet(f"{out_prefix}.{i}.annot.parquet", index=False)

def gwas_gz_to_parquet(path, out_prefix):
    gwas_ss = pd.read_csv(path, sep="\t")
    gwas_ss = gwas_ss[["SNP", "CHR", "BP", "ALLELE1", "ALLELE0", "A1FREQ", "INFO", "CHISQ_BOLT_LMM"]]
    gwas_ss["A1FREQ"] = np.minimum(gwas_ss["A1FREQ"], 1-gwas_ss["A1FREQ"])
    gwas_ss.columns = ["SNP", "CHR", "BP", "A1", "A2", "MAF", "INFO", "CHISQ"]
    gwas_ss.to_parquet(f"{out_prefix}.parquet")

def multi_gwas_gz_to_parquet(path, out_prefix, files):
    gwas_files = pd.read_csv(files,header=None)[0].values
    prefix = "bolt_337K_unrelStringentBrit_MAF0.001_v3."
    suffix = ".bgen.stats.gz"
    names = [x.removeprefix(prefix).removesuffix(suffix) for x in gwas_files]
    cols = ["SNP", "CHR", "BP", "ALLELE1", "ALLELE0", "A1FREQ", "INFO", "CHISQ_BOLT_LMM"]
    maf = []
    chisq = []
    gwas_ss = None
    for i,f in enumerate(gwas_files):
        print(f)
        if i == 0:
            gwas_ss = pd.read_csv(f"{path}/{f}", sep="\t", usecols=cols)
            maf.append(np.minimum(gwas_ss["A1FREQ"].values, 1-gwas_ss["A1FREQ"].values))
            chisq.append(gwas_ss["CHISQ_BOLT_LMM"].values)
            gwas_ss.columns = ["SNP", "CHR", "BP", "A1", "A2", "MAF", "INFO", "CHISQ"]
            gwas_ss.drop("CHISQ",axis=1,inplace=True)
        else:
            x2 = pd.read_csv(f"{path}/{f}", sep="\t", usecols=["A1FREQ", "CHISQ_BOLT_LMM"])
            maf.append(np.minimum(x2["A1FREQ"].values, 1-x2["A1FREQ"].values))
            chisq.append(x2["CHISQ_BOLT_LMM"].values)
    gwas_ss["MAF"] = np.mean(maf, axis=0)
    print(gwas_ss)
    chisq = pd.DataFrame(np.column_stack(chisq))
    chisq.columns = names
    print(chisq)
    out = pd.concat([gwas_ss, chisq],axis=1)
    out.to_parquet(f"{out_prefix}.parquet")

def multi_gwas_gz_to_parquet_z_scores(path, out_prefix, files):
    gwas_files = pd.read_csv(files,header=None)[0].values
    prefix = "bolt_337K_unrelStringentBrit_MAF0.001_v3."
    suffix = ".bgen.stats.gz"
    names = [x.removeprefix(prefix).removesuffix(suffix) for x in gwas_files]
    cols = ["SNP", "CHR", "BP", "ALLELE1", "ALLELE0", "A1FREQ", "INFO", "BETA", "CHISQ_BOLT_LMM"]
    maf = []
    z = []
    gwas_ss = None
    for i,f in enumerate(gwas_files):
        print(f)
        if i == 0:
            gwas_ss = pd.read_csv(f"{path}/{f}", sep="\t", usecols=cols)
            maf.append(np.minimum(gwas_ss["A1FREQ"].values, 1-gwas_ss["A1FREQ"].values))
            z_scores = gwas_ss["CHISQ_BOLT_LMM"].values
            z_scores[z_scores < 0] = np.nan
            z.append(np.sqrt(z_scores) * np.sign(gwas_ss["BETA"].values))
            gwas_ss.columns = ["SNP", "CHR", "BP", "A1", "A2", "MAF", "INFO", "BETA", "CHISQ"]
            gwas_ss.drop(["BETA","CHISQ"],axis=1,inplace=True)
        else:
            x2 = pd.read_csv(f"{path}/{f}", sep="\t", usecols=["A1FREQ", "BETA", "CHISQ_BOLT_LMM"])
            maf.append(np.minimum(x2["A1FREQ"].values, 1-x2["A1FREQ"].values))
            z_scores = x2["CHISQ_BOLT_LMM"].values
            z_scores[z_scores < 0] = np.nan
            z.append(np.sqrt(z_scores) * np.sign(x2["BETA"].values))
    gwas_ss["MAF"] = np.mean(maf, axis=0)
    print(gwas_ss)
    z = pd.DataFrame(np.column_stack(z))
    z.columns = names
    print(z)
    out = pd.concat([gwas_ss, z],axis=1)
    out.to_parquet(f"{out_prefix}.parquet")

def gwas_parquet_to_zarr(path, out_prefix):
    gwas_ss = pd.read_parquet(path)
    chisq = gwas_ss.CHISQ.values.reshape(-1,1)
    zarr.create_array(store=f"{out_prefix}.zarr", name=f"chisq", data=chisq.astype(np.float32), overwrite=True)

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

    #annot_parquet_to_float32("/home/divyanshi/projects/finemapping/baseline_annots/e_baselineLF2.2.UKB/baselineLF2.2.UKB", 
    #                        "/scratch4/davidwang/datasets/ukbb/annotation/baseline/scores")

    #multi_gwas_gz_to_parquet("/home/divyanshi/projects/finemapping/data/broad_alkesgroup/337k_british_unrelated", 
    #                    "/scratch4/davidwang/datasets/ukbb/gwas/all_traits", 
    #                    "/scratch4/davidwang/datasets/ukbb/gwas/boltlmm_files.txt")
    
    #multi_gwas_gz_to_parquet_z_scores("/home/divyanshi/projects/finemapping/data/broad_alkesgroup/337k_british_unrelated", 
    #                    "/scratch4/davidwang/datasets/ukbb/gwas/all_traits_z_scores", 
    #                    "/scratch4/davidwang/datasets/ukbb/gwas/boltlmm_files.txt")

    sparse_R_to_zarr_R("/home/divyanshi/projects/finemapping/data/broad_alkesgroup/ld_scores", 
                        "/scratch4/davidwang/datasets/ukbb/ld_matrix/ukbb_zscores")

