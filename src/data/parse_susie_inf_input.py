import numpy as np
import pandas as pd
import argparse
import os 
import scipy.sparse as sparse
import torch 

def load_ld_npz(ld_prefix):
        
    #load SNPs info
    snps_filename_parquet = ld_prefix+'.parquet'
    snps_filename_gz = ld_prefix+'.gz'
    if os.path.exists(snps_filename_parquet):
        df_ld_snps = pd.read_parquet(snps_filename_parquet)
    elif os.path.exists(snps_filename_gz):
        df_ld_snps = pd.read_table(snps_filename_gz, sep='\s+')
        df_ld_snps.rename(columns={'allele1':'A1', 'allele2':'A2', 'position':'BP', 'chromosome':'CHR', 'rsid':'SNP'}, inplace=True, errors='ignore')
    else:
        raise ValueError('couldn\'t find SNPs file %s or %s'%(snps_filename_parquet, snps_filename_gz))
        
    #load LD matrix
    R_filename = ld_prefix+'.npz'
    if not os.path.exists(R_filename):
        raise IOError('%s not found'%(R_filename))
    ld_arr = sparse.load_npz(R_filename).toarray()
    ld_arr = ld_arr+ld_arr.T
    assert np.allclose(np.diag(ld_arr), 1.0)
    assert np.all(~np.isnan(ld_arr))
    
    #sanity checks
    assert ld_arr.shape[0] == ld_arr.shape[1]
    if ld_arr.shape[0] != df_ld_snps.shape[0]:
        raise ValueError('LD matrix has a different number of SNPs than the SNPs file')
    
    return ld_arr, df_ld_snps

def read_sum_stats_files(ss_meta, trait, chr):
    ss_list = []
    for i,d in enumerate(ss_meta[0].values):
        sumstats_file = f"{d}/{trait}.{chr}.{ss_meta[1].values[i]}.gz"
        ss_list.append(pd.read_csv(sumstats_file, sep="\t"))
    assert np.all([x.BP == ss_list[0].BP for x in ss_list])
    out = ss_list[0][["SNP","CHR","BP","A1","A2","MAF","N","Z"]]
    priors = pd.concat([x["SNPVAR"] for x in ss_list], axis=1)
    priors.columns = ss_meta[2].values
    return pd.concat([out, priors],axis=1)

def main(args):
    traits = pd.read_csv(args.trait_file, header=None)[0].values
    ss_meta = pd.read_csv(args.sum_stats_meta, sep="\t", header=None)

    # Read LD matrix
    ld_arr, df_ld_snps = load_ld_npz(args.ld_mat)
    df_ld_index = df_ld_snps['CHR'].astype("str") + '.' + df_ld_snps['BP'].astype("str") + '.' + df_ld_snps['A1'] + '.' + df_ld_snps['A2']
    df_ld_snps.index = df_ld_index 
    df_ld = pd.DataFrame(ld_arr, index=df_ld_index, columns=df_ld_index)
     
    # Read Sum Stats
    snps_in_id_file = None
    for i,t in enumerate(traits): 
        df_sumstats = read_sum_stats_files(ss_meta, t, args.chr)
        df_sumstats = df_sumstats.query('CHR==%s'%(args.chr)).copy()
        df_sumstats = df_sumstats.query('%s <= BP <= %s'%(args.start, args.end))

        if df_sumstats.shape[0] == 0:
            sys.exit(0) # exit if file is empty after filtering region
        if df_sumstats.shape[0] == 1:
            print("size 1 exists.")
        df_sumstats_index = df_sumstats['CHR'].astype("str") + '.' + df_sumstats['BP'].astype("str") + '.' + df_sumstats['A1'] + '.' + df_sumstats['A2']    
        df_sumstats.index = df_sumstats_index

        # only compute this once for the first trait. The assert below checks for errors.
        snps_in_ld_file = df_sumstats.index.isin(df_ld.index)
        
        df_sumstats = df_sumstats.loc[snps_in_ld_file]
        df_ld = df_ld.loc[df_sumstats.index, df_sumstats.index]
        df_ld_snps = df_ld_snps.loc[df_ld.index]

        assert np.all(df_ld.index == df_sumstats.index)
        assert np.all(df_ld_snps.index == df_sumstats.index)

        df_sumstats[["SNP","CHR","BP","A1","A2","MAF","N","Z"]].to_csv(f"{args.out_dir}/{t}.{args.chr}.{args.start}_{args.end}.gz", index=None, sep="\t", compression='gzip')

        for p in ss_meta[2].values:
            if p != "non_funct":
                if df_sumstats[p].sum() > 0:
                    df_sumstats[p] = df_sumstats[p].values / df_sumstats[p].sum()
                else:
                    df_sumstats[p] = 1 # Set prior to uniform if priors sum to 0
                    df_sumstats[p] = df_sumstats[p].values / df_sumstats[p].sum()
                df_priors = df_sumstats[p] # make these sum to 1
                df_priors.name = "SNPVAR"
                df_priors.to_csv(f"{args.out_dir}/{t}.{args.chr}.{args.start}_{args.end}.{p}_priors.gz", index=None, sep="\t", compression='gzip')

        # Use torch to do eigendecomp (use lower by default)
        dsq_file = f"{args.out_dir}/{t}.{args.chr}.{args.start}_{args.end}.Dsq.npz"
        v_file = f"{args.out_dir}/{t}.{args.chr}.{args.start}_{args.end}.V.npz"
        if not os.path.exists(dsq_file) and not os.path.exists(v_file):
            N = df_sumstats.N.values[0]
            eigenvals,V = torch.linalg.eigh(torch.from_numpy(df_ld.values))
            Dsq = N * eigenvals
            np.savez_compressed(dsq_file, Dsq.numpy())
            np.savez_compressed(v_file, V.numpy())

if __name__ == '__main__':
    # Need a flag to not load and save the ld matrix (assuming it's already saved.)
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr", type=str, help="Chromosome.")
    parser.add_argument("--start", type=str, help="Window start coordinate.")
    parser.add_argument("--end", type=str, help="Window end coordinate.")
    parser.add_argument("--sum_stats_meta", type=str, help="Sum stats metadata file.")
    parser.add_argument("--ld_mat", type=str, help="LD matrix directory.")
    parser.add_argument("--trait_file", type=str, help="Trait file.")
    parser.add_argument("--out_dir", type=str, help="Directory to output.")
    args = parser.parse_args()
    
    main(args)