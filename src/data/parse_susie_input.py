import pandas as pd
import argparse 
import numpy as np

def main(args):
    sum_stats_file = args.sum_stats
    prior_file = args.prior_dir
    out_file = args.out
    t = args.trait_name
    exclude = pd.read_csv(args.exclude_file, sep="\t")
    
    gwas = pd.read_parquet(sum_stats_file)
    for c in range(1,23):
        priors = pd.read_parquet(f"{prior_file}/{c}.parquet", columns=["SNP","CHR","BP","A1","A2",t])
        priors.columns = ["SNP","CHR","BP","A1","A2","SNPVAR"]
        priors = gwas.loc[gwas.CHR == c,:].merge(priors, on=["SNP","CHR","BP","A1","A2"])
        remove_regions = exclude.loc[exclude.CHR == c,:]
        remove_index = np.full(priors.shape[0], False, dtype=bool)
        for i in range(remove_regions.shape[0]):
            remove_index = remove_index | ((priors.BP >= remove_regions.START.values[i]) & (priors.BP < remove_regions.END.values[i]))
        priors = priors.loc[~remove_index,:]
        priors = priors[["SNP","CHR","BP","A1","A2","SNPVAR","MAF","N","Z"]] # Reorder columns
        priors.to_csv(f"{out_file}.{c}.snpvar_dldsc.gz", sep="\t", compression='gzip', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sum_stats", type=str, help="Munged summary statistic file prefix.")
    parser.add_argument("--prior_dir", type=str, help="Directory containing prior files.")
    parser.add_argument("--out", type=str, help="Out file prefix.")
    parser.add_argument("--trait_name", type=str, help="Trait name.")
    parser.add_argument("--exclude_file", type=str, help="Regions to exclude (e.g. MHC regions)")
    args = parser.parse_args()
    
    main(args)