import numpy as np; np.set_printoptions(precision=4, linewidth=200)
import pandas as pd; pd.set_option('display.width', 200)
import os
import logging
import scipy.stats as stats
from tqdm import tqdm
DEFAULT_REGIONS_FILE = "/home/davidwang/software/polyfun_clean/polyfun/ukb_regions.tsv.gz"
COLS = ["SNP","CHR","BP","A1","A2","MAF","N","Z","prob","alpha","post_mean","tausq","sigmasq","cs"]
NEWCOLS = ["SNP","CHR","BP","A1","A2","MAF","N","Z","PIP","ALPHA_MEAN","BETA_MEAN","ALPHA_SD","SIGMASQ","CREDIBLE_SET"]

def set_snpid_index(df, copy=False, allow_duplicates=False, allow_swapped_indel_alleles=False):
    if copy:
        df = df.copy()
    is_indel = (df['A1'].str.len()>1) | (df['A2'].str.len()>1)
    alleles_are_alphabetical = df['A1'] < df['A2']
    if allow_swapped_indel_alleles:
        df['A1_first'] = alleles_are_alphabetical
    else:
        df['A1_first'] = alleles_are_alphabetical | is_indel
    df['A1s'] = df['A2'].copy()
    df.loc[df['A1_first'], 'A1s'] = df.loc[df['A1_first'], 'A1'].copy()
    df['A2s'] = df['A1'].copy()
    df.loc[df['A1_first'], 'A2s'] = df.loc[df['A1_first'], 'A2'].copy()
    s_chr = df['CHR'].map(lambda c: int(c) if str(c)[0] in ['0','1','2','3','4','5,','6','7','8','9'] else c).astype(str)
    s_bp = df['BP'].astype(int).astype(str)
    df.index = s_chr + '.' + s_bp + '.' + df['A1s'] + '.' + df['A2s']
    df.index.name = 'snpid'
    df.drop(columns=['A1_first', 'A1s', 'A2s'], inplace=True)
    
    #check for duplicate SNPs
    if not allow_duplicates:
        is_duplicate_snp = df.index.duplicated()
        if np.any(is_duplicate_snp):
            df_dup_snps = df.loc[is_duplicate_snp]
            snp_colums = [c for c in ['SNP', 'CHR', 'BP', 'A1', 'A2'] if c in df.columns]
            df_dup_snps = df_dup_snps.loc[~df_dup_snps.index.duplicated(), snp_colums]
            error_msg = 'Duplicate SNPs were found in the input data:\n%s'%(df_dup_snps)
            raise ValueError(error_msg)
    return df

def main(args):
    
    #read sumstats file
    df_sumstats = pd.read_table(args.sumstats)

    #compute p-values if needed
    if args.pvalue_cutoff is not None:
        df_sumstats['P'] = stats.chi2(1).sf(df_sumstats['Z']**2)
        
    #read regions file
    df_regions = pd.read_table(args.regions_file)
    if args.chr is not None:
        df_regions = df_regions.query('CHR==%d'%(args.chr))
        if df_regions.shape[0]==0: raise ValueError('no SNPs found in chromosome %d'%(args.chr))
    df_regions_keep = df_regions.apply(lambda r: np.sum((df_sumstats['CHR']==r['CHR']) & (df_sumstats['BP'].between(r['START'], r['END']))) > 1, axis=1)
    df_regions = df_regions.loc[df_regions_keep]

    #aggregate outputs
    df_sumstats_list = []
    for _, r in tqdm(df_regions.iterrows()):
        chr_num, start, end, url_prefix = r['CHR'], r['START'], r['END'], r['URL_PREFIX']
        
        #apply p-value filter if needed
        if args.pvalue_cutoff is not None:
            df_sumstats_r = df_sumstats.query('CHR==%d & %d <= BP <= %d'%(chr_num, start, end))
            if np.all(df_sumstats_r['P'] > args.pvalue_cutoff): continue        
        
        output_file_r = '%s.%s.%s_%s.susieinf.bgz'%(args.out_prefix, chr_num, start, end)
        #repro_MENARCHE_AGE.19.59000001_62000001.susieinf.bgz
        if not os.path.exists(output_file_r):
            err_msg = 'output file for chromosome %d bp %d-%d doesn\'t exist'%(chr_num, start, end)
            if args.allow_missing_jobs:
                logging.warning(err_msg)
                continue
            else:
                raise IOError(err_msg + '.\nTo override this error, please provide the flag --allow-missing-jobs')
        df_sumstats_r = pd.read_table(output_file_r, sep="\t", usecols=COLS, compression="gzip")
        df_sumstats_r.columns = NEWCOLS
        
        #add the current region to the credible set
        df_sumstats_r['CREDIBLE_SET'] = 'chr%s:%s-%s:'%(chr_num, start, end) + df_sumstats_r['CREDIBLE_SET'].astype(str)
        
        #mark distance from center
        middle = (start+end)//2
        df_sumstats_r['DISTANCE_FROM_CENTER'] = np.abs(df_sumstats_r['BP'] - middle)
        df_sumstats_list.append(df_sumstats_r)
    if len(df_sumstats_list)==0:
        raise ValueError('no output files found')
    
    
    #keep only the most central result for each SNP
    df_sumstats = pd.concat(df_sumstats_list, axis=0)
    df_sumstats.sort_values('DISTANCE_FROM_CENTER', inplace=True, ascending=True)
    df_sumstats = set_snpid_index(df_sumstats, allow_duplicates=True)
    df_sumstats = df_sumstats.loc[~df_sumstats.index.duplicated(keep='first')]
    del df_sumstats['DISTANCE_FROM_CENTER']
    df_sumstats.sort_values(['CHR', 'BP'], inplace=True, ascending=True)
    
    #write output file
    if args.adjust_beta_freq:
        df_sumstats['BETA_MEAN'] /= np.sqrt(2*df_sumstats['MAF']*(1-df_sumstats['MAF']))
        #df_sumstats['BETA_SD']   /= np.sqrt(2*df_sumstats['MAF']*(1-df_sumstats['MAF']))
    df_sumstats.to_csv(args.out, sep='\t', index=False)
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    #general parameters
    parser.add_argument('--sumstats', required=True, help='Name of sumstats file')
    parser.add_argument('--out-prefix', required=True, help='prefix of output files')
    parser.add_argument('--out', required=True, help='name of the aggregated output files')
    parser.add_argument('--allow-missing-jobs', default=False, action='store_true', help='whether to allow missing jobs')
    parser.add_argument('--regions-file', default=DEFAULT_REGIONS_FILE, help='name of file of regions and their URLs')
    parser.add_argument('--chr', default=None, type=int, help='Target chromosome (if not provided, all chromosomes will be considered)')
    parser.add_argument('--pvalue-cutoff', type=float, default=None, help='only consider regions that have at least one SNP with a p-value greater than this cutoff')
    parser.add_argument('--adjust-beta-freq', default=False, action='store_true', help='If specified, the posterior estimates of the SNP effect sizes will be on per-allele scale rather than a per-standardized genotype scale')
    
    #extract args
    args = parser.parse_args()
    
    #check that the output directory exists
    if len(os.path.dirname(args.out))>0 and not os.path.exists(os.path.dirname(args.out)):
        raise ValueError('output directory %s doesn\'t exist'%(os.path.dirname(args.out)))    

    #invoke main function
    main(args)
    