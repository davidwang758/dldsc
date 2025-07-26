import os
import numpy as np
import pandas as pd
import polars as pl
import zarr
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import time
import sys

META_COLS = ["CHR", "BP", "A1", "A2"]

class GWAS_Dataset():
    def __init__(self, gwas_parquet):
        self.data = pl.scan_parquet(gwas_parquet)

    def get_chisq(self):
        return self.data.select("CHISQ").collect().to_numpy()

class Annotation_Dataset():
    def __init__(self, annot_parquet, features=[]):
        self.data = {}
        self.features = features
        for i, c in enumerate(annot_parquet.chr):
            self.data[str(c)] = pl.scan_parquet(annot_parquet.parquet[i])

    def get_annot(self, chr):
        if len(self.features) > 0:
            return self.data[chr].select(self.features).collect().to_numpy()
        else:
            return self.data[chr].select(pl.exclude(["SNP", "CHR", "BP", "A1", "A2"])).collect().to_numpy()
    
class DLDSC_Dataset(Dataset):
    def __init__(self, chisq, annotation, R2, batch_id, index, weights=None, disk_cache=None):
        self.chisq = chisq
        self.annotation = annotation
        self.R2 = R2 
        self.weights = weights

        self.gwas_index = index["gwas"]
        self.annot_index = index["annotation"]
        self.R2_row_index = index["R2_row"]
        self.R2_col_index = index["R2_col"]

        self.batch_id = batch_id
        self.n_batch = len(batch_id)

        self.disk_cache = disk_cache

    def __getitem__(self, idx):        
        cur_batch = self.batch_id[idx]

        # Load chisq and annotations
        y = torch.from_numpy(self.chisq[self.gwas_index[cur_batch],])
        x = torch.from_numpy(self.annotation[self.annot_index[cur_batch],])

        # Load LD matrix
        if self.disk_cache is None:
            R2 = torch.from_numpy(self.R2[cur_batch].oindex[self.R2_row_index[cur_batch],self.R2_col_index[cur_batch]])
        else:
            r2_file = f"{self.disk_cache}/{cur_batch}.npy"
            if os.path.exists(r2_file):
                R2 = torch.from_numpy(np.load(r2_file))
            else:
                R2 = torch.from_numpy(self.R2[cur_batch].oindex[self.R2_row_index[cur_batch],self.R2_col_index[cur_batch]])
                np.save(r2_file, R2)
                
        # Load weights
        if self.weights is not None:
            w = torch.from_numpy(self.weights[cur_batch])
        else:
            w = torch.ones((y.size(0), 1))

        return x, y, R2, w

    def __len__(self):
        return self.n_batch

class DLDSC_DataLoader:
    def __init__(self, gwas, annotation, R2, batch_id, index, weights = None, shuffle=True, num_workers=0, disk_cache=None, pin_memory=False):
        self.gwas = gwas.get_chisq()
        self.annotation = annotation
        self.R2 = R2
        self.weights = weights

        self.index = index
        self.batch_id = batch_id

        self.num_workers = num_workers
        self.disk_cache = disk_cache
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.chroms = batch_id.chr.unique()

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.chroms)

        for c in self.chroms:
            batch_id_c = self.batch_id.id.values[self.batch_id.chr == c]
            dataset = DLDSC_Dataset(self.gwas, self.annotation.get_annot(str(c)), self.R2, batch_id_c, self.index, 
                                    weights = self.weights, disk_cache = self.disk_cache)

            loader = DataLoader(dataset, 
                                batch_size = 1, # DO NOT CHANGE THIS, CANNOT COLLATE BATCHES
                                shuffle = self.shuffle,
                                num_workers = self.num_workers,
                                pin_memory=self.pin_memory,
                                collate_fn = lambda batch: batch[0])

            yield from loader

class DLDSC_DataLoader_Old(IterableDataset):
    def __init__(self, chisq, annotation, R2, batch_id, index, weights=None, shuffle=True):
        # Data
        self.chisq = chisq["chisq"]
        self.annotation = annotation 
        self.R2 = R2 
        self.weights = weights

        # Index
        self.gwas_index = index["gwas"]
        self.annot_index = index["annotation"]
        self.R2_row_index = index["R2_row"]
        self.R2_col_index = index["R2_col"]

        # Batch ID
        # All keys in the 4 index objects are the same.
        self.batch_id = batch_id.loc[[id in self.gwas_index.keys() for id in batch_id.id],:]
        self.batch_id.reset_index(drop=True, inplace=True)
        self.n_batch = len(self.batch_id.id)

        self.shuffle = shuffle
        self.current_batch_index = 0
        self.current_chrom = None
        self.current_annotation = None

    def __iter__(self):
        self.current_batch_index = 0
        if self.shuffle:
            self.batch_id = self.batch_id.sample(frac=1).reset_index(drop=True)
            chroms = self.batch_id.chr.unique()
            np.random.shuffle(chroms)
            self.batch_id["tmp"] = pd.Categorical(self.batch_id.chr, categories=chroms, ordered=True)
            self.batch_id = self.batch_id.sort_values('tmp').reset_index(drop=True)
        return self

    def __next__(self):
        # Stop after iterating through all batches
        if self.current_batch_index >= self.n_batch:
            raise StopIteration
        
        cur_chrom = str(self.batch_id.chr[self.current_batch_index])
        cur_batch = self.batch_id.id[self.current_batch_index]
        # Load next chromsome 
        if self.current_chrom != cur_chrom:
            self.current_chrom = cur_chrom
            self.current_annotation = self.annotation[self.current_chrom][:]

        # Load data
        y = torch.from_numpy(self.chisq[self.gwas_index[cur_batch],].astype(np.float32))
        x = torch.from_numpy(self.current_annotation[self.annot_index[cur_batch],].astype(np.float32))
        R2 = torch.from_numpy(self.R2[cur_batch].oindex[self.R2_row_index[cur_batch],self.R2_col_index[cur_batch]])

        if self.weights is not None:
            w = torch.from_numpy(self.weights[cur_batch]).view(-1,1)
        else:
            w = torch.ones((y.size(0), 1))

        self.current_batch_index += 1

        return x, y, R2, w

    def __len__(self):
        return len(self.n_batch)

def build_index(gwas_path, annotation_paths, R2_meta, min_batch_maf, min_ld_maf, max_chisq, min_info, out_prefix, square=False):
    if os.path.exists(f"{out_prefix}.idx"):
        return

    print("Building index.")
    gwas_ss = pd.read_parquet(gwas_path, columns=META_COLS + ["MAF", "CHISQ", "INFO"])
    gwas_ss["index_gwas"] = np.arange(gwas_ss.shape[0])
    gwas_ss.query('CHISQ <= @max_chisq and CHISQ >= 0', inplace=True)

    annot = [pd.read_parquet(a, columns=META_COLS) for a in annotation_paths]
    merged = {}
    for a in annot:
        assert np.all(a.CHR == a.CHR[0]), f"Not all chromosomes are the same in file {a}."
        a["index_annot"] = np.arange(a.shape[0])
        chrom = a.CHR[0]
        merged[chrom] = a.merge(gwas_ss.query('CHR == @chrom'), on=META_COLS, how="inner")

    merged_batch = {}
    merged_ld = {}
    for c in merged.keys():
        merged_batch[c] = merged[c].query('MAF >= @min_batch_maf and INFO >= @min_info') 
        merged_ld[c] = merged[c].query('MAF >= @min_ld_maf') 

    gwas_index = {}
    annot_index = {}
    R2_index_row = {}
    R2_index_col = {}

    R2 = pd.read_csv(R2_meta, sep="\t")
    for i in range(R2.shape[0]):
        print(i)
        snps = pd.read_csv(R2.file[i], sep="\t")
        snps.columns = ["rsid"] + META_COLS
        snps["index_ld"] = np.arange(snps.shape[0])
        start = R2.start[i]
        end = R2.end[i]
        q1 = "BP >= @start + 1000000 & BP < @end - 1000000"
        q2 = "BP >= @start & BP < @end"
        overlap_batch = snps.query(q1).merge(merged_batch[R2.chr[i]].query(q1), on=META_COLS, how="inner")
        if square:
            overlap_ld = overlap_batch
        else:
            overlap_ld = snps.merge(merged_ld[R2.chr[i]].query(q2), on=META_COLS, how="inner")

        if (overlap_batch.shape[0] == 0) | (overlap_ld.shape[0] == 0):
            print(f"Batch {R2.id[i]} is empty after filtering. Skipping indexing this batch.")
            continue

        gwas_index[R2.id[i]] = overlap_batch.index_gwas.values
        annot_index[R2.id[i]] = overlap_ld.index_annot.values
        R2_index_row[R2.id[i]] = overlap_batch.index_ld.values
        R2_index_col[R2.id[i]] = overlap_ld.index_ld.values

    out = {"gwas": gwas_index, "annotation": annot_index, "R2_row": R2_index_row, "R2_col": R2_index_col}
    with open(f"{out_prefix}.idx", 'wb') as f:
        pickle.dump(out, f)

def validate_index(gwas_path, annotation_paths, R2_meta, index_prefix):
    print("Validating index.")
    with open(f"{index_prefix}.idx", "rb") as f:
        index = pickle.load(f)

    gwas_index = index["gwas"]
    annotation_index = index["annotation"]
    R2_row_index = index["R2_row"]
    R2_col_index = index["R2_col"]

    gwas_ss = pd.read_parquet(gwas_path, columns=META_COLS)
    annot = {}
    for f in annotation_paths:
        a = pd.read_parquet(f, columns=META_COLS)
        assert np.all(a.CHR == a.CHR[0]), f"Not all chromosomes are the same in file {a}."
        annot[a.CHR[0]] = a
    R2 = pd.read_csv(R2_meta, sep="\t")

    val = []
    for i in range(R2.shape[0]):
        k = R2.id[i]
        c = R2.chr[i]
        # Empty keys are not indexed
        if k in gwas_index.keys():
            r2 = pd.read_csv(R2.file[i], sep="\t", usecols=["chromosome", "position", "allele1", "allele2"])
            
            gwas_i = gwas_ss.iloc[gwas_index[k],:].values
            annot_i = annot[c].iloc[annotation_index[k],:]
            r2_row_i = r2.iloc[R2_row_index[k],:].values
            r2_col_i = r2.iloc[R2_col_index[k],:].values
            
            passed = np.all(gwas_i == r2_row_i) & np.all(annot_i == r2_col_i)
            val.append(passed)
            if passed:
                print(f"Batch ID {k} passed validation.")

    if np.all(val):
        print("All indices passed.")
    else:
        print("Not all indices passed.")

if __name__ == '__main__':
    gwas_path = "/scratch4/davidwang/datasets/ukbb/gwas/BMI.parquet"
    #annotation_meta = "/scratch4/davidwang/datasets/ukbb/annotation/all/scores.meta.tsv"
    annotation_paths = [f"/scratch4/davidwang/datasets/ukbb/annotation/all/scores.{i}.annot.parquet" for i in range(1,23)]
    R2_meta = "/scratch4/davidwang/datasets/ukbb/ld_matrix/R2_table.tsv"

    out_prefix = "/scratch4/davidwang/datasets/ukbb/index/BMI_RMAF:0.01_CMAF:0.001_X2:80_INFO:0.6"
    #build_index(gwas_path, annotation_paths, R2_meta, 0.01, 0.001, 80, 0.6, out_prefix, square=False)
    #validate_index(gwas_path, annotation_paths, R2_meta, out_prefix)

    #out_prefix = "/scratch4/davidwang/datasets/ukbb/index/BMI_MAF:0.01_X2:80_INFO:0.6_SQ"
    #build_index(gwas_path, annotation_paths, R2_meta, 0.01, 0.01, 80, 0.6, out_prefix, square=True)
    #validate_index(gwas_path, annotation_paths, R2_meta, out_prefix)

    # Test dataloader 
    gwas_data = GWAS_Dataset(gwas_path)
    annotation_meta = "/tmp/david/all/scores.meta.tsv"
    annot_data = Annotation_Dataset(pd.read_csv(annotation_meta, sep="\t"))
    ld_mat_data =  zarr.open("/scratch4/davidwang/datasets/ukbb/ld_matrix/ukbb.zarr", mode="r")
    batch_id = pd.read_csv(R2_meta, sep="\t")
    with open(f"{out_prefix}.idx", "rb") as f:
        index = pickle.load(f)

    batch_id = batch_id.loc[[id in index["gwas"].keys() for id in batch_id.id],:]
    #batch_id = batch_id.loc[batch_id.chr == 2,:]
    #dc = "/scratch4/davidwang/datasets/ukbb/ld_matrix/BMI_RMAF:0.01_CMAF:0.001_X2:80_INFO:0.6"
    #dc = "/scratch6/davidwang/datasets/BMI_MAF:0.01_X2:80_INFO:0.6_SQ"
    #dc = "/scratch4/davidwang/datasets/ukbb/ld_matrix/BMI_MAF:0.01_X2:80_INFO:0.6_SQ"
    dc = "/tmp/david/BMI_MAF:0.01_X2:80_INFO:0.6_SQ"
    #dc = "/tmp/david"
    dataloader = DLDSC_DataLoader(gwas_data, annot_data, ld_mat_data, batch_id, index, weights = None, shuffle=False, num_workers=16, disk_cache=dc)
    #dataloader = DLDSC_DataLoader(gwas_data, annot_data, ld_mat_data, batch_id, index, weights = None, shuffle=False, num_workers=0, disk_cache=None)
    #dataloader_old = DLDSC_DataLoader_Old(gwas_zarr, annot_zarr, ld_mat_zarr, batch_id, index, shuffle=False)
    for i in range(100):
        start = time.time()
        for x, y, R2, w in dataloader:
            #print(x.shape)
            #print(y.shape)
            print(R2.shape)
            #print(w.shape)
        end = time.time()
        print(end - start)

