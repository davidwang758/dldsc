import h5py
import zarr
import numpy as np
import pandas as pd
import sys

def borzoi_h5_baseline_parquet_to_parquet(borzoi_h5_file, baseline_prefix, out_prefix, meta_file, scaled_h5_file=None):
    meta_data = pd.read_csv(meta_file, sep="\t")
    with h5py.File(borzoi_h5_file, 'r') as f:
        #chr = np.array([x.removeprefix("chr") for x in f["chr"].asstr()[:]]).astype(np.int64)
        #bp = f["pos"][:]
        #a1 = f["ref_allele"].asstr()[:]
        #a2 = f["alt_allele"].asstr()[:]
        chr = meta_data.CHR.values
        bp = meta_data.BP.values
        a1 = meta_data.A1.values
        a2 = meta_data.A2.values
        ids = f["target_ids"].asstr()[:]
        labels = f["target_labels"].asstr()[:]
        chroms = np.unique(chr)

        meta = pd.DataFrame({"chr": chroms, "parquet": [f"{out_prefix}.{c}.annot.parquet" for c in chroms]})
        meta.to_csv(f"{out_prefix}.meta.tsv",index=None, sep="\t")
        map_df = pd.DataFrame({"id":ids, "label":labels})
        map_df.to_csv(f"{out_prefix}.id_label_map.tsv", index=None, sep="\t")
        borzoi_features = pd.DataFrame({"id": ids})
        borzoi_features.to_csv(f"{out_prefix}.borzoi.txt",index=None, header=None)
        for c in chroms:
            print(c)
            keep = np.where(chr == c)[0]
            if scaled_h5_file is not None:
                with h5py.File(scaled_h5_file, 'r') as sf:
                    out = pd.DataFrame(sf["logD2"][keep,:].astype(np.float32))
            else:
                out = pd.DataFrame(f["logD2"][keep,:].astype(np.float32))
            out.columns = ids
            out["CHR"] = chr[keep]
            out["BP"] = bp[keep]
            out["A1"] = a1[keep]
            out["A2"] = a2[keep]

            # Read baseline
            baseline_df = pd.read_parquet(f"{baseline_prefix}.{c}.annot.parquet")
            casting_dict = {col: 'float32' for col in baseline_df.columns[5:]}
            baseline_df = baseline_df.astype(casting_dict)
            baseline_features = pd.DataFrame({"id": baseline_df.columns[5:]})
            baseline_features.to_csv(f"{out_prefix}.baseline.txt",index=None, header=None)

            pd.concat([baseline_features, borzoi_features]).to_csv(f"{out_prefix}.all.txt",index=None, header=None)

            # Merge
            out = baseline_df.merge(out, on=["CHR", "BP", "A1", "A2"], how="inner")
            out.sort_values(by='BP', inplace=True)
            assert not out.isnull().values.any(), f"DF for chr{c} contains NANs."
            
            # Write
            out.to_parquet(f"{out_prefix}.{c}.annot.parquet", index=False)

            del out

def borzoi_pca_make_meta(path, path_prefix, out_prefix):
    chroms = np.arange(22) + 1
    meta = pd.DataFrame({"chr": chroms, "parquet": [f"{out_prefix}.{c}.annot.parquet" for c in chroms]})
    meta.to_csv(f"{out_prefix}.meta.tsv",index=None, sep="\t")

    for c in chroms:
        baseline_df = pd.read_parquet(f"{path_prefix}.{c}.annot.parquet")
        casting_dict = {col: 'float32' for col in baseline_df.columns[5:]}
        baseline_df = baseline_df.astype(casting_dict)
        pca_df = pd.read_parquet(f"{path}.{c}.annot.parquet")
        casting_dict = {col: 'float32' for col in pca_df.columns[5:]}
        pca_df = pca_df.astype(casting_dict)
        if c == 1:
            pca_features = pd.DataFrame({"id": pca_df.columns[5:]})
            pca_features.to_csv(f"{out_prefix}.borzoi_pca102.txt",index=None, header=None)
        all_df = baseline_df.merge(pca_df, on=["SNP","CHR", "BP", "A1", "A2"])
        all_features = pd.DataFrame({"id": all_df.columns[5:]})
        all_features.to_csv(f"{out_prefix}.all_pca102.txt",index=None, header=None)
        all_df.to_parquet(f"{out_prefix}.{c}.annot.parquet")

def borzoi_h5_to_zarr(path, out_prefix):
    with h5py.File(path, 'r') as f:
        chr = np.array([x.removeprefix("chr") for x in np.char.decode(f["chr"][:], encoding="utf-8")])
        bp = f["pos"][:]
        a1 = np.char.decode(f["ref_allele"][:], encoding="utf-8")
        a2 = np.char.decode(f["alt_allele"][:], encoding="utf-8")
        chroms = np.unique(chr)
        snps = []
        for c, p, a_1, a_2 in zip(chr, bp, a1, a2):
            snps.append(f"{c}:{p}:{a_1}:{a_2}")
        for c in chroms:
            print(c)
            keep = np.where(chr == c)[0]
            out = f["logD2"][keep,:].astype(np.float32)
            zarr.create_array(store=f"{out_prefix}.zarr", name=c, data=out, overwrite=True)
            del out

            meta = {"SNP": snps,
                    "CHR": chr,
                    "BP": bp,
                    "A1": a1,
                    "A2": a2}
            pd.DataFrame(meta).to_parquet(f"{out_prefix}.{c}.annot.parquet")

def borzoi_h5_to_npy(path, out_prefix):
    with h5py.File(path, 'r') as f:
        chr = np.array([x.removeprefix("chr") for x in np.char.decode(f["chr"][:], encoding="utf-8")])
        chroms = np.unique(chr)
        for c in chroms:
            print(c)
            keep = np.where(chr == c)[0]
            out = f["logD2"][keep,:].astype(np.float32)
            np.save(f"{out_prefix}.{c}.annot.npy", out)
            del out

if __name__ == '__main__':
    #borzoi_h5_to_zarr("/home/divyanshi/projects/finemapping/borzoi_veps_vstitch_merged/scores.h5", 
    #                  "/scratch4/davidwang/datasets/ukbb/annotation/borzoi/scores")

    #borzoi_h5_to_npy("/home/divyanshi/projects/finemapping/borzoi_veps_vstitch_merged/scores.h5", 
    #                  "/scratch4/davidwang/datasets/ukbb/annotation/borzoi/scores")

    #borzoi_h5_baseline_parquet_to_parquet("/home/divyanshi/projects/finemapping/borzoi_veps_vstitch_merged/scores.h5", 
    #                                    "/home/divyanshi/projects/finemapping/baseline_annots/baselineLF2.2.UKB/baselineLF2.2.UKB",
    #                                    "/scratch4/davidwang/datasets/ukbb/annotation/all/scores",
    #                                    "/home/divyanshi/projects/finemapping/borzoi_veps/vcf/bolt_337k.txt")

    borzoi_pca_make_meta("/home/divyanshi/projects/finemapping/borzoi_annots/e-pca-95perc-20/borzoi",
                        "/home/divyanshi/projects/finemapping/baseline_annots/e_baselineLF2.2.UKB/baselineLF2.2.UKB",
                        "/scratch4/davidwang/datasets/ukbb/annotation/borzoi_pca102/scores")


    #borzoi_h5_baseline_parquet_to_parquet("/home/divyanshi/projects/finemapping/borzoi_veps_vstitch_merged/scores.h5", 
    #                                    "/home/divyanshi/projects/finemapping/baseline_annots/baselineLF2.2.UKB/baselineLF2.2.UKB",
    #                                    "/scratch4/davidwang/datasets/ukbb/annotation/all_scaled/scores",
    #                                    scaled_h5_file="/home/divyanshi/projects/finemapping/borzoi_veps_vstitch_merged/scores_dm_scaled.h5")