#!/usr/bin/env bash

mkdir -p data/raw
cd data/raw

# inst info
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Finst%5Finfo%2Etxt%2Egz
gunzip GSE92742_Broad_LINCS_cell_info.txt.gz

# gene info
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fgene%5Finfo%2Etxt%2Egz
gunzip GSE92742_Broad_LINCS_gene_info.txt.gz

# cell info
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fcell%5Finfo%2Etxt%2Egz
gunzip GSE92742_Broad_LINCS_inst_info.txt.gz

# perturbation info
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz
gunzip GSE92742_Broad_LINCS_pert_info.txt.gz

wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_metrics.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_metrics.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_auxiliary_datasets.tar.gz
tar -xzf GSE92742_Broad_LINCS_auxiliary_datasets.tar.gz

# expression info
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel2%5FGEX%5Fdelta%5Fn49216x978%2Egctx%2Egz
gunzip GSE92742_Broad_LINCS_Level2_GEX_delta_n49216x978.gctx.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel2%5FGEX%5Fepsilon%5Fn1269922x978%2Egctx%2Egz
gunzip GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz
gunzip GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz

cd ..

# if saving to a different directory, soft link to /data
#cd data
#ln -s /mnt/covid19/l1000/GSE92742_Broad_LINCS_Level2_GEX_delta_n49216x978.gctx GSE92742_Broad_LINCS_Level2_GEX_delta_n49216x978.gctx
#ln -s /mnt/covid19/l1000/GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx
#ln -s /mnt/covid19/l1000/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx
#ln -s /mnt/covid19/l1000/GSE92742_Broad_LINCS_cell_info.txt GSE92742_Broad_LINCS_cell_info.txt
#ln -s /mnt/covid19/l1000/GSE92742_Broad_LINCS_gene_info.txt GSE92742_Broad_LINCS_gene_info.txt
#ln -s /mnt/covid19/l1000/GSE92742_Broad_LINCS_inst_info.txt GSE92742_Broad_LINCS_inst_info.txt
#ln -s /mnt/covid19/l1000/GSE92742_Broad_LINCS_pert_info.txt GSE92742_Broad_LINCS_pert_info.txt

# must filter institution info to match gene expression data
python filter_inst_info.py

wget ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.owl.gz
gunzip chebi_lite.owl.gz
