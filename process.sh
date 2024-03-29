#!/usr/bin/env bash

mkdir -p data/processed
mkdir -p ~/Desktop/cmap-imputation

# (1) remove inst_id's from inst_info which are from a different protocol than the rest of the dataset
python3 -m processing.filter_inst_info_epsilon

# (2) convert the gctx file to a pandas pickle file for faster read/write
python3 -m processing.convert_to_pickle

# (2) create a pandas pickle file for the level 3 data which contains only the l1000 genes
#python3 processing/prune_level3

# (4) create imputed/filtered versions of the level 2 data
python3 -m processing.impute_dropout

# (3) for each pert_id, pick the dosage amount/time which has the greatest number of samples as the "canonical" perturbation
python3 -m processing.filter_most_common_dosage

# (4) create files ranking the pert_id's and cell_id's by the number of available entries
python3 -m processing.create_rank_files

# (5) create files containing the average gene expression vectors for each cell_id/pert_id pair
python3 -m processing.create_averages

# (5) create files containing the average gene expression vectors for each cell_id/pert_id pair
python3 -m processing.extract_single_sample
