#!/usr/bin/env bash

# remove inst_id's from inst_info which are from the delta probes
python3 processing/filter_inst_info_epsilon.py
# convert the gctx file to a pandas pickle file for faster read/write
python3 processing/convert_to_pickle.py
# for each pert_id, pick the dosage amount/time which has the greatest number of samples
python3 processing/filter_most_common_dosage.py
# create imputed/filtered versions of the level 2 data
python3 processing/impute_dropout.py
# create a pandas pickle file for the level 3 data with only l1000 genes
python3 processing/prune_level3.py
# create files ranking the pert_id's and cell_id's by the number of available entries
python3 processing/create_rank_files.py
# create files containing the average gene expression vectors for each cell_id/pert_id pair
python3 processing/create_averages.py
