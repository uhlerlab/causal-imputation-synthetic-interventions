#!/usr/bin/env bash

#python3 -m visuals.umap.umap_all
#python3 -m visuals.umap.plot_umap_interventions_single_celltype
#python3 -m visuals.plot_availability_original
#python3 -m visuals.plot_availability_subset
python3 -m evaluation.main
#python3 -m evaluation.plot_statistics
#python3 -m evaluation.plot_donors_training_error
python3 -m evaluation.plot_donor_vs_r2
