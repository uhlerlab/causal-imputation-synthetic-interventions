#!/usr/bin/env bash

python3 -m visuals.umap.umap_all
python3 -m visuals.umap.plot_umap_interventions_single_celltype
python3 -m visuals.plot_availability_original
python3 -m visuals.plot_availability_subset
python3 -m evaluation.vary_perturbation_block
python3 -m evaluation.plot_statistics
