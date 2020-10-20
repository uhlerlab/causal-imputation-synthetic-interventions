# perturbation-transportability

You can setup a virtual environment with all necessary packages by running
```
bash setup.sh
```

The data files are too large to be kept in the repository, but can be downloaded by running `download.sh`.

The files are organized as follows:
* `src` contains the baseline algorithms and the synthetic interventions algorithm
* `processing` contains scripts for processing the raw data
* `evaluation` contains classes for evaluating the performance of various algorithms
* `scratch` contains temporary files for one-off tasks, such as checking that new code works

Once you have downloaded the data, you can run the processing scripts in the correct order via
```
bash process.sh
```

Figures can be reproduced via the following files:
* **Figure 2**: visuals/umap/umap_all.py
* **Figure 3a**: evaluation/vary_perturbation_block.py
* **Figure 3b**: evaluation/plot_statistics.py
* **Figure 6a, 7a, 7b**: visuals/plot_availability_matrix.py
* **Figure 6b, 7c, 7d**:
* **Figure 8**: visuals/umap/plot_umap_interventions_single_celltype.py
* **Figure 9**: evaluation/vary_perturbation_block.py