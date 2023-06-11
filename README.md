# Causal Imputation via Synthetic Interventions

This code accompanies [Causal Imputation via Synthetic Interventions](https://arxiv.org/abs/2011.03127). See [here](https://scholar.google.com/scholar?cluster=9041495278699782854&hl=en&as_sdt=0,22) for the citation.

The files are organized as follows:
* `src` contains the baseline algorithms and the synthetic interventions algorithm
* `processing` contains scripts for processing the raw data
* `evaluation` contains classes for evaluating the performance of various algorithms
* `scratch` contains temporary files for one-off tasks, such as checking that new code works

### Replicating data processing from the raw data [Optional]
The data files are too large (5Gb) to be kept in the repository, but can be downloaded by running `download.sh`.

Once you have downloaded the data, you can run the processing scripts in the correct order via
```
bash process.sh
```
This will create ~44Gb of processed data, and may take about 1 hour.

### Replicating algorithm results and plots

For convenience, the processed data is available at the following anonymized link: [https://drive.google.com/drive/folders/1WKZRHY-v2zgu6XZqXLiCwEArqP4KfziR?usp=sharing](https://drive.google.com/drive/folders/1WKZRHY-v2zgu6XZqXLiCwEArqP4KfziR?usp=sharing). 
Copy the contents of this folder to **data/processed/**

To re-create the results from the paper, first set up a virtual environment with
the necessary packages by running
```
bash setup.sh
```
Then, run the algorithms and the plotting code via
```
source venv/bin/activate
bash create_plots.sh
```
Because of the reliance on parts of the raw dataset, this script does not include calls to
the functions which create the UMAP plots and the plots of which cell/drug pairs are
available. This will be remedied when publicly releasing the code.
