"""
Create a "corrected" version of the dataset.

Gene expression measurements of 1 correspond to missing values. Wherever possible, we replace these values
by the mean expression level for the same gene, in the same perturbation/cell type pair. If the value of
a gene is missing for all samples from the same perturbation/cell type pair, we replace the value with the
average of *all* genes in the same perturbation/cell type pair.
"""

from filenames import LINCS2_EPSILON_IMPUTED_FILE, load_cmap_original, LINCS2_EPSILON_825_FILE
from filenames import PERT_ID_FIELD
from filenames import NUM_DROPOUTS_FILE
import numpy as np
from tqdm import tqdm
from time import time
import pandas as pd
tqdm.pandas()

original_data = load_cmap_original()
print("[processing/impute_dropout] Replacing 1 with NaN")
start = time()
data = pd.DataFrame(
    np.where(original_data.values != 1, original_data.values, np.nan),
    index=original_data.index,
    columns=original_data.columns
)
print(f"[processing/impute_dropout] Replacing took {time() - start} seconds")
num_dropouts = data.isna().sum(axis=1)
num_dropouts.to_pickle(NUM_DROPOUTS_FILE)

print("[processing/impute_dropout] Filtering out genes with any NaNs")
start = time()
retained_genes = ~np.isnan(data.values).any(axis=0)
filtered_data = data.loc[:, retained_genes]
print(f"[processing/impute_dropout] Filtering took {time() - start} seconds. {sum(retained_genes)} left.")
start = time()
filtered_data.to_pickle(LINCS2_EPSILON_825_FILE)
print(f"[processing/impute_dropout] Saving filtered genes took {time() - start} seconds.")

print("[processing/impute_dropout] Computing means")
means_per_gene = data.groupby(level=['cell_id', PERT_ID_FIELD]).mean()
means_per_group = data.groupby(level=['cell_id', PERT_ID_FIELD]).progress_apply(lambda x: np.nanmean(x.values))
means_per_gene = means_per_gene.mask(means_per_gene.isna(), means_per_group, axis=0)
print("[processing/impute_dropout] Repeating means")
sizes = data.groupby(level=['cell_id', PERT_ID_FIELD]).size()
repeated_means = np.zeros(data.shape)
for key, ixs in tqdm(data.groupby(level=['cell_id', PERT_ID_FIELD]).indices.items()):
    repeated_means[ixs] = means_per_gene.loc[key]
print("[processing/impute_dropout] Using mask")
start = time()
# imputed_data = data.mask(data.isna(), repeated_means)
imputed_data = pd.DataFrame(np.where(~np.isnan(data), data, np.nan), index=data.index, columns=data.columns)
print(f"[processing/impute_dropout] Mask took {time() - start} seconds")

print("[processing/impute_dropout] Saving imputed data")
start = time()
imputed_data.index = original_data.index.get_level_values('inst_id')
imputed_data.to_pickle(LINCS2_EPSILON_IMPUTED_FILE)
print(f"[processing/impute_dropout] Saving imputed data took {time() - start} seconds.")
print(f"[processing/impute_dropout] Are there any NaN's remaining in the imputed data? {imputed_data.isna().any().any()}")
