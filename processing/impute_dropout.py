"""
Create a "corrected" version of the dataset.

Gene expression measurements of 1 correspond to missing values. Wherever possible, we replace these values
by the mean expression level for the same gene, in the same perturbation/cell type pair. If the value of
a gene is missing for all samples from the same perturbation/cell type pair, we replace the value with the
average of *all* genes in the same perturbation/cell type pair.
"""

from filenames import LINCS2_EPSILON_IMPUTED_FILE, load_cmap_original, LINCS2_EPSILON_825_FILE, save_gctx
from filenames import PERT_ID_FIELD, PERT_OTHER_FIELD
import numpy as np
from tqdm import tqdm
tqdm.pandas()

print("Loading data")
original_data = load_cmap_original()
print("Replacing 1 with NaN")
data = original_data.replace(1, np.nan)

print("Filtering out genes with any NaNs")
retained_genes = ~data.isna().any(axis=0)
filtered_data = data.loc[:, retained_genes]
save_gctx(filtered_data, LINCS2_EPSILON_825_FILE)

print("Computing means")
means_per_gene = data.groupby(level=['cell_id', PERT_ID_FIELD]).mean()
means_per_group = data.groupby(level=['cell_id', PERT_ID_FIELD]).progress_apply(lambda x: np.nanmean(x.values))
means_per_gene = means_per_gene.mask(means_per_gene.isna(), means_per_group, axis=0)
print("Repeating means")
sizes = data.groupby(level=['cell_id', PERT_ID_FIELD]).size()
repeated_means = np.zeros(data.shape)
for key, ixs in tqdm(data.groupby(level=['cell_id', PERT_ID_FIELD]).indices.items()):
    repeated_means[ixs] = means_per_gene.loc[key]
print("Using mask")
imputed_data = data.mask(data.isna(), repeated_means)

print("Saving imputed data")
imputed_data = imputed_data
imputed_data.index = original_data.index.get_level_values('inst_id')
save_gctx(imputed_data, LINCS2_EPSILON_IMPUTED_FILE)
print(f"Are there any NaN's remaining in the imputed data? {imputed_data.isna().any().any()}")
