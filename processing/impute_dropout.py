"""
Create a "corrected" version of the dataset.

Gene expression measurements of 1 correspond to missing values. Wherever possible, we replace these values
by the mean expression level for the same gene, in the same perturbation/cell type pair. If the value of
a gene is missing for all samples from the same perturbation/cell type pair, we replace the value with the
average of *all* genes in the same perturbation/cell type pair.
"""

from cmapPy.pandasGEXpress.write_gctx import write
from cmapPy.pandasGEXpress.GCToo import GCToo
from filenames import load_inst_info, LINCS2_EPSILON_IMPUTED_FILE, load_cmap_original
import numpy as np
from tqdm import tqdm
tqdm.pandas()

print("Loading data")
original_data = load_cmap_original().T
print("Replacing 1 with NaN")
data = original_data.replace(1, np.nan)

print("Adding cell_id/pert_id rows to data")
inst_info = load_inst_info()
inst_info = inst_info.set_index('inst_id')
inst_info = inst_info.loc[data.index]
data['cell_id'] = inst_info['cell_id'].values
data['pert_id'] = inst_info['pert_id'].values
data = data.set_index(['cell_id', 'pert_id'])

print("Computing means")
means_per_gene = data.groupby(level=['cell_id', 'pert_id']).mean()
means_per_group = data.groupby(level=['cell_id', 'pert_id']).progress_apply(lambda x: np.nanmean(x.values))
means_per_gene = means_per_gene.mask(means_per_gene.isna(), means_per_group, axis=0)
print("Repeating means")
sizes = data.groupby(level=['cell_id', 'pert_id']).size()
repeated_means = np.zeros(data.shape)
for key, ixs in tqdm(data.groupby(level=['cell_id', 'pert_id']).indices.items()):
    repeated_means[ixs] = means_per_gene.loc[key]
print("Using mask")
imputed_data = data.mask(data.isna(), repeated_means)

print("Saving imputed data")
imputed_data = imputed_data.T
imputed_data.columns = original_data.index
gctoo = GCToo(imputed_data)
write(gctoo, LINCS2_EPSILON_IMPUTED_FILE)
print(f"Are there any NaN's remaining in the imputed data? {imputed_data.isna().any().any()}")
