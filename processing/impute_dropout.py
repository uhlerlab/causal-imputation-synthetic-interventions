from cmapPy.pandasGEXpress.parse import parse
from cmapPy.pandasGEXpress.write_gctx import write
from cmapPy.pandasGEXpress.GCToo import GCToo
from filenames import LINCS2_EPSILON_FILE, load_inst_info, LINCS2_EPSILON_IMPUTED_FILE
import numpy as np
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

print("Loading data")
data = parse(LINCS2_EPSILON_FILE).data_df.T
print("Replacing 1 with NaN")
data.replace(1, np.nan)

print("Adding cell_id/pert_id rows to data")
inst_info = load_inst_info()
inst_info = inst_info.set_index('inst_id')
inst_info = inst_info.loc[data.index]
data['cell_id'] = inst_info['cell_id'].values
data['pert_id'] = inst_info['pert_id'].values
data = data.set_index(['cell_id', 'pert_id'])


print("Computing means")
means = data.groupby(level=['cell_id', 'pert_id']).mean()
print("Repeating means")
repeated_means = pd.concat((means.loc[ix]for ix in data.index), axis=1).T
print("Using mask")
imputed_data = data.mask(data.isna(), repeated_means)

# print("Saving imputed data")
# gctoo = GCToo(imputed_data)
# write(gctoo, LINCS2_EPSILON_IMPUTED_FILE)
