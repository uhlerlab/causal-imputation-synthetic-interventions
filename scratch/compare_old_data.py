import pandas as pd
from evaluation.helpers.get_data_block import get_data_block, remove_celltypes_no_control
from numpy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt

old_df0 = pd.read_csv('old_data/df0.csv', sep='\t', index_col=0)
old_df1 = pd.read_csv('old_data/df1_all.csv', sep='\t', index_col=0)
new_df, _, _, _ = get_data_block(0, None, 0, 20, 'level2_filtered_common')

old_df = pd.concat([old_df0, old_df1])
old_df.drop(columns=['unit', 'intervention'], inplace=True)
_, s_old, _ = svd(old_df.values)
_, s_new, _ = svd(new_df.values)

plt.clf()
plt.plot(np.cumsum(s_old), label='old')
plt.plot(np.cumsum(s_new), label='new')
plt.legend()
plt.savefig('spectra.png')

old_units = set(old_df1.unit)
old_ivs = set(old_df1.intervention)
new_units = set(new_df.index.get_level_values('unit'))
new_ivs = set(new_df.index.get_level_values('intervention'))

print(f"old # units: {len(old_units)}")
print(f"new # units: {len(new_units)}")
print(f"old # ivs: {len(old_ivs)}")
print(f"new # ivs: {len(new_ivs)}")
print(f"difference in units: {len(old_units.symmetric_difference(new_units))}")
print(f"difference in ivs: {len(old_ivs.symmetric_difference(new_ivs))}")
