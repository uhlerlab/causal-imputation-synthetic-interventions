import pandas as pd
from evaluation.helpers.get_data_block import get_data_block, remove_celltypes_no_control

old_df0 = pd.read_csv('old_data/df0.csv')
old_df1 = pd.read_csv('old_data/df1_all.csv', sep='\t', index_col=0)
new_df, _, _, _ = get_data_block(0, None, 0, 20)
new_df_ = remove_celltypes_no_control(new_df)

old_units = set(old_df1.unit)
old_ivs = set(old_df1.intervention)
new_units = set(new_df.index.get_level_values('unit'))
new_ivs = set(new_df.index.get_level_values('intervention'))
