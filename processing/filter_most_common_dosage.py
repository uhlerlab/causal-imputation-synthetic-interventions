from filenames import load_cmap_original, load_cmap_filtered, load_cmap_imputed, load_cmap_level3, load_inst_info_epsilon
from filenames import PERT_ID_FIELD
import ipdb
from time import time
import os

pert_meta_cols = ['pert_dose', 'pert_dose_unit', 'pert_time', 'pert_time_unit']
inst_info = load_inst_info_epsilon()
inst_info['pert_dose_full'] = inst_info['pert_dose'].astype(str) + inst_info['pert_dose_unit'].astype(str)
inst_info['pert_time_full'] = inst_info['pert_time'].astype(str) + inst_info['pert_time_unit']
inst_info['pert_meta_full'] = inst_info['pert_dose_full'] + ',' + inst_info['pert_time_full']
inst_info['pert_id_meta'] = inst_info[PERT_ID_FIELD] + ',' + inst_info['pert_meta_full']
dosage_counts = inst_info.groupby([PERT_ID_FIELD, 'pert_meta_full']).size()
dosage_counts.rename('count', inplace=True)
dosage_counts.sort_values(inplace=True, ascending=False)
dosage_count_df = dosage_counts.reset_index('pert_meta_full')
most_common_dosage_df = dosage_count_df[~dosage_count_df.index.duplicated(keep='first')]
assert most_common_dosage_df.shape[0] == inst_info[PERT_ID_FIELD].nunique()

most_common_dosages = set(most_common_dosage_df.index.to_series() + ',' + most_common_dosage_df['pert_meta_full'])
retained_inst_info = inst_info.query('pert_id_meta in @most_common_dosages')
retained_inst_ids = retained_inst_info.index
assert retained_inst_info.shape[0] == most_common_dosage_df['count'].sum()


def main(data, name):
    # filter data to only have inst_ids corresponding to the most common dosages for each pert_id
    os.makedirs('data/processed/most_common_dosages', exist_ok=True)
    start = time()
    retained_data = data.loc[retained_inst_ids]
    print(f"Filtering took {time() - start} seconds")
    retained_data.to_pickle(f'data/processed/most_common_dosages/{name}.pkl')


if __name__ == '__main__':
    files = {
        # 'level2_filtered': load_cmap_filtered,
        'level2_imputed': load_cmap_imputed,
        # 'level2': load_cmap_original,
        # 'level3': load_cmap_level3
    }

    for name, data_loader in files.items():
        data = data_loader()
        main(data, name)


