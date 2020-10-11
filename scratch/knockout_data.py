from filenames import load_pert_info
from filenames import load_cmap_original
import os
import pandas as pd


def get_ko_data():
    filename = 'data/processed/ko_data.pkl'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    else:
        pert_info = load_pert_info()
        ko_pert_ids = set(pert_info[pert_info['pert_type'] == 'trt_sh']['pert_id'])
        data = load_cmap_original()
        filtered_data = data[data.index.get_level_values('pert_id').isin(ko_pert_ids)]
        filtered_data.to_pickle(filename)
        return filtered_data


ko_data = get_ko_data()
