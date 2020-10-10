import numpy as np
from utils import pandas_minmax
from filenames import PERT_ID_FIELD
import os
from time import time
from utils import optional_str


def main(data, name, log2=False, minmax=False):
    os.makedirs("data/processed/averages", exist_ok=True)

    start = time()
    if log2:
        print("[processing/create_averages] Log2")
        data = np.log2(data+1)
    if minmax:
        print("[processing/create_averages] Minmax")
        data = pandas_minmax(data, axis=1)

    print("[processing/create_averages] Mean")
    means = data.groupby(level=['cell_id', PERT_ID_FIELD]).mean()
    print("[processing/create_averages] Saving")
    means.to_pickle(f"data/processed/averages/{name}{optional_str('_log2', log2)}{optional_str('_minmax', minmax)}.pkl")
    print(f"[processing/create_averages] Computing/saving averages took {time() - start} seconds")


if __name__ == '__main__':
    from filenames import load_cmap_filtered, load_cmap_imputed, load_cmap_original, load_cmap_level3
    from filenames import load_cmap_most_common_filtered, load_cmap_most_common_imputed, load_cmap_most_common_original, load_cmap_most_common_level3

    files = {
        # 'level2_filtered': load_cmap_filtered,
        # 'level2_imputed': load_cmap_imputed,
        # 'level2': load_cmap_original,
        # 'level3': load_cmap_level3,
        'level2_filtered_common': load_cmap_most_common_filtered,
        'level2_imputed_common': load_cmap_most_common_imputed,
        'level2_common': load_cmap_most_common_original,
        'level3_common': load_cmap_most_common_level3,
    }

    for name, data_loader in files.items():
        data = data_loader()
        main(data, name, log2=False, minmax=False)
        main(data, name, log2=True, minmax=True)

