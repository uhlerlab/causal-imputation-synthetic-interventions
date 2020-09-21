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
        print("Log2")
        data = np.log2(data+1)
    if minmax:
        print("Minmax")
        data = pandas_minmax(data, axis=1)
        print("Mean")
    
    means = data.groupby(level=['cell_id', PERT_ID_FIELD]).mean()
    print("Saving")
    means.to_pickle(f"data/processed/averages/{name}{optional_str('_log2', log2)}{optional_str('_minmax', minmax)}.pkl")
    print(f"Computing/saving averages took {time() - start} seconds")


if __name__ == '__main__':
    from filenames import load_cmap_filtered, load_cmap_imputed, load_cmap_original, load_cmap_level3

    files = {
        'level2_filtered': load_cmap_filtered,
        'level2_imputed': load_cmap_imputed,
        'level2': load_cmap_original,
        'level3': load_cmap_level3
    }

    for name, data_loader in files.items():
        data = data_loader()
        main(data, name, log2=False, minmax=False)
        main(data, name, log2=True, minmax=True)

