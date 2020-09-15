import numpy as np
from utils import pandas_minmax
from filenames import PERT_ID_FIELD
import os
from time import time


def main(data_loader, name):
    data = data_loader()
    os.makedirs("data/processed/averages", exist_ok=True)

    start = time()
    print("Log2")
    data = np.log2(data+1)
    print("Minmax")
    data = pandas_minmax(data, axis=1)
    print("Mean")
    means = data.groupby(level=['cell_id', PERT_ID_FIELD]).mean()
    print("Saving")
    means.to_pickle(f"data/processed/averages/{name}.pkl")
    print(f"Computing/saving averages took {time() - start} seconds")


if __name__ == '__main__':
    from filenames import load_cmap_filtered, load_cmap_imputed

    files = {
        'level2_filtered': load_cmap_filtered,
        'level2_imputed': load_cmap_imputed
    }

    for name, data in files.items():
        main(data, name)

