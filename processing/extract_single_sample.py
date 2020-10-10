from time import time
from sklearn.utils import shuffle
import os

os.makedirs('data/processed/single_samples/', exist_ok=True)

random_state = 231321


def main(data, name):
    start = time()
    shuffled_data = shuffle(data, random_state=random_state)
    random_data = shuffled_data.groupby(level=['cell_id', 'pert_id']).first()
    print(f"Took {time() - start} seconds to sample")

    random_data.to_pickle(f"data/processed/single_samples/{name}.pkl")


if __name__ == '__main__':
    from filenames import load_cmap_original, load_cmap_filtered, load_cmap_imputed, load_cmap_level3
    from filenames import load_cmap_most_common_original, load_cmap_most_common_filtered, load_cmap_most_common_imputed
    from filenames import load_cmap_most_common_level3

    files = {
        'level2_filtered': load_cmap_filtered,
        'level2_imputed': load_cmap_imputed,
        'level2': load_cmap_original,
        'level3': load_cmap_level3,
        'level2_filtered_common': load_cmap_most_common_filtered,
        'level2_imputed_common': load_cmap_most_common_imputed,
        'level2_common': load_cmap_most_common_original,
        'level3_common': load_cmap_most_common_level3,
    }

    for name, data_loader in files.items():
        data = data_loader()
        main(data, name)
