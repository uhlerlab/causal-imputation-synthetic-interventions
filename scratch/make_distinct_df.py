from filenames import load_pert_info
from sklearn.utils import shuffle
import os
os.makedirs('data/processed/distinct_all_samples/', exist_ok=True)

random_state = 12310893


pert_inames = {
    'DMSO',
    'testosterone',
    'estriol'
}
pert_info = load_pert_info()
pert_ids = set(pert_info[pert_info['pert_iname'].isin(pert_inames)]['pert_id'])
print(f"Number of pert_id's : {len(pert_ids)}")


def main(data, name):
    filtered_data = data[data.index.get_level_values('pert_id').isin(pert_ids)]

    filtered_data.to_pickle(f'data/processed/distinct_all_samples/{name}.pkl')

    print(f"Averages")
    averages = filtered_data.groupby(['cell_id', 'pert_id']).mean()
    averages.to_pickle(f'data/processed/averages/{name}_distinct.pkl')

    print("Single samples")
    shuffled_data = shuffle(filtered_data, random_state=random_state)
    random_data = shuffled_data.groupby(level=['cell_id', 'pert_id']).first()
    random_data.to_pickle(f'data/processed/single_samples/{name}_distinct.pkl')


if __name__ == '__main__':
    from filenames import load_cmap_filtered, load_cmap_imputed, load_cmap_original, load_cmap_level3

    files = {
        'level2_filtered': load_cmap_filtered,
        # 'level2_imputed': load_cmap_imputed,
        # 'level2': load_cmap_original,
        # 'level3': load_cmap_level3,
    }

    for name, data_loader in files.items():
        data = data_loader()
        main(data, name)
        main(data, name)
