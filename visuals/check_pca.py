from cmapPy.pandasGEXpress.parse import parse
import os
from filenames import _format_cmap, load_num_dropouts
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import pandas_minmax
import ipdb
cm = plt.get_cmap('plasma')
sns.set()

cell_ids = ['MCF7', 'A549']


def main(name, cmap_file):
    os.makedirs('visuals/figures/pcas/', exist_ok=True)
    data = parse(cmap_file).data_df
    data = _format_cmap(data)
    num_dropouts = load_num_dropouts()

    pert_id = 'DMSO'
    fig, axes = plt.subplots(len(cell_ids), 1)
    for ax, cell_id in zip(axes, cell_ids):
        cell_data = data.query('cell_id == @cell_id and pert_id == @pert_id')
        cell_data = np.log2(cell_data+1)
        cell_data = pandas_minmax(cell_data, axis=1)
        num_dropouts_cell = num_dropouts.loc[cell_data.index.get_level_values('inst_id')]
        print(cell_data.max(axis=1))
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(cell_data)
        print(pca_data.shape)
        ax.scatter(pca_data[:, 0], pca_data[:, 1], c=num_dropouts_cell.values, cmap=cm)
        ax.set_title(cell_id)
    plt.savefig(f'visuals/figures/pcas/{name}.png')


if __name__ == '__main__':
    from filenames import LINCS2_EPSILON_FILE_GCTX, LINCS2_EPSILON_IMPUTED_FILE, LINCS2_EPSILON_825_FILE, LINCS3_PRUNED_FILE

    files = {
        'original_level2': LINCS2_EPSILON_FILE_GCTX,
        'imputed_level2': LINCS2_EPSILON_IMPUTED_FILE,
        'filtered_level2': LINCS2_EPSILON_825_FILE,
        'level3': LINCS3_PRUNED_FILE,
    }

    for name, file in files.items():
        print(name)
        main(name, file)
