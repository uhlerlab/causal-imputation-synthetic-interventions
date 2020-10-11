import matplotlib.pyplot as plt
from filenames import load_pert_info
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from utils import pandas_minmax
import umap

DO_PCA = True
if DO_PCA:
    reducer = PCA(n_components=2)
else:
    reducer = umap.UMAP()

LOG2 = True
MINMAX = True
colormap = plt.get_cmap('tab20')
sns.set()

cell_ids = {
    'MCF7',
    'A549',
    # 'A375',
    # 'PC3'
}
# data = pd.read_pickle('data/processed/distinct_all_samples/level2_filtered.pkl')
# data = data[data.index.get_level_values('pert_id') != 'DMSO']
data = pd.read_pickle('data/processed/ko_data.pkl')
data = data[data.index.get_level_values('cell_id').isin(cell_ids)]

if LOG2:
    data = np.log2(data+1)
if MINMAX:
    data = pandas_minmax(data, axis=1)


pert_info = load_pert_info()
pert_ids = set(data.index.get_level_values('pert_id'))
pert_info = pert_info[pert_info['pert_id'].isin(pert_ids)]
pert_id2pert_iname = pert_info[['pert_id', 'pert_iname']].set_index('pert_id').squeeze().to_dict()
data['pert_iname'] = data.index.get_level_values('pert_id').map(pert_id2pert_iname)
data.set_index('pert_iname', append=True, inplace=True)
pert_inames = set(data.index.get_level_values('pert_iname'))

target_pert_inames = {'SEC16A', 'ANO10'}
target_pert_info = pert_info[pert_info['pert_iname'].isin(target_pert_inames)]
print(target_pert_info)
data = data[data.index.get_level_values('pert_iname').isin(target_pert_inames)]
# target_pert_ids = {'TRCN0000000620', 'TRCN0000040152'}
# data = data[data.index.get_level_values('pert_id').isin(target_pert_ids)]

print(f"Embedding {data.shape[0]} samples")
embedded_values = reducer.fit_transform(data.values)

print("Plotting, colored by pert_iname")
plt.clf()
pert2ix = {pert: ix for ix, pert in enumerate(pert_inames)}
colors = data.index.get_level_values('pert_iname').map(pert2ix)
plt.scatter(embedded_values[:, 0], embedded_values[:, 1], c=colors, cmap=colormap)
plt.legend()
plt.savefig('scratch/distinct_pca_pert_iname_coloring.png')

print("Plotting, colored by pert_id")
plt.clf()
pert2ix = {pert: ix for ix, pert in enumerate(pert_ids)}
colors = data.index.get_level_values('pert_id').map(pert2ix)
plt.scatter(embedded_values[:, 0], embedded_values[:, 1], c=colors, cmap=colormap)
plt.legend()
plt.savefig('scratch/distinct_pca_pert_id_coloring.png')

print("Plotting, colored by cell type")
plt.clf()
cell2ix = {cell: ix for ix, cell in enumerate(set(data.index.get_level_values('cell_id')))}
colors = data.index.get_level_values('cell_id').map(cell2ix)
plt.scatter(embedded_values[:, 0], embedded_values[:, 1], c=colors, cmap=colormap)
plt.legend()
plt.savefig('scratch/distinct_pca_celltype_coloring.png')
