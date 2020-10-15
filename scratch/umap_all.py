import matplotlib.pyplot as plt
from filenames import load_pert_info, load_cmap_original
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from utils import pandas_minmax
import umap
from matplotlib.patches import Patch
from evaluation.helpers.get_data_block import get_data_block
from sklearn.utils import shuffle
import itertools as itr
sns.set()

DO_PCA = False
if DO_PCA:
    reducer = PCA(n_components=2)
else:
    reducer = umap.UMAP()

colormap = plt.get_cmap('tab20')
colormap = list(itr.chain.from_iterable(itr.repeat(colormap.colors, 4)))

data = load_cmap_original()
control_data = data[data.index.get_level_values("pert_id") == "DMSO"]
control_data = shuffle(control_data)
control_data = control_data.groupby("cell_id").head(100)
pert_data = data[data.index.get_level_values("pert_id") != "DMSO"]
print("Shuffling")
pert_data = shuffle(pert_data)
print("Picking heads")
pert_data = pert_data.groupby("cell_id").head(100)
data = pd.concat([control_data, pert_data])

LOG2 = True
MINMAX = True
if LOG2:
    print("log2")
    data = np.log2(data+1)
if MINMAX:
    print("minmax")
    data = pandas_minmax(data, axis=1)
print("Embedding")
embedded_data = reducer.fit_transform(data)

num_control = control_data.shape[0]
print("Plotting, colored by cell type")
plt.clf()
fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(12, 6)
ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')
cell2ix = {cell: ix for ix, cell in enumerate(set(data.index.get_level_values('cell_id')))}
control_colors = control_data.index.get_level_values('cell_id').map(cell2ix)
control_colors = [colormap[c] for c in control_colors]
pert_colors = pert_data.index.get_level_values('cell_id').map(cell2ix)
pert_colors = [colormap[c] for c in pert_colors]
# ax1.scatter(embedded_data[:num_control, 0], embedded_data[:num_control, 1], c=control_colors, alpha=.1, s=100, linewidths=1)
# ax2.scatter(embedded_data[num_control:, 0], embedded_data[num_control:, 1], c=pert_colors, marker='P', s=20)
ax1.scatter(embedded_data[:num_control, 0], embedded_data[:num_control, 1], c=control_colors)
ax2.scatter(embedded_data[num_control:, 0], embedded_data[num_control:, 1], c=pert_colors)
ax1.set_title('Control')
ax2.set_title('Perturbation')
lgd = plt.legend(
    handles=[
        Patch(color=colormap[ix], label=celltype) for celltype, ix in cell2ix.items()
    ],
    ncol=4,
    bbox_to_anchor=(1.05, 1),
    fontsize='small'
)
# plt.title('Control')
# plt.savefig('scratch/umap_celltype_coloring_dmso.png', bbox_inches='tight')
plt.savefig('scratch/umap_celltype_coloring.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

# plt.clf()
# plt.scatter(embedded_data[num_control:, 0], embedded_data[num_control:, 1], c=pert_colors)
# plt.gca().set_yticklabels([])
# plt.title('Perturbation')
# plt.savefig('scratch/umap_celltype_coloring_pert.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

# print("Plotting, colored by pert type")
# plt.clf()
# pert2ix = {pert: ix for ix, pert in enumerate(set(data.index.get_level_values('intervention')))}
# colors = data.index.get_level_values('intervention').map(pert2ix)
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=colors, cmap=colormap)
# plt.legend(handles=[
#     Patch(color=colormap(ix), label=celltype) for celltype, ix in pert2ix.items()
# ])
# plt.savefig('scratch/umap_pert_coloring.png')
