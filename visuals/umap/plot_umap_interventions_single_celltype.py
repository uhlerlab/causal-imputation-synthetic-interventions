import matplotlib.pyplot as plt
from filenames import load_pert_info, load_cmap_original, load_cmap_filtered
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from utils import pandas_minmax
import umap
from matplotlib.patches import Patch, Circle
import itertools as itr
import random
from matplotlib.lines import Line2D
sns.set()
# sns.set_style('white')

random.seed(123124849)

DO_PCA = False
if DO_PCA:
    reducer = PCA(n_components=2)
else:
    reducer = umap.UMAP()

colormap = plt.get_cmap('tab20')
colormap = list(itr.chain.from_iterable(itr.repeat(colormap.colors, 4)))

data = load_cmap_filtered()
data = data[data.index.get_level_values("cell_id") == 'VCAP']
num_samples_per_pert = data.groupby("pert_id").size()
highly_sampled_perts = num_samples_per_pert[num_samples_per_pert > 20]
perts = random.sample(list(highly_sampled_perts.index), 70) + ["DMSO"]
data = data[data.index.get_level_values("pert_id").isin(set(perts))]
data = data.groupby("pert_id").head(100)

LOG2 = True
MINMAX = True
if LOG2:
    print("[visuals/umap/plot_umap_interventions_single_celltype] log2")
    data = np.log2(data+1)
if MINMAX:
    print("[visuals/umap/plot_umap_interventions_single_celltype] minmax")
    data = pandas_minmax(data, axis=1)
print("[visuals/umap/plot_umap_interventions_single_celltype] Embedding")
embedded_data = reducer.fit_transform(data)

control_ixs = data.index.get_level_values('pert_id') == 'DMSO'
control_data = data[control_ixs]
pert_data = data[~control_ixs]
control_embedded_data = embedded_data[control_ixs]
pert_embedded_data = embedded_data[~control_ixs]

print("[visuals/umap/plot_umap_interventions_single_celltype] Plotting, colored by cell type")
plt.clf()
pert2ix = {pert: ix for ix, pert in enumerate(perts[:-1])}
colors = pert_data.index.get_level_values('pert_id').map(pert2ix)
colors = [colormap[c] for c in colors]
plt.scatter(control_embedded_data[:, 0], control_embedded_data[:, 1], s=100, color='k')
plt.scatter(pert_embedded_data[:, 0], pert_embedded_data[:, 1], c=colors, s=10, alpha=.5)
lgd = plt.legend(
    handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap[ix], label=pert, alpha=.5) for pert, ix in pert2ix.items()
    ] + [Line2D([0], [0], marker='o', markersize=15, color='w', markerfacecolor='k', label='DMSO')],
    ncol=4,
    bbox_to_anchor=(1.05, 1),
    fontsize='small'
)
plt.savefig('visuals/figures/umap_pert_coloring_vcap.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

