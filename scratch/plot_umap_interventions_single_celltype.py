import matplotlib.pyplot as plt
from filenames import load_pert_info, load_cmap_original
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from utils import pandas_minmax
import umap
from matplotlib.patches import Patch
import itertools as itr
import random
sns.set()

DO_PCA = False
if DO_PCA:
    reducer = PCA(n_components=2)
else:
    reducer = umap.UMAP()

colormap = plt.get_cmap('tab20')
colormap = list(itr.chain.from_iterable(itr.repeat(colormap.colors, 4)))

data = load_cmap_original()
data = data[data.index.get_level_values("cell_id") == 'VCAP']
num_samples_per_pert = data.groupby("pert_id").size()
highly_sampled_perts = num_samples_per_pert[num_samples_per_pert > 20]
perts = random.sample(list(highly_sampled_perts.index), 70)
data = data[data.index.get_level_values("pert_id").isin(set(perts))]
data = data.groupby("pert_id").head(100)

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

print("Plotting, colored by cell type")
plt.clf()
pert2ix = {pert: ix for ix, pert in enumerate(perts)}
colors = data.index.get_level_values('pert_id').map(pert2ix)
colors = [colormap[c] for c in colors]
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=colors)
lgd = plt.legend(
    handles=[
        Patch(color=colormap[ix], label=pert) for pert, ix in pert2ix.items()
    ],
    ncol=4,
    bbox_to_anchor=(1.05, 1),
    fontsize='small'
)
plt.savefig('scratch/umap_pert_coloring_vcap.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

