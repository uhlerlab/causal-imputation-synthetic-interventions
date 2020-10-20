from visuals.plot_availability_matrix import plot_availability_matrix
import matplotlib.pyplot as plt
import numpy as np
from filenames import load_cmap_original

df = load_cmap_original()

plt.clf()
plt.grid(False)
plot_availability_matrix(df)
plt.savefig('visuals/figures/availability.png')

counts = df.groupby(['intervention', 'unit']).size()
sorted_perturbations = counts.groupby('intervention').size().sort_values(ascending=True)
sorted_celltypes = counts.groupby('unit').size().sort_values(ascending=False)
count_matrix = np.zeros((len(sorted_perturbations), len(sorted_celltypes)))


import seaborn as sns
sns.set()

plt.figure(figsize=(10, 10))
plt.scatter(list(range(len(sorted_celltypes))), list(sorted_celltypes.values))
plt.ylabel('Number of Perturbations')
plt.xlabel('Cell Type')
plt.xticks(list(range(len(sorted_celltypes))))
ax = plt.gca()
ax.tick_params(axis='x', bottom=False, top=False, labelsize='small')
ax.set_xticklabels(sorted_celltypes.index, ha='right', rotation=70)
plt.tight_layout()
plt.savefig('visuals/figures/sorted_celltypes.png')

sns.set()
plt.figure(figsize=(10, 10))
plt.scatter(list(range(len(sorted_perturbations))), list(reversed(sorted_perturbations.values)))
plt.ylabel('Number of Cell Types')
plt.xlabel('Perturbation')
plt.tight_layout()
plt.savefig('visuals/figures/sorted_perturbations.png')
