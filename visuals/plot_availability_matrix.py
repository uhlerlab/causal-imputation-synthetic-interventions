import numpy as np
from numpy.ma import masked_array
import matplotlib.pyplot as plt


def plot_availability_matrix(df, ytick_space=10, savefig=None):
    counts = df.groupby(['intervention', 'unit']).size()
    sorted_perturbations = counts.groupby('intervention').size().sort_values(ascending=True)
    sorted_celltypes = counts.groupby('unit').size().sort_values(ascending=False)

    count_matrix = np.zeros((len(sorted_perturbations), len(sorted_celltypes)))

    pert2ix = {pert: ix for ix, pert in enumerate(sorted_perturbations.index)}
    cell2ix = {cell: ix for ix, cell in enumerate(sorted_celltypes.index)}
    pert_ixs, cell_ixs = counts.index.get_level_values('intervention').map(pert2ix), counts.index.get_level_values('unit').map(cell2ix)
    count_matrix[pert_ixs, cell_ixs] = counts.values
    # masked_count_matrix = masked_array(count_matrix, count_matrix==0)

    plt.imshow(count_matrix != 0, aspect='auto', interpolation='none', cmap='binary')
    plt.xlabel("Cell Types")
    plt.ylabel("Perturbation IDs")
    plt.xticks(list(range(len(sorted_celltypes))))
    plt.yticks(list(reversed(range(len(sorted_perturbations), 0, -ytick_space))), list(reversed(range(0, len(sorted_perturbations), ytick_space))))
    ax = plt.gca()
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize='x-small')
    ax.set_xticklabels([str(ix+1)+":"+ct for ix, ct in enumerate(sorted_celltypes.index)], ha='right', rotation=70)
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(f"evaluation/availability/{savefig}.pdf", dpi=200)
