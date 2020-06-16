"""
Plot the spectral energy pattern of the matrix consisting of control data (DMSO) for all cell types.
"""
import random
from processing.most_common_manager import MostCommonManager
from processing.average_manager import AverageManager
import os
import matplotlib.pyplot as plt
from numpy.linalg import svd
from src.matrix import spectral_energy
import seaborn as sns
sns.set()

random.seed(188181)

mc_manager = MostCommonManager(None, 0)
for log2 in [True, False]:
    avg_manager = AverageManager(
        'control',
        mc_manager.get_most_common_gctx(),
        log2=log2,
        minmax=False
    )

    avgs = avg_manager.get_space2average_df()['original']
    dmso_mat = avgs[avgs.index.get_level_values('intervention') == 'DMSO']
    dmso_mat = dmso_mat - dmso_mat.mean()
    u, s, v = svd(dmso_mat)
    energy = spectral_energy(s)

    plt.clf()
    plt.plot(energy)
    plt.ylabel('Spectral Energy')
    plt.xlabel('Number of singular vectors')
    plt.savefig(os.path.join('exploration', 'figures', f'spectral_energy_control_log2={log2}.png'))
