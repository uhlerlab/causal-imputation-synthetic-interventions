import matplotlib.pyplot as plt
import os
# import seaborn as sns
import pandas as pd
import numpy as np
# sns.set()
plt.style.use('ggplot')

os.makedirs('evaluation/statistics_histograms/', exist_ok=True)

alg = 'predict_synthetic_intervention_hsvt_ols'
num_donors = None
energy = .95
hypo_percent = .1
cell_start = None
pert_start = None
num_cells = None
num_perts = 100
name = 'level2'
average = True
num_folds = None

folder = f'cell={cell_start},{num_cells}cells,pert={pert_start},{num_perts}perts,name={name},num_folds={num_folds},average={average}'
unit_filename = f'alg={alg},num_desired_donors={num_donors},energy={energy},hypo_test=True,hypo_test_percent={hypo_percent},donor_dim=unit_stats.pkl'
iv_filename = f'alg={alg},num_desired_donors={num_donors},energy={energy},hypo_test=True,hypo_test_percent={hypo_percent},donor_dim=intervention_stats.pkl'
statistics_unit = pd.read_pickle(os.path.join('evaluation/results', folder, unit_filename))
statistics_iv = pd.read_pickle(os.path.join('evaluation/results', folder, iv_filename))
print(statistics_unit.isna().sum())
print(statistics_iv.isna().sum())
statistics_unit = statistics_unit[~statistics_unit.isna()]
statistics_iv = statistics_iv[~statistics_iv.isna()]

bins = np.linspace(0, 3, 11)
hist_unit, _ = np.histogram(statistics_unit, bins=bins)
hist_iv, _ = np.histogram(statistics_iv, bins=bins)
plt.clf()
plt.bar(bins[:-1], hist_unit/hist_unit.sum(), label='SI-Context', alpha=.5, width=3/10)
plt.bar(bins[:-1], hist_iv/hist_iv.sum(), label='SI-Action', alpha=.5, width=3/10)
plt.xlabel('Statistic')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('evaluation/statistics_histograms/statistic_histogram.png')
