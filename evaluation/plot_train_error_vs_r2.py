import matplotlib.pyplot as plt
from evaluation.helpers import PredictionManager, EvaluationManager
import os
from src.algorithms import impute_unit_mean, impute_intervention_mean, impute_two_way_mean
from src.algorithms.synthetic_interventions2 import predict_synthetic_intervention_ols
# import seaborn as sns
import pandas as pd
import numpy as np
# sns.set()
plt.style.use('ggplot')

os.makedirs('evaluation/train_error_vs_r2/', exist_ok=True)
os.makedirs('evaluation/proj_stat_vs_r2/', exist_ok=True)

alg = 'predict_synthetic_intervention_ols'
num_donors = None
cell_start = None
pert_start = None
num_cells = None
num_perts = 100
name = 'level2'
average = True
num_folds = None

folder = f'cell={cell_start},{num_cells}cells,pert={pert_start},{num_perts}perts,name={name},num_folds={num_folds},average={average}'
unit_filename = f'alg={alg},num_desired_donors={num_donors},donor_dim=unit_stats.pkl'
iv_filename = f'alg={alg},num_desired_donors={num_donors},donor_dim=intervention_stats.pkl'
statistics_unit = pd.read_pickle(os.path.join('evaluation/results', folder, unit_filename))
statistics_iv = pd.read_pickle(os.path.join('evaluation/results', folder, iv_filename))
print(statistics_unit.isna().sum())
print(statistics_iv.isna().sum())
statistics_unit = statistics_unit[~statistics_unit.isna()]
statistics_iv = statistics_iv[~statistics_iv.isna()]

pm = PredictionManager(
    cell_start=cell_start,
    num_cells=num_cells,
    num_perts=num_perts,
    name=name,
    num_folds=None,
    average=average
)
pm.predict(impute_unit_mean, overwrite=False)
pm.predict(
    predict_synthetic_intervention_ols,
    num_desired_donors=None,
    donor_dim='intervention',
    progress=False,
    overwrite=False,
)
em = EvaluationManager(pm)
relative_mse_df = em.relative_mse()
rmse_df = em.rmse()
alg = 'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention'
relative_mses = relative_mse_df[relative_mse_df.index.get_level_values("alg") == alg]
rmses = rmse_df[rmse_df.index.get_level_values("alg") == alg]
statistics_iv["relative_mse"] = relative_mses.values
statistics_iv["rmse"] = rmses.values

plt.clf()
plt.scatter(statistics_iv["train_error"], relative_mses.values)
plt.xlabel("Training error per entry")
plt.ylabel("Relative MSE")
plt.yscale("log")
plt.xscale("log")
plt.savefig("evaluation/train_error_vs_r2/relative_mse.png")

plt.clf()
plt.scatter(statistics_iv["stat"], relative_mses.values, c=statistics_iv["train_error"], s=4)
plt.xlabel("Projection statistic")
plt.ylabel("Relative MSE")
plt.yscale("log")
# plt.xscale("log")
plt.savefig("evaluation/proj_stat_vs_r2/relative_mse.png")

plt.clf()
plt.scatter(statistics_iv["stat"], rmses.values)
plt.xlabel("Projection statistic")
plt.ylabel("RMSE")
plt.yscale("log")
# plt.xscale("log")
plt.savefig("evaluation/proj_stat_vs_r2/rmse.png")

plt.clf()
plt.scatter(statistics_iv["train_error"], rmses.values)
plt.xlabel("Train error")
plt.ylabel("RMSE")
plt.yscale("log")
# plt.xscale("log")
plt.savefig("evaluation/train_error_vs_r2/rmse.png")

print(statistics_unit["stat"].median())
print(statistics_iv["stat"].median())

# bins = np.linspace(0, 3, 11)
# hist_unit, _ = np.histogram(statistics_unit, bins=bins)
# hist_iv, _ = np.histogram(statistics_iv, bins=bins)
# plt.clf()
# plt.bar(bins[:-1], hist_unit/hist_unit.sum(), label='SI-Context', alpha=.5, width=3/10)
# plt.bar(bins[:-1], hist_iv/hist_iv.sum(), label='SI-Action', alpha=.5, width=3/10)
# plt.xlabel('Statistic')
# plt.ylabel('Frequency')
# plt.legend()
# plt.savefig('evaluation/statistics_histograms/statistic_histogram.png')
