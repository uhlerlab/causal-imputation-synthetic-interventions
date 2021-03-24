from evaluation.helpers import PredictionManager
from src.algorithms import impute_missforest, impute_mice, impute_miceforest, predict_synthetic_intervention_ols, impute_als
import numpy as np
import matplotlib.pyplot as plt

name = 'level2'
average = True
num_perts = 100

avg_tensorly_times5 = []
avg_tensorly_times10 = []
avg_mice_times = []
avg_mf_times = []
avg_micef_times = []
avg_si_times = []
num_genes_list = [5, 10, 15]
for num_genes in num_genes_list:
    pm = PredictionManager(
        num_cells=None,
        num_perts=num_perts,
        name=name,
        num_folds=None,
        average=average,
        num_genes=num_genes
    )

    tensorly_predictions, tensorly_times5 = pm.predict(impute_als, overwrite=False, rank=5)
    tensorly_predictions, tensorly_times10 = pm.predict(impute_als, overwrite=False, rank=10)
    mice_predictions, mice_times = pm.predict(impute_mice, overwrite=False)
    mf_predictions, mf_times = pm.predict(impute_missforest, overwrite=False)
    micef_predictions, micef_times = pm.predict(impute_miceforest, overwrite=False)
    si_predictions, si_times = pm.predict(
        predict_synthetic_intervention_ols,
        num_desired_donors=None,
        donor_dim='intervention',
        progress=False,
        overwrite=False,
    )
    avg_tensorly_times5.append(np.mean(tensorly_times5))
    avg_tensorly_times10.append(np.mean(tensorly_times10))
    avg_mice_times.append(np.mean(mice_times))
    avg_mf_times.append(np.mean(mf_times))
    avg_micef_times.append(np.mean(micef_times))
    avg_si_times.append(np.mean(si_times))


plt.yscale("log")
plt.plot(num_genes_list, avg_tensorly_times5, label="Tensor (5)")
plt.plot(num_genes_list, avg_tensorly_times10, label="Tensor (10)")
plt.plot(num_genes_list, avg_mice_times, label="MICE (sklearn)")
plt.plot(num_genes_list, avg_mf_times, label="MissForest")
plt.plot(num_genes_list, avg_micef_times, label="MICE")
plt.plot(num_genes_list, avg_si_times, label="SI")
plt.xlabel("Number of genes (p)")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("visuals/figures/times-imputations.png")
