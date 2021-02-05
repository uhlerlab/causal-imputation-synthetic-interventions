import matplotlib.pyplot as plt
import os
# import seaborn as sns
import pandas as pd
import numpy as np
from evaluation.helpers import EvaluationManager, PredictionManager
from src.algorithms import predict_synthetic_intervention_ols
from sklearn.linear_model import LinearRegression

# sns.set()
plt.style.use('ggplot')

os.makedirs('evaluation/statistics_histograms/', exist_ok=True)

alg = 'predict_synthetic_intervention_ols'
num_donors = None
cell_start = None
pert_start = None
num_cells = None
num_perts = 100
name = 'level2'
average = True
num_folds = None

pm = PredictionManager(
    cell_start=None,
    pert_start=None,
    num_cells=None,
    num_perts=100,
    name='level2',
    average=True,
    num_folds=None,
)
pm.predict(
    predict_synthetic_intervention_ols,
    num_desired_donors=None,
    donor_dim='intervention',
    progress=False,
    overwrite=False,
)
pm.predict(
    predict_synthetic_intervention_ols,
    num_desired_donors=None,
    donor_dim='unit',
    progress=False,
    overwrite=False,
)
evaluation_manager = EvaluationManager(pm)
r2 = evaluation_manager.r2()

folder = f'cell={cell_start},{num_cells}cells,pert={pert_start},{num_perts}perts,name={name},num_folds={num_folds},average={average}'
print(folder)
unit_filename = f'alg={alg},num_desired_donors={num_donors},donor_dim=unit_stats.pkl'
iv_filename = f'alg={alg},num_desired_donors={num_donors},donor_dim=intervention_stats.pkl'
statistics_unit = pd.read_pickle(os.path.join('evaluation/results', folder, unit_filename))
statistics_iv = pd.read_pickle(os.path.join('evaluation/results', folder, iv_filename))

plt.clf()
plt.scatter(statistics_unit["num_donors"], statistics_unit["num_training"])
plt.xlabel("Number of Donor Units")
plt.ylabel("Number of Training Actions")
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("visuals/figures/donor-context-training-action.png")

plt.clf()
plt.scatter(statistics_iv["num_donors"], statistics_iv["num_training"])
plt.xlabel("Number of Donor Actions")
plt.ylabel("Number of Training Contexts")
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("visuals/figures/donor-action-training-context.png")

iv_r2 = r2.query("alg == 'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention'")
unit_r2 = r2.query("alg == 'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=unit'")

lr = LinearRegression()
lr.fit(statistics_unit["num_donors"].values.reshape(-1, 1), unit_r2.values)
print(lr.coef_)
plt.clf()
plt.scatter(statistics_unit["num_donors"], unit_r2.values)
plt.xlabel("Number of Donor Contexts")
plt.ylabel("R^2")
cb = plt.colorbar()
cb.set_label("Number of Training Actions")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("visuals/figures/donor-context-r2.png")

lr = LinearRegression()
positive_ixs = (iv_r2.values >= 0).flatten()
lr.fit(statistics_iv["num_donors"].values.reshape(-1, 1)[positive_ixs, :], iv_r2.values[positive_ixs])
print(np.sum(iv_r2.values < 0))
print(lr.coef_)
plt.clf()
plt.scatter(statistics_iv["num_donors"], iv_r2.values, c=statistics_iv["num_training"])
plt.xlabel("Number of Donor Actions")
plt.ylabel("$R^2$")
cb = plt.colorbar()
cb.set_label("Number of Training Contexts")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("visuals/figures/donor-action-r2.png")

# print(statistics_unit.isna().sum())
# print(statistics_iv.isna().sum())
# statistics_unit = statistics_unit[~statistics_unit.isna()]
# statistics_iv = statistics_iv[~statistics_iv.isna()]
