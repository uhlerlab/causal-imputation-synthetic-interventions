from evaluation.helpers.get_data_block import get_data_block
import numpy as np
from src.algorithms import predict_synthetic_intervention_hsvt_ols, impute_unit_mean
from evaluation.helpers.evaluation_manager import compute_r2_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

num_cells = None
num_perts = 100

data, _, _, _ = get_data_block(
    num_cells=num_cells,
    num_perts=num_perts,
    name='level2_filtered',
    average=False
)


hsvt_predictions = np.zeros([data.shape[0], data.shape[1]])
mean_predictions = np.zeros([data.shape[0], data.shape[1]])
stats = np.zeros([data.shape[0], 2])
for ix in range(data.shape[0]):
    other_data = data.iloc[[j for j in range(data.shape[0]) if ix != j]]
    target = data.iloc[[ix]].index
    prediction, stat = predict_synthetic_intervention_hsvt_ols(
        other_data,
        target,
        None,
        energy=.99,
        hypo_test_percent=.2,
        donor_dim='intervention',
        center=True,
        equal_rank=True
    )
    hsvt_predictions[ix] = prediction
    mean_predictions[ix] = impute_unit_mean(other_data, target)
    stats[ix] = stat

print('===============================')
pass_ixs = stats[:, 0] < stats[:, 1]
print(f"Number passing: {sum(pass_ixs)}")
pass_data = data.iloc[pass_ixs]
hsvt_pass_predictions = hsvt_predictions[pass_ixs]
mean_pass_predictions = mean_predictions[pass_ixs]

mean_r2s_pass = compute_r2_matrix(pass_data.values, mean_pass_predictions)
hsvt_r2s_pass = compute_r2_matrix(pass_data.values, hsvt_pass_predictions)

mean_r2s = compute_r2_matrix(data.values, mean_predictions)
hsvt_r2s = compute_r2_matrix(data.values, hsvt_predictions)

print(f"(pass) median r2 for unit mean: {np.median(mean_r2s_pass)}")
print(f"(pass) median r2 for hsvt: {np.median(hsvt_r2s_pass)}")
print(f"(pass) mean is better: {np.sum(mean_r2s_pass > hsvt_r2s_pass)}")

print(f"(all) median r2 for unit mean: {np.median(mean_r2s)}")
print(f"(all) median r2 for hsvt: {np.median(hsvt_r2s)}")
print(f"(all) mean is better: {np.sum(mean_r2s > hsvt_r2s)}")

plt.clf()
colormap = plt.get_cmap('seismic')
plt.scatter(stats[:, 0], hsvt_r2s, c=pass_ixs, cmap=colormap)
plt.ylim([0, 1])
plt.xlabel('Statistic')
plt.ylabel('R^2')
plt.ion()
plt.show()
