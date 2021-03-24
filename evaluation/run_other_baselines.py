from evaluation.helpers import PredictionManager
from src.algorithms import impute_missforest, impute_mice, impute_miceforest, predict_synthetic_intervention_ols, impute_als
from evaluation.helpers.evaluation_manager import compute_r2_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.plot_utils import boxplots

alg_names = {
    'alg=impute_mice': "MICE (sklearn IterativeImputer)",
    'alg=impute_missforest': "MissForest",
    'alg=impute_miceforest': "MiceForest",
    'alg=impute_als': "Tensor Decomposition",
    'alg=impute_als,rank=5': "Tensor Decomposition, Rank 5",
    'alg=impute_als,rank=100': "Tensor Decomposition, Rank 100",
    'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention': 'SI'
}

boxColors = ['#addd8e', '#31a354', '#7fcdbb', '#2c7fb8',
             '#feb24c', '#f03b20', '#c51b8a', '#756bb1']

name = 'level2'
average = True
num_perts = 100
num_genes = 15

pm = PredictionManager(
    num_cells=None,
    num_perts=num_perts,
    name=name,
    num_folds=None,
    average=average,
    num_genes=num_genes
)
# tensorly_predictions, tensorly_times = pm.predict(impute_als, overwrite=False, rank=100)
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

num_rows_per_alg = sum((len(ixs) for ixs in pm.fold_test_ixs))
num_rows = num_rows_per_alg * len(pm.prediction_filenames)
r2s = np.zeros(num_rows)

ix = 0
index = []
for alg, prediction_filename in pm.prediction_filenames.items():
    print(f'[EvaluationManager.r2] computing r2 for {alg}')
    predicted_df = pd.read_pickle(prediction_filename)
    for fold_ix, test_ixs in enumerate(pm.fold_test_ixs):
        test_df = pm.gene_expression_df.iloc[test_ixs]
        predicted_df_fold = predicted_df[predicted_df.index.get_level_values('fold') == fold_ix]
        r2s[ix:(ix + predicted_df_fold.shape[0])] = compute_r2_matrix(test_df.values, predicted_df_fold.values)

        units, ivs = predicted_df_fold.index.get_level_values('unit'), predicted_df_fold.index.get_level_values(
            'intervention')
        index.extend(list(zip(units, ivs, [fold_ix] * len(units), [alg] * len(units))))
        ix += predicted_df_fold.shape[0]

r2_df = pd.DataFrame(r2s, index=pd.MultiIndex.from_tuples(index, names=['unit', 'intervention', 'fold_ix', 'alg']))


algs = list(pm.prediction_filenames.keys())
print(algs)
r2_dict = {alg_names[alg]: r2_df.query('alg == @alg').values.flatten() for alg in algs}
plt.clf()
boxplots(
    r2_dict,
    boxColors,
    xlabel='Algorithm',
    ylabel='$R^2$ score per context/action pair',
    title=pm.result_string,
    top=1,
    bottom=.5,
    scale=.03
)
plt.savefig(f"visuals/figures/imputation_num_genes={num_genes}.png")
plt.show()

