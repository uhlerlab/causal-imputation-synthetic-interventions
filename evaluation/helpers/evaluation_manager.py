from evaluation.helpers.prediction_manager2 import PredictionManager
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from evaluation.plot_utils import boxplots
sns.set()


boxColors = ['#addd8e', '#31a354', '#7fcdbb', '#2c7fb8',
             '#feb24c', '#f03b20', '#c51b8a', '#756bb1']

alg_names = {
    'alg=impute_unit_mean': 'Mean in Unit',
    'alg=impute_intervention_mean': 'Mean in Intervention',
    'alg=impute_two_way_mean': '2-way Mean',
    'alg=predict_intervention_fixed_effect,control_intervention=DMSO': 'Fixed Effect',
    'alg=predict_synthetic_control_unit_ols,num_desired_interventions=None': 'SI',
    'alg=predict_synthetic_control_unit_hsvt_ols,num_desired_interventions=None,energy=0.95': 'SI+hsvt,.95',
    'alg=predict_synthetic_control_unit_hsvt_ols,num_desired_interventions=None,energy=0.99': 'SI+hsvt,.99',
    'alg=predict_synthetic_control_unit_hsvt_ols,num_desired_interventions=None,energy=0.999': 'SI+hsvt,.999',
    # 'alg=predict_synthetic_control_unit_hsvt_ols,num_desired_interventions=None,progress=False,energy=0.8': 'SI+hsvt,.8',
}


def compute_r2_matrix(true_values, predicted_values):
    """Faster implementation of sklearn.metrics.r2_score for matrices"""
    true_means = true_values.mean(axis=1)
    baseline_error = np.sum((true_values - true_means[:, None])**2, axis=1)
    true_error = np.sum((true_values - predicted_values)**2, axis=1)
    return 1 - true_error/baseline_error


def compute_r2_vector(true_values, predicted_values):
    true_mean = true_values.mean()
    baseline_error = np.sum((true_values - true_mean)**2)
    true_error = np.sum((true_values - predicted_values)**2)
    return 1 - true_error/baseline_error


class EvaluationManager:
    def __init__(self, prediction_manager: PredictionManager):
        self.prediction_manager = prediction_manager

    def r2(self):
        num_rows_per_alg = sum((len(ixs) for ixs in self.prediction_manager.fold_test_ixs))
        num_rows = num_rows_per_alg * len(self.prediction_manager.prediction_filenames)
        r2s = np.zeros(num_rows)
        index = []

        ix = 0
        for alg, prediction_filename in self.prediction_manager.prediction_filenames.items():
            print(f'[EvaluationManager.r2] computing r2 for {alg}')
            predicted_df = pd.read_pickle(prediction_filename)
            for fold_ix, test_ixs in enumerate(self.prediction_manager.fold_test_ixs):
                # get test data that was held out in this fold
                test_df = self.prediction_manager.gene_expression_df.iloc[test_ixs]
                predicted_df_fold = predicted_df[predicted_df.index.get_level_values('fold') == fold_ix]
                predicted_df_fold = predicted_df_fold.reset_index('fold', drop=True)
                assert (test_df.index == predicted_df_fold.index).all()

                # compute the R2 score for each gene expression profile
                r2s[ix:(ix+predicted_df_fold.shape[0])] = compute_r2_matrix(test_df.values, predicted_df_fold.values)
                units, ivs = predicted_df_fold.index.get_level_values('unit'), predicted_df_fold.index.get_level_values('intervention')
                index.extend(list(zip(units, ivs, [fold_ix]*len(units), [alg]*len(units))))
                ix += predicted_df_fold.shape[0]

        res = pd.DataFrame(r2s, index=pd.MultiIndex.from_tuples(index, names=['unit', 'intervention', 'fold_ix', 'alg']))
        return res

    def r2_per_iv(self):
        r2s = []
        index = []
        for alg, prediction_filename in self.prediction_manager.prediction_filenames.items():
            print(f'[EvaluationManager.r2_in_iv] computing r2 for {alg}')
            predicted_df = pd.read_pickle(prediction_filename)
            for fold_ix, test_ixs in enumerate(self.prediction_manager.fold_test_ixs):
                test_df = self.prediction_manager.gene_expression_df.iloc[test_ixs]
                predicted_df_fold = predicted_df[predicted_df.index.get_level_values('fold') == fold_ix]

                iv_ix_dict = test_df.groupby('intervention').indices
                ivs = list(iv_ix_dict.keys())
                test_values = test_df.values
                predicted_values = predicted_df_fold.values
                r = [
                    compute_r2_vector(test_values[iv_ix].flatten(), predicted_values[iv_ix].flatten())
                    for iv, iv_ix in iv_ix_dict.items()
                ]
                r2s.extend(r)
                index.extend(list(zip(ivs, [fold_ix]*len(ivs), [alg]*len(ivs))))

        res = pd.DataFrame(r2s, index=pd.MultiIndex.from_tuples(index, names=['intervention', 'fold_ix', 'alg']))
        return res

    def boxplot(self):
        r2_df = self.r2()
        algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: r2_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='$R^2$ score per (cell type, intervention) pair',
            title='Estimated Gene Expression',
            top=1,
            bottom=.5,
            scale=.03
        )
        os.makedirs('evaluation/plots', exist_ok=True)
        plt.savefig(f'evaluation/plots/boxplot_{self.prediction_manager.result_string}.png')

    def boxplot_per_intervention(self):
        r2_df = self.r2_per_iv()
        algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: r2_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='$R^2$ score per (cell type, intervention) pair',
            title=self.prediction_manager.result_string,
            top=1,
            bottom=.5,
            scale=.03
        )
        os.makedirs('evaluation/plots', exist_ok=True)
        plt.savefig(f'evaluation/plots/boxplot_by_iv_{self.prediction_manager.result_string}.png')
        plt.savefig(f'evaluation/plots/boxplot_by_iv_{self.prediction_manager.result_string}.png')
