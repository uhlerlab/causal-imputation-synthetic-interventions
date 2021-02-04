from evaluation.helpers.prediction_manager import PredictionManager
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

energy = 0.95
alg_names = {
    'alg=impute_unit_mean': 'Mean over Actions',
    'alg=impute_intervention_mean': 'Mean over Contexts',
    'alg=impute_two_way_mean': '2-way Mean',
    'alg=predict_intervention_fixed_effect,control_intervention=DMSO': 'Fixed Effect',
    f'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=unit': f'SI-Context',
    f'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention': f'SI-Action',
    f'alg=predict_synthetic_intervention_hsvt_ols,num_desired_donors=None,energy=0.95,hypo_test=False,donor_dim=intervention': f'SI-action-HSVT',
    f'alg=predict_synthetic_intervention_hsvt_ols,num_desired_donors=None,energy=0.95,hypo_test=True,hypo_test_percent=0.1,donor_dim=intervention,equal_rank=True': f'SI-action-HSVT, +test',
}


def compute_r2_matrix(true_values, predicted_values):
    """Faster implementation of sklearn.metrics.r2_score for matrices"""
    true_means = true_values.mean(axis=1)
    baseline_error = np.sum((true_values - true_means[:, None])**2, axis=1)
    true_error = np.sum((true_values - predicted_values)**2, axis=1)
    return 1 - true_error/baseline_error


def compute_rmse_matrix(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values)**2))


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

                if alg == 'alg=predict_synthetic_intervention_ols,num_desired_interventions=None,donor_dim=unit':
                    ipdb.set_trace()

                # compute the R2 score for each gene expression profile
                r2s[ix:(ix+predicted_df_fold.shape[0])] = compute_r2_matrix(test_df.values, predicted_df_fold.values)
                units, ivs = predicted_df_fold.index.get_level_values('unit'), predicted_df_fold.index.get_level_values('intervention')
                index.extend(list(zip(units, ivs, [fold_ix]*len(units), [alg]*len(units))))
                ix += predicted_df_fold.shape[0]

        res = pd.DataFrame(r2s, index=pd.MultiIndex.from_tuples(index, names=['unit', 'intervention', 'fold_ix', 'alg']))
        return res

    def plot_times(self):
        alg2times = dict()
        algs = list(self.prediction_manager.time_filenames.keys())

        for alg in algs:
            times = np.loadtxt(self.prediction_manager.time_filenames[alg])
            alg2times[alg_names[alg]] = np.log(times)
        boxplots(
            alg2times,
            boxColors,
            xlabel='Algorithm',
            ylabel='Log (# of seconds) per prediction.',
            title=self.prediction_manager.result_string,
            top=-3,
            bottom=-6,
        )
        os.makedirs('evaluation/plots', exist_ok=True)
        filename = f'evaluation/plots/time_boxplot_{self.prediction_manager.result_string}.png'
        plt.savefig(filename)
        plt.title("")
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/causal-imputation-time-{self.prediction_manager.result_string}.png'))
        print(f"Saved to {os.path.abspath(filename)}")

    def rmse(self):
        num_rows_per_alg = sum((len(ixs) for ixs in self.prediction_manager.fold_test_ixs))
        num_rows = num_rows_per_alg * len(self.prediction_manager.prediction_filenames)
        rmses = np.zeros(num_rows)
        index = []

        ix = 0
        algs = list(self.prediction_manager.prediction_filenames.keys())

        for alg in algs:
            print(f'[EvaluationManager.rmse] computing rmse for {alg}')
            predicted_df = pd.read_pickle(self.prediction_manager.prediction_filenames[alg])
            for fold_ix, test_ixs in enumerate(self.prediction_manager.fold_test_ixs):
                # get test data that was held out in this fold
                test_df = self.prediction_manager.gene_expression_df.iloc[test_ixs]
                predicted_df_fold = predicted_df[predicted_df.index.get_level_values('fold') == fold_ix]
                predicted_df_fold = predicted_df_fold.reset_index('fold', drop=True)
                assert (test_df.index == predicted_df_fold.index).all()

                if alg == 'alg=predict_synthetic_intervention_ols,num_desired_interventions=None,donor_dim=unit':
                    ipdb.set_trace()

                # compute the R2 score for each gene expression profile
                rmses[ix:(ix + predicted_df_fold.shape[0])] = compute_rmse_matrix(test_df.values, predicted_df_fold.values)
                units, ivs = predicted_df_fold.index.get_level_values('unit'), predicted_df_fold.index.get_level_values(
                    'intervention')
                index.extend(list(zip(units, ivs, [fold_ix] * len(units), [alg] * len(units))))
                ix += predicted_df_fold.shape[0]

        res = pd.DataFrame(
            rmses,
            index=pd.MultiIndex.from_tuples(index, names=['unit', 'intervention', 'fold_ix', 'alg'])
        )
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
        algs = list(set(r2_df.index.get_level_values('alg')))
        # algs = list(alg_names.keys())
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
        filename = f'evaluation/plots/boxplot_{self.prediction_manager.result_string}.png'
        plt.savefig(filename)
        plt.title("")
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/causal-imputation-r2-{self.prediction_manager.result_string}.png'))
        print(f"Saved to {os.path.abspath(filename)}")

    def boxplot_rmse(self):
        rmse_df = self.rmse()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        print(algs)
        # algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: rmse_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='RMSE per (cell type, intervention) pair',
            bottom=0,
            top=800,
            title=self.prediction_manager.result_string,
        )
        os.makedirs('evaluation/plots', exist_ok=True)
        filename = f'evaluation/plots/rmse_boxplot_{self.prediction_manager.result_string}.png'
        plt.savefig(filename)
        plt.title("")
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/rmse_causal-imputation-r2-{self.prediction_manager.result_string}.png'))
        print(f"Saved to {os.path.abspath(filename)}")

    def boxplot_per_intervention(self):
        r2_df = self.r2_per_iv()
        algs = list(set(r2_df.index.get_level_values('alg')))
        r2_dict = {alg_names[alg]: r2_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='$R^2$ score per (cell type, intervention) pair',
            title=self.prediction_manager.result_string,
            top=1.1,
            bottom=0,
            scale=.08
        )
        os.makedirs('evaluation/plots', exist_ok=True)
        filename = f'evaluation/plots/boxplot_by_iv_{self.prediction_manager.result_string}.png'
        plt.savefig(filename)
        plt.title("")
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/causal-imputation-r2-per-iv-{self.prediction_manager.result_string}.png'))
        print(f"Saved to {os.path.abspath(filename)}")

    def statistic_vs_best(self):
        s = 'alg=predict_synthetic_intervention_hsvt_ols,num_desired_donors=None,energy=0.95,hypo_test=True,hypo_test_percent=0.1,donor_dim=intervention,equal_rank=True'
        stats_filename = self.prediction_manager.statistic_filenames[s]
        stats = pd.read_pickle(stats_filename)
        r2s = self.r2()
        r2s_best = r2s.groupby('fold_ix').max()

        stats = stats.values[:, 0] + np.random.normal(0, .05, size=len(stats))

        plt.clf()
        plt.scatter(stats, r2s_best.values)
        plt.xlabel("Statistic")
        plt.ylabel("Best R^2")
        plt.ylim([0, 1])
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/statistic-vs-best_{self.prediction_manager.result_string}.png'))

        plt.clf()
        r2s_mean = r2s[r2s.index.get_level_values('alg') == 'alg=impute_unit_mean']
        plt.scatter(stats, r2s_mean.values)
        plt.xlabel("Statistic")
        plt.ylabel("R^2 of mean-over-actions")
        plt.ylim([0, 1])
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/statistic-vs-mean-r2_{self.prediction_manager.result_string}.png'))

        plt.clf()
        r2s_si_hsvt = r2s[r2s.index.get_level_values('alg') == 'alg=predict_synthetic_intervention_hsvt_ols,num_desired_donors=None,energy=0.95,hypo_test=True,hypo_test_percent=0.1,donor_dim=intervention,equal_rank=True']
        plt.scatter(stats, r2s_si_hsvt.values)
        plt.xlabel("Statistic")
        plt.ylabel("R^2 of SI-action")
        plt.ylim([0, 1])
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/statistic-vs-si-hsvt-r2_{self.prediction_manager.result_string}.png'))

        plt.clf()
        r2s_si = r2s[r2s.index.get_level_values('alg') == 'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention']
        plt.scatter(stats, r2s_si.values)
        plt.xlabel("Statistic")
        plt.ylabel("R^2 of SI-action")
        plt.ylim([0, 1])
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/statistic-vs-si-r2_{self.prediction_manager.result_string}.png'))

        nan_ixs = np.isnan(stats)
        stats = stats[~nan_ixs]
        r2s_best = r2s_best.values[~nan_ixs].flatten()
        r2s_mean = r2s_mean.values[~nan_ixs].flatten()
        r2s_si_hsvt = r2s_si_hsvt.values[~nan_ixs].flatten()
        r2s_si = r2s_si.values[~nan_ixs].flatten()

        print(np.polyfit(stats[r2s_best > 0], r2s_best[r2s_best > 0], 1))
        print(np.polyfit(stats[r2s_mean > 0], r2s_mean[r2s_mean > 0], 1))
        print(np.polyfit(stats[r2s_si > 0], r2s_si[r2s_si > 0], 1))
        print(np.polyfit(stats[r2s_si_hsvt > 0], r2s_si_hsvt[r2s_si_hsvt > 0], 1))

        ipdb.set_trace()
