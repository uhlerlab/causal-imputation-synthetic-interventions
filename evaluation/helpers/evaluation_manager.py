from evaluation.helpers.prediction_manager import PredictionManager
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from scipy.stats import iqr
from evaluation.plot_utils import boxplots
sns.set()


boxColors = ['#addd8e', '#31a354', '#7fcdbb', '#2c7fb8',
             '#feb24c', '#f03b20', '#c51b8a', '#756bb1']

energy = 0.95
alg_names = {
    'alg=impute_intervention_mean': 'Mean over Contexts',
    f'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=unit': f'SI-Context',
    'alg=impute_two_way_mean': '2-way Mean',
    'alg=impute_unit_mean': 'Mean over Actions',
    "alg=impute_mice": "MICE",
    'alg=predict_intervention_fixed_effect,control_intervention=DMSO': 'Fixed Effect',
    f'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention': f'SI-Action',
    f'alg=predict_synthetic_intervention_hsvt_ols,num_desired_donors=None,energy=0.95,hypo_test=False,donor_dim=intervention': f'SI-action-HSVT',
    f'alg=predict_synthetic_intervention_hsvt_ols,num_desired_donors=None,energy=0.95,hypo_test=True,hypo_test_percent=0.1,donor_dim=intervention,equal_rank=True': f'SI-action-HSVT, +test',
}


def plot_mse_availability(df, sorted_perturbations=None, sorted_celltypes=None, ytick_space=10, vmin=None, vmax=None):
    plt.clf()
    if sorted_perturbations is None:
        sorted_perturbations = df.groupby('intervention').size().sort_values(ascending=True)
    if sorted_celltypes is None:
        sorted_celltypes = df.groupby('unit').size().sort_values(ascending=False)

    mse_matrix = np.zeros((len(sorted_perturbations), len(sorted_celltypes)))
    mse_matrix.fill(np.nan)

    pert2ix = {pert: ix for ix, pert in enumerate(sorted_perturbations.index)}
    cell2ix = {cell: ix for ix, cell in enumerate(sorted_celltypes.index)}
    pert_ixs, cell_ixs = df.index.get_level_values('intervention').map(pert2ix), df.index.get_level_values(
        'unit').map(cell2ix)
    mse_matrix[pert_ixs, cell_ixs] = df.values.flatten()
    # masked_count_matrix = masked_array(count_matrix, count_matrix==0)

    plt.imshow(mse_matrix, aspect='auto', interpolation='none', cmap='hot', vmin=vmin, vmax=vmax)
    plt.grid(False)
    plt.xlabel("Cell Types")
    plt.ylabel("Perturbation IDs")
    plt.xticks(list(range(len(sorted_celltypes))))
    plt.yticks(list(reversed(range(len(sorted_perturbations), 0, -ytick_space))),
               list(reversed(range(0, len(sorted_perturbations), ytick_space))))
    ax = plt.gca()
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize='x-small')
    ax.set_xticklabels([str(ix + 1) + ":" + ct for ix, ct in enumerate(sorted_celltypes.index)], ha='right',
                       rotation=70)
    plt.tight_layout()


def compute_r2_matrix(true_values, predicted_values):
    """Faster implementation of sklearn.metrics.r2_score for matrices"""
    true_means = true_values.mean(axis=1)
    baseline_error = np.sum((true_values - true_means[:, None])**2, axis=1)
    true_error = np.sum((true_values - predicted_values)**2, axis=1)
    return 1 - true_error/baseline_error


def compute_relative_mse_matrix(true_values, predicted_values, baseline):
    baseline_error = np.sum((true_values - baseline)**2, axis=1)
    true_error = np.sum((true_values - predicted_values)**2, axis=1)
    return true_error/baseline_error


def compute_relative_rmse_iqr(true_values, predicted_values, baseline):
    baseline_error = compute_normalized_rmse_iqr_matrix(true_values, baseline)
    true_error = compute_normalized_rmse_iqr_matrix(true_values, predicted_values)
    return true_error/baseline_error


def compute_rmse_matrix(true_values, predicted_values):
    rmse = np.sqrt(np.mean((true_values - predicted_values)**2, axis=1))
    return rmse


def compute_mean_absolute_error_matrix(true_values, predicted_values):
    mae = np.mean(np.abs(true_values - predicted_values), axis=1)
    return mae


def compute_relative_mean_absolute_error_matrix(true_values, predicted_values):
    mae = np.mean(np.abs(true_values - predicted_values), axis=1)
    rmae = mae / np.median(true_values, axis=1)
    return rmae


def compute_normalized_rmse_matrix(true_values, predicted_values):
    rmse = compute_rmse_matrix(true_values, predicted_values)
    normalized_rmse = rmse / np.mean(true_values, axis=1)
    return normalized_rmse


def compute_normalized_rmse_iqr_matrix(true_values, predicted_values):
    rmse = compute_rmse_matrix(true_values, predicted_values)
    iqrs = iqr(true_values, axis=1)
    normalized_rmse = rmse / iqrs
    return normalized_rmse


def compute_r2_vector(true_values, predicted_values):
    true_mean = true_values.mean()
    baseline_error = np.sum((true_values - true_mean)**2)
    true_error = np.sum((true_values - predicted_values)**2)
    return 1 - true_error/baseline_error


def compute_cosine_sim(true_vectors, predicted_vectors):
    numerators = np.sum(true_vectors * predicted_vectors, axis=1)
    denominators = np.linalg.norm(true_vectors, axis=1) * np.linalg.norm(predicted_vectors, axis=1)
    return numerators/denominators


class EvaluationManager:
    def __init__(self, prediction_manager: PredictionManager):
        self.prediction_manager = prediction_manager
        self.plot_folder = f"evaluation/new_plots/{self.prediction_manager.result_string}"
        os.makedirs(self.plot_folder, exist_ok=True)
        self.plot_folder2 = os.path.expanduser(f'~/Desktop/cmap-imputation/{self.prediction_manager.result_string}')
        os.makedirs(self.plot_folder2, exist_ok=True)
        self.collected_results = None
        self.true_results = None
        self.true_diffs = None
        self.predicted_diffs = None

    def savefig(self, filename):
        plt.savefig(f"{self.plot_folder}/{filename}.png", dpi=200)
        print(f"[EvaluationManager] Saving to {os.path.abspath(filename)}")

    def plots(self, method):
        self._collect_results()
        self.boxplot_r2()
        self.boxplot_cosines()
        self.boxplot_rmse()
        self.boxplot_rmae()
        self.boxplot_normalized_rmse()
        self.boxplot_normalized_rmse_iqr()
        self.boxplot_relative_mse()
        self.boxplot_mae()
        self.boxplot_relative_rmse_iqr()
        self.plot_times()

        # stats_df = self.get_res_and_stats(method)
        # self.plot_train_error_vs_rmse(stats_df)
        # self.plot_train_error_vs_cosine(stats_df)
        # self.plot_train_error_vs_relative_mse(stats_df)

        self.plot_method_mse(method)
        self.plot_method_relative_mse(method)
        self.plot_quantile_relative_mse(method)

        self.cosines()

    def _collect_results(self):
        num_rows_per_alg = sum((len(ixs) for ixs in self.prediction_manager.fold_test_ixs))
        num_rows = num_rows_per_alg * len(self.prediction_manager.prediction_filenames)
        num_features = self.prediction_manager.gene_expression_df.shape[1]
        predicted_vals = np.zeros((num_rows, num_features))
        true_vals = np.zeros((num_rows, num_features))
        index = []

        ix = 0
        for alg, prediction_filename in self.prediction_manager.prediction_filenames.items():
            print(f'[EvaluationManager._collect_results] computing statistic for {alg}')
            predicted_df = pd.read_pickle(prediction_filename)
            predicted_vals[ix:(ix + predicted_df.shape[0])] = predicted_df.values

            ixs = np.concatenate(self.prediction_manager.fold_test_ixs)
            true_vals[ix:(ix + predicted_df.shape[0])] = self.prediction_manager.gene_expression_df.iloc[ixs].values

            units, ivs = predicted_df.index.get_level_values('unit'), predicted_df.index.get_level_values(
                'intervention')
            index.extend(list(zip(units, ivs, list(range(predicted_df.shape[0])), [alg] * len(units))))
            ix += predicted_df.shape[0]

        pd_index = pd.MultiIndex.from_tuples(index, names=['unit', 'intervention', 'fold_ix', 'alg'])
        self.collected_results = pd.DataFrame(predicted_vals, index=pd_index)
        self.true_results = pd.DataFrame(true_vals, index=pd_index)

        controls = self.prediction_manager.gene_expression_df
        controls = controls[controls.index.get_level_values("intervention") == "DMSO"]
        controls.reset_index("intervention", drop=True, inplace=True)
        controls.columns = self.true_results.columns
        self.true_diffs = self.true_results.subtract(controls, level="unit")
        self.predicted_diffs = self.collected_results.subtract(controls, level="unit")

    def r2(self):
        if self.collected_results is None: self._collect_results()
        r2s = compute_r2_matrix(self.true_results.values, self.collected_results.values)
        return pd.DataFrame(r2s, index=self.true_results.index)

    def cosines(self):
        if self.collected_results is None: self._collect_results()
        cosines = compute_cosine_sim(self.true_diffs.values, self.predicted_diffs.values)
        return pd.DataFrame(cosines, index=self.true_results.index)

    def rmse(self):
        if self.collected_results is None: self._collect_results()
        rmses = compute_rmse_matrix(self.true_results.values, self.collected_results.values)
        return pd.DataFrame(rmses, index=self.true_results.index)

    def rmae(self):
        if self.collected_results is None: self._collect_results()
        rmaes = compute_relative_mean_absolute_error_matrix(self.true_results.values, self.collected_results.values)
        return pd.DataFrame(rmaes, index=self.true_results.index)

    def mean_absolute_error(self):
        if self.collected_results is None: self._collect_results()
        maes = compute_mean_absolute_error_matrix(self.true_results.values, self.collected_results.values)
        return pd.DataFrame(maes, index=self.true_results.index)

    def normalized_rmse(self):
        if self.collected_results is None: self._collect_results()
        nrmses = compute_normalized_rmse_matrix(self.true_results.values, self.collected_results.values)
        return pd.DataFrame(nrmses, index=self.true_results.index)

    def normalized_rmse_iqr(self):
        if self.collected_results is None: self._collect_results()
        nrmses = compute_normalized_rmse_iqr_matrix(self.true_results.values, self.collected_results.values)
        return pd.DataFrame(nrmses, index=self.true_results.index)

    def relative_mse(self, baseline_alg="alg=impute_unit_mean"):
        if self.collected_results is None: self._collect_results()
        baseline = self.collected_results[self.collected_results.index.get_level_values("alg") == baseline_alg]
        num_algs = int(self.collected_results.values.shape[0] / baseline.shape[0])
        baseline = np.tile(baseline, (num_algs, 1))
        relative_mses = compute_relative_mse_matrix(self.true_results.values, self.collected_results.values, baseline)
        return pd.DataFrame(relative_mses, index=self.true_results.index)

    def relative_normalized_rmse_iqr(self, baseline_alg="alg=impute_unit_mean"):
        if self.collected_results is None: self._collect_results()
        baseline = self.collected_results[self.collected_results.index.get_level_values("alg") == baseline_alg]
        num_algs = int(self.collected_results.values.shape[0] / baseline.shape[0])
        baseline = np.tile(baseline, (num_algs, 1))
        relative_mses = compute_relative_rmse_iqr(self.true_results.values, self.collected_results.values, baseline)
        return pd.DataFrame(relative_mses, index=self.true_results.index)

    # === BOX PLOTS
    def boxplot_r2(self):
        r2_df = self.r2()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        r2_dict = {alg_names[alg]: r2_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Coefficient of determination, $R^2$',
            title=self.prediction_manager.result_string,
            top=1.05,
            bottom=.5,
            scale=.03
        )
        plt.title("")
        self.savefig(f'r2_boxplot')

    def boxplot_cosines(self):
        cosine_df = self.cosines()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        cos_dict = {alg_names[alg]: cosine_df.query('alg == @alg').values.flatten() for alg in algs}
        # === FILTER OUT NAN'S, WHICH COME FROM FIXED EFFECTS PREDICTING THE CONTROL WHEN THERE IS NO
        # === OTHER CONTEXT WITH THE INTERVENTION
        cos_dict = {k: v[~np.isnan(v)] for k, v in cos_dict.items()}
        # cos_dict = {k: np.nan_to_num(v) for k, v in cos_dict.items()}
        plt.clf()
        boxplots(
            cos_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Cosine similarity',
            title=self.prediction_manager.result_string,
            top=1.05,
            bottom=0,
            scale=.04
        )
        plt.title("")
        self.savefig(f'cosine_boxplot')

    def boxplot_rmse(self):
        rmse_df = self.rmse()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        # algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: rmse_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Root-mean-square error (RMSE)',
            bottom=0,
            top=1000,
            scale=.04,
            title=self.prediction_manager.result_string,
        )
        plt.title("")
        self.savefig("rmse_boxplot")

    def boxplot_rmae(self):
        rmae_df = self.rmae()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        # algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: rmae_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Relative mean absolute error (RMAE)',
            bottom=0,
            top=1,
            scale=.04,
            title=self.prediction_manager.result_string,
        )
        plt.title("")
        self.savefig("rmae_boxplot")

    def boxplot_mae(self):
        mae_df = self.mean_absolute_error()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        # algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: mae_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Mean absolute error (MAE)',
            bottom=0,
            top=1000,
            scale=.04,
            title=self.prediction_manager.result_string,
        )
        plt.title("")
        self.savefig("mae_boxplot")

    def boxplot_normalized_rmse(self):
        rmse_df = self.normalized_rmse()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        # algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: rmse_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Normalized RMSE',
            bottom=0,
            top=2,
            scale=.04,
            title=self.prediction_manager.result_string,
        )
        plt.title("")
        self.savefig("normalized_rmse_boxplot")

    def boxplot_normalized_rmse_iqr(self):
        rmse_df = self.normalized_rmse_iqr()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        # algs = list(alg_names.keys())
        r2_dict = {alg_names[alg]: rmse_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Normalized root-mean-square error (NRMSE)',
            bottom=0,
            top=1.5,
            scale=.04,
            title=self.prediction_manager.result_string,
        )
        plt.title("")
        self.savefig("normalized_rmse_iqr_boxplot")

    def boxplot_relative_mse(self):
        df = self.relative_mse()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        r2_dict = {alg_names[alg]: df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Relative MSE',
            title=self.prediction_manager.result_string,
            top=1,
            bottom=.5,
            scale=.03
        )
        plt.ylim([0, 5])
        plt.title("")
        self.savefig("relative_mse_boxplot")

    def boxplot_relative_rmse_iqr(self):
        df = self.relative_mse()
        algs = list(self.prediction_manager.prediction_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))
        r2_dict = {alg_names[alg]: df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='Relative RMSE-IQR',
            title=self.prediction_manager.result_string,
            top=2.4,
            bottom=0,
            scale=.03
        )
        plt.ylim([0, 5])
        plt.title("")
        self.savefig("relative_rmse_iqr_boxplot")

    def plot_times(self):
        alg2times = dict()
        algs = list(self.prediction_manager.time_filenames.keys())
        algs = sorted(algs, key=lambda alg: list(alg_names).index(alg))

        for alg in algs:
            times = np.loadtxt(self.prediction_manager.time_filenames[alg])
            alg2times[alg_names[alg]] = np.log10(times)
        boxplots(
            alg2times,
            boxColors,
            xlabel='Algorithm',
            ylabel='Log base 10 of (# of seconds) per prediction.',
            title=self.prediction_manager.result_string,
            top=2.5,
            bottom=-2.5,
        )
        plt.title("")
        self.savefig(f'time_boxplot')

    # === PER-ENTRY
    def plot_method_mse(self, method, ytick_space=10):
        df = self.rmse()
        df = df[df.index.get_level_values("alg") == method]
        plot_mse_availability(df)
        self.savefig(f"mse_availability")

    def plot_method_relative_mse(self, method, ytick_space=10):
        df = self.relative_mse()
        df = df[df.index.get_level_values("alg") == method]
        sorted_perturbations = df.groupby('intervention').size().sort_values(ascending=True)
        sorted_celltypes = df.groupby('unit').size().sort_values(ascending=False)

        plot_mse_availability(df, sorted_perturbations, sorted_celltypes, vmin=0, vmax=10)
        self.savefig(f"relative_mse_availability")

        plot_mse_availability(df[df >= 1], sorted_perturbations, sorted_celltypes, vmin=0, vmax=10)
        self.savefig(f"relative_mse_availability_just_worse")

    def plot_quantile_relative_mse(self, method):
        df = self.relative_mse()
        vals = df[df.index.get_level_values("alg") == method].values
        grid = np.linspace(.01, 1, 100)
        quantiles = np.quantile(vals, grid)
        plt.clf()
        plt.plot(grid, quantiles)
        plt.axhline(1, color='k')
        plt.ylabel("Relative MSE of SI-Action at quantile")
        self.savefig(f"relative_mse_quantile_{method}")

    def get_res_and_stats(self, method):
        relative_mse_df = self.relative_mse()
        relative_mse_df = relative_mse_df[relative_mse_df.index.get_level_values("alg") == method]
        rmse_df = self.rmse()
        rmse_df = rmse_df[rmse_df.index.get_level_values("alg") == method]
        cos_df = self.cosines()
        cos_df = cos_df[cos_df.index.get_level_values("alg") == method]

        stats_df = pd.read_pickle(f"{self.prediction_manager.result_folder}/{method}_stats.pkl")

        stats_df["relative_mse"] = relative_mse_df.values
        stats_df["cosine"] = cos_df.values
        stats_df["rmse"] = rmse_df.values

        return stats_df

    def plot_train_error_vs_rmse(self, stats_df):
        plt.clf()
        plt.scatter(stats_df["train_error"], stats_df["rmse"])
        plt.xlabel("Training error")
        plt.ylabel("RMSE")
        plt.tight_layout()
        self.savefig("train_error_vs_rmse")

    def plot_train_error_vs_cosine(self, stats_df):
        plt.clf()
        plt.scatter(stats_df["train_error"], stats_df["cosine"])
        plt.xlabel("Training error")
        plt.ylabel("Cosine")
        plt.tight_layout()
        self.savefig("train_error_vs_cosine")

    def plot_train_error_vs_relative_mse(self, stats_df):
        plt.clf()
        plt.scatter(stats_df["train_error"], stats_df["relative_mse"])
        plt.xlabel("Training error")
        plt.ylabel("Relative MSE")
        plt.tight_layout()
        self.savefig("train_error_vs_relative_mse")

    # === OLD
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

    def boxplot_per_intervention(self):
        r2_df = self.r2_per_iv()
        algs = list(set(r2_df.index.get_level_values('alg')))
        r2_dict = {alg_names[alg]: r2_df.query('alg == @alg').values.flatten() for alg in algs}
        plt.clf()
        boxplots(
            r2_dict,
            boxColors,
            xlabel='Algorithm',
            ylabel='$R^2$ score per context/action pair',
            title=self.prediction_manager.result_string,
            top=1.1,
            bottom=0,
            scale=.08
        )
        plt.title("")
        self.savefig("boxplot_per_iv")

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
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/{self.prediction_manager.result_string}/statistic-vs-best_{self.prediction_manager.result_string}.png'))

        plt.clf()
        r2s_mean = r2s[r2s.index.get_level_values('alg') == 'alg=impute_unit_mean']
        plt.scatter(stats, r2s_mean.values)
        plt.xlabel("Statistic")
        plt.ylabel("R^2 of mean-over-actions")
        plt.ylim([0, 1])
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/{self.prediction_manager.result_string}/statistic-vs-mean-r2_{self.prediction_manager.result_string}.png'))

        plt.clf()
        r2s_si_hsvt = r2s[r2s.index.get_level_values('alg') == 'alg=predict_synthetic_intervention_hsvt_ols,num_desired_donors=None,energy=0.95,hypo_test=True,hypo_test_percent=0.1,donor_dim=intervention,equal_rank=True']
        plt.scatter(stats, r2s_si_hsvt.values)
        plt.xlabel("Statistic")
        plt.ylabel("R^2 of SI-action")
        plt.ylim([0, 1])
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/{self.prediction_manager.result_string}/statistic-vs-si-hsvt-r2_{self.prediction_manager.result_string}.png'))

        plt.clf()
        r2s_si = r2s[r2s.index.get_level_values('alg') == 'alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention']
        plt.scatter(stats, r2s_si.values)
        plt.xlabel("Statistic")
        plt.ylabel("R^2 of SI-action")
        plt.ylim([0, 1])
        plt.savefig(os.path.expanduser(f'~/Desktop/cmap-imputation/{self.prediction_manager.result_string}/statistic-vs-si-r2_{self.prediction_manager.result_string}.png'))

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
