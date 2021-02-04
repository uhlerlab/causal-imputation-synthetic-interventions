import pandas as pd
import os
import numpy as np
from typing import List, Optional
from evaluation.helpers.get_data_block import get_data_block
from tqdm import tqdm
import ipdb
from multiprocessing import Pool, cpu_count
from p_tqdm import p_map
from src.algorithms import predict_synthetic_intervention_ols, predict_synthetic_intervention_hsvt_ols
from time import time


BLACKLIST_KWARGS = {'verbose', 'multithread', 'overwrite', 'progress'}
ADD_METRIC = True


class PredictionManager:
    def __init__(
            self,
            cell_start: Optional[int] = None,
            num_cells: Optional[int] = 10,
            pert_start: Optional[int] = None,
            num_perts: Optional[int] = 20,
            average: bool = True,
            name='level2_filtered',
            num_folds: Optional[int] = 5,
            seed: int = 8838
    ):
        self.result_string = f'cell={cell_start},{num_cells}cells,pert={pert_start},{num_perts}perts,name={name},num_folds={num_folds},average={average}'
        self.result_folder = os.path.join('evaluation', 'results', self.result_string)
        os.makedirs(self.result_folder, exist_ok=True)

        if name == 'old_data':
            old_df0 = pd.read_csv('old_data/df0.csv', sep='\t', index_col=0)
            old_df1 = pd.read_csv('old_data/df1_all.csv', sep='\t', index_col=0)
            old_df = pd.concat([old_df0, old_df1])
            old_df.set_index(['unit', 'intervention'], inplace=True)
            self.gene_expression_df = old_df
        else:
            self.gene_expression_df, _, _ , _ = get_data_block(
                num_cells=num_cells,
                num_perts=num_perts,
                cell_start=cell_start,
                pert_start=pert_start,
                name=name,
                average=average
            )
        print(self.gene_expression_df.groupby('intervention').size().sort_values(ascending=False))
        print(self.gene_expression_df.groupby('unit').size().sort_values(ascending=False))
        # sort so that DMSO comes first
        control_ixs = self.gene_expression_df.index.get_level_values('intervention') == "DMSO"
        num_control_profiles = control_ixs.sum()
        sort_ixs = np.argsort(1 - control_ixs)
        self.gene_expression_df = self.gene_expression_df.iloc[sort_ixs]

        np.random.seed(seed)
        num_profiles = self.gene_expression_df.shape[0]
        profile_ixs = list(range(num_control_profiles, num_profiles))
        self.num_folds = num_folds if num_folds is not None else len(profile_ixs)
        print(f"[PredictionManager.__init__] creating test/training indices")
        self.fold_test_ixs = np.array_split(np.random.permutation(profile_ixs), self.num_folds)
        self.fold_train_ixs = [
            list(range(num_control_profiles)) + list(set(profile_ixs) - set(test_ixs))
            for test_ixs in self.fold_test_ixs
        ]
        print(f"[PredictionManager.__init__] done creating test/training indices")

        self.prediction_filenames = dict()
        self.statistic_filenames = dict()
        self.time_filenames = dict()

    def predict(
            self,
            alg,
            overwrite=False,
            multithread=False,
            **kwargs
    ) -> (List[pd.DataFrame], List[np.array]):
        """
        Use an algorithm to predict held-out values from each fold.

        Parameters
        ----------
        alg: A function taking a dataframe of unit/iv pairs and gene expression values, and returning another dataframe
            of predicted gene expression values for other unit/iv pairs.
        overwrite: if True, overwrite previous results even if they are stored.
        kwargs: any additional arguments (e.g., regularization parameter) passed to the algorithm.

        Returns
        -------

        """
        # form a string to fully identify the algorithm and its parameter settings
        kwarg_str = '' if not kwargs else ',' + ','.join(f'{k}={v}' for k, v in kwargs.items() if k not in BLACKLIST_KWARGS)
        full_alg_name = f'alg={alg.__name__}{kwarg_str}'

        # filenames for the results of each fold
        alg_results_filename = os.path.join(self.result_folder, full_alg_name + '.pkl')
        alg_times_filename = os.path.join(self.result_folder, full_alg_name + '_times.pkl')
        alg_stats_filename = os.path.join(self.result_folder, full_alg_name + '_stats.pkl')
        self.prediction_filenames[full_alg_name] = alg_results_filename
        self.statistic_filenames[full_alg_name] = alg_stats_filename
        self.time_filenames[full_alg_name] = alg_times_filename

        # if results already exist, just load them
        if not overwrite and os.path.exists(alg_results_filename):
            print(f"[PredictionManager.predict] loading predictions for {full_alg_name}")
            prediction_df = pd.read_pickle(alg_results_filename)
            times_taken = np.loadtxt(alg_times_filename)
        else:
            print(f"[PredictionManager.predict] Predicting for {full_alg_name}")

            def run(p):
                fold_ix, train_ixs, test_ixs = p
                training_df = self.gene_expression_df.iloc[train_ixs]
                targets = self.gene_expression_df.iloc[test_ixs].index

                start_time = time()
                if alg == predict_synthetic_intervention_hsvt_ols:
                    df, stat_df = alg(training_df, targets, **kwargs)
                    df = df.loc[targets]
                    stat_df = stat_df.loc[targets]
                    df['fold'] = [fold_ix]*df.shape[0]
                    stat_df['fold'] = [fold_ix]*stat_df.shape[0]
                    df.set_index('fold', append=True, inplace=True)
                    stat_df.set_index('fold', append=True, inplace=True)

                    assert df.shape == (len(test_ixs), training_df.shape[1])
                    return df, stat_df, time() - start_time
                else:
                    df = alg(training_df, targets, **kwargs)
                    df = df.loc[targets]
                    df['fold'] = [fold_ix]*df.shape[0]
                    df.set_index('fold', append=True, inplace=True)

                    assert df.shape == (len(test_ixs), training_df.shape[1])
                    return df, time() - start_time

            tasks = list(zip(range(self.num_folds), self.fold_train_ixs, self.fold_test_ixs))

            if multithread:
                with Pool(cpu_count()-1) as pool:
                    prediction_dfs = p_map(run, tasks)
            else:
                results = list(tqdm((run(task) for task in tasks), total=len(tasks)))

            if alg == predict_synthetic_intervention_hsvt_ols:
                prediction_dfs, stat_dfs, times_taken = zip(*results)
                prediction_df = pd.concat(prediction_dfs, axis=0)
                prediction_df.to_pickle(alg_results_filename)
                statistic_df = pd.concat(stat_dfs, axis=0)
                statistic_df.to_pickle(alg_stats_filename)
                print(f"[PredictionManager.predict] Saving predictions to {alg_results_filename}")
                print(f"[PredictionManager.predict] Saving stats to {alg_stats_filename}")
            else:
                prediction_dfs, times_taken = zip(*results)
                prediction_df = pd.concat(prediction_dfs, axis=0)
                prediction_df.to_pickle(alg_results_filename)
                print(f"[PredictionManager.predict] Saving predictions to {alg_results_filename}")
            times_taken = np.array(times_taken)
            np.savetxt(alg_times_filename, times_taken)
            print(f"[PredictionManager.predict] Saving times to {alg_results_filename}")

        return prediction_df, times_taken






