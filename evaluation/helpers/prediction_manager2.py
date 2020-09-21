import pandas as pd
import os
import numpy as np
from typing import List
from evaluation.helpers.get_data_block import get_data_block
from tqdm import tqdm
import ipdb

BLACKLIST_KWARGS = {'verbose'}
ADD_METRIC = True


class PredictionManager:
    def __init__(
            self,
            cell_start,
            cell_stop,
            pert_start,
            pert_stop,
            name='level2_filtered',
            num_folds: int = 5,
            seed: int = 8838
    ):
        self.result_string = f'cell={cell_start}-{cell_stop},pert={pert_start}-{pert_stop},name={name},num_folds={num_folds}'
        self.result_folder = os.path.join('evaluation', 'results', self.result_string)

        self.gene_expression_df, self.units, self.interventions, _ = get_data_block(cell_start, cell_stop, pert_start, pert_stop, name=name)
        # sort so that DMSO comes first
        control_ixs = self.gene_expression_df.index.get_level_values('intervention') == "DMSO"
        num_control_profiles = control_ixs.sum()
        sort_ixs = np.argsort(1 - control_ixs)
        self.gene_expression_df = self.gene_expression_df.iloc[sort_ixs]

        np.random.seed(seed)
        num_profiles = self.gene_expression_df.shape[0]
        profile_ixs = list(range(num_control_profiles, num_profiles))
        self.num_folds = num_folds
        self.fold_test_ixs = np.array_split(np.random.permutation(profile_ixs), num_folds)
        self.fold_train_ixs = [
            list(range(num_control_profiles)) + list(set(profile_ixs) - set(test_ixs))
            for test_ixs in self.fold_test_ixs
        ]

        self._predictions = dict()

    def predict(
            self,
            alg,
            overwrite=False,
            **kwargs
    ) -> List[pd.DataFrame]:
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

        # simply return if the results are already loaded.
        if self._predictions.get(full_alg_name) is not None:
            return self._predictions[full_alg_name]
        else:
            # filenames for the results of each fold
            alg_results_folder = os.path.join(self.result_folder, full_alg_name)
            os.makedirs(alg_results_folder, exist_ok=True)
            result_filenames = [os.path.join(alg_results_folder, f'fold={k}.pkl') for k in range(self.num_folds)]

            # if results already exist, just load them
            if not overwrite and os.path.exists(result_filenames[0]):
                self._predictions[full_alg_name] = [pd.read_pickle(filename) for filename in result_filenames]
            else:
                print(f"Predicting for {full_alg_name}")

                # predict for each fold
                self._predictions[full_alg_name] = []
                for train_ixs, test_ixs, filename in tqdm(zip(self.fold_train_ixs, self.fold_test_ixs, result_filenames), total=self.num_folds):
                    training_df = self.gene_expression_df.iloc[train_ixs]
                    targets = self.gene_expression_df.iloc[test_ixs].index
                    df = alg(training_df, targets, **kwargs)

                    # save the results
                    self._predictions[full_alg_name].append(df)
                    df.to_pickle(filename)

            return self._predictions[full_alg_name]





