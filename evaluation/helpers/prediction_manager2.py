import pandas as pd
import os
import numpy as np
from typing import List
from evaluation.helpers.get_data_block import get_data_block

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
        self.result_folder = os.path.join('evaluation', 'results', f'cell={cell_start}-{cell_stop},pert={pert_start}-{pert_stop}', f'num_folds={num_folds}')

        self.gene_expression_df, self.units, self.interventions = get_data_block(cell_start, cell_stop, pert_start, pert_stop, name=name)
        self.control_df = self.gene_expression_df.query('intervention == "DMSO"')

        self.control_intervention = "DMSO"

        np.random.seed(seed)
        num_profiles = self.gene_expression_df.shape[0]
        profile_ixs = list(range(num_profiles))
        self.num_folds = num_folds
        self.fold_test_ixs = np.array_split(np.random.permutation(profile_ixs), num_folds)
        self.fold_train_ixs = [list(set(profile_ixs) - set(test_ixs)) for test_ixs in self.fold_test_ixs]

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
            result_filenames = [os.path.join(alg_results_folder, f'fold={k}.csv') for k in range(self.num_folds)]

            # if results already exist, just load them
            if not overwrite and os.path.exists(result_filenames[0]):
                self._predictions[full_alg_name] = [pd.read_csv(filename, index_col=0) for filename in result_filenames]
            else:
                print(f"Predicting for {full_alg_name}")

                # predict for each fold
                self._predictions[full_alg_name] = []
                for train_ixs, test_ixs, filename in zip(self.fold_train_ixs, self.fold_test_ixs, result_filenames):
                    training_df = self.gene_expression_df.iloc[train_ixs]
                    targets = self.gene_expression_df.iloc[test_ixs].index.to_list()
                    if ADD_METRIC:
                        control_df = self.control_df.copy()
                        control_df['metric'] = 'm0'
                        training_df['metric'] = 'm0'
                        control_df = control_df.reset_index(['unit', 'intervention'])
                        training_df = training_df.reset_index(['unit', 'intervention'])
                    else:
                        control_df = self.control_df

                    # depending on the type of the algorithm, feed dataframes in the right format
                    df = alg(
                        control_df,
                        training_df,
                        targets=targets,
                        **kwargs
                    )
                    # df = df.drop(columns=['metric'])

                    # save the results
                    self._predictions[full_alg_name].append(df)
                    df.to_csv(filename)

            return self._predictions[full_alg_name]





