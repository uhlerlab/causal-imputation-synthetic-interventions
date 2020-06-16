from sklearn.metrics import r2_score
import pandas as pd
import os
from enum import Enum
import numpy as np
from typing import List

BLACKLIST_KWARGS = {'verbose'}
ADD_METRIC = True


class PredictionManager:
    def __init__(
            self,
            result_folder: str,
            gene_expression_df: pd.DataFrame,
            control_intervention: str,
            num_folds: int = 5,
            seed: int = 8838
    ):
        """
        Base class for testing Phase I.

        Parameters
        ----------
        result_folder: name of folder in which to save results.
        gene_expression_df: DataFrame containing all gene expression data.
        control_intervention: name of the control intervention for methods which need a specified control.
        num_folds: Number of folds used when checking performance.
        seed: random seed determining partitioning of folds.
        """
        self.result_folder = os.path.join('evaluation', 'results', result_folder, f'num_folds={num_folds}')

        self.gene_expression_df = gene_expression_df
        self.control_df = gene_expression_df[gene_expression_df.index.get_level_values('intervention') == control_intervention]

        self.units = list(gene_expression_df.index.get_level_values('unit').unique())
        self.interventions = list(gene_expression_df.index.get_level_values('intervention').unique())
        self.control_intervention = control_intervention

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
            alg_name: str,
            overwrite=False,
            **kwargs
    ) -> List[pd.DataFrame]:
        """
        Use an algorithm to predict held-out values from each fold.

        Parameters
        ----------
        alg: A function taking a dataframe of unit/iv pairs and gene expression values, and returning another dataframe
            of predicted gene expression values for other unit/iv pairs.
        alg_name: A string signifying the name of the algorithm for saving/loading results.
        overwrite: if True, overwrite previous results even if they are stored.
        kwargs: any additional arguments (e.g., regularization parameter) passed to the algorithm.

        Returns
        -------

        """
        # form a string to fully identify the algorithm and its parameter settings
        kwarg_str = '' if not kwargs else ',' + ','.join(f'{k}={v}' for k, v in kwargs.items() if k not in BLACKLIST_KWARGS)
        full_alg_name = f'alg={alg_name}{kwarg_str}'

        # simply return if the results are already loaded.
        if self._predictions.get(full_alg_name) is not None:
            return self._predictions[full_alg_name]
        else:
            # filenames for the results of each fold
            folder = os.path.join(self.result_folder, full_alg_name)
            os.makedirs(folder, exist_ok=True)
            filenames = [os.path.join(folder, f'fold={k}.csv') for k in range(self.num_folds)]

            # if results already exist, just load them
            if not overwrite and os.path.exists(filenames[0]):
                self._predictions[full_alg_name] = [pd.read_csv(filename, index_col=0) for filename in filenames]
            else:
                print(f"Predicting for {full_alg_name}")

                # predict for each fold
                self._predictions[full_alg_name] = []
                for train_ixs, test_ixs, filename in zip(self.fold_train_ixs, self.fold_test_ixs, filenames):
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

    def r2(
            self,
            alg,
            alg_name: str,
            overwrite=False,
            **kwargs
    ) -> List[pd.DataFrame]:
        """
        Compute the R2 score for each profile in each fold.

        Parameters
        ----------
        alg: A function taking a dataframe of unit/iv pairs and gene expression values, and returning another dataframe
            of predicted gene expression values for other unit/iv pairs.
        alg_name: A string signifying the name of the algorithm for saving/loading results.
        overwrite: if True, overwrite previous results even if they are stored.
        kwargs: any additional arguments (e.g., regularization parameter) passed to the algorithm.

        Returns
        -------

        """

        predicted_dfs = self.predict(alg, alg_name, overwrite=overwrite, **kwargs)
        r2_dfs = []
        for test_ixs, predicted_df in zip(self.fold_test_ixs, predicted_dfs):
            # get test data that was held out in this fold
            test_df = self.gene_expression_df.iloc[test_ixs]
            test_df = test_df.sort_values('unit').sort_values('intervention')
            test_df = test_df.set_index(['unit', 'intervention'])

            # take the predicted df at the corresponding indices
            predicted_df = predicted_df.set_index(['unit', 'intervention'])
            predicted_df = predicted_df.loc[test_df.index]

            # compute the R2 score for each gene expression profile
            r2s = []
            for ((unit, iv), predicted_values), (_, true_values) in zip(predicted_df.iterrows(), test_df.iterrows()):
                r2s.append({'unit': unit, 'intervention': iv, f'r2_{alg_name}': r2_score(true_values, predicted_values)})
            r2_df = pd.DataFrame(r2s)
            r2_df = r2_df.set_index(['unit', 'intervention'])
            r2_dfs.append(r2_df)

        return r2_dfs





