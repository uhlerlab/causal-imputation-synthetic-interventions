from evaluation.prediction_manager2 import PredictionManager
import pandas as pd
from sklearn.metrics import r2_score


class EvaluationManager:
    def __init__(self, prediction_manager: PredictionManager):
        self.prediction_manager = prediction_manager

    def r2(self):
        for alg, predicted_dfs in self.prediction_manager._predictions:
            r2_dfs = []
            for test_ixs, predicted_df in zip(self.prediction_manager.fold_test_ixs, predicted_dfs):
                # get test data that was held out in this fold
                test_df = self.prediction_manager.gene_expression_df.iloc[test_ixs]
                test_df = test_df.sort_values('unit').sort_values('intervention')
                test_df = test_df.set_index(['unit', 'intervention'])

                # take the predicted df at the corresponding indices
                predicted_df = predicted_df.set_index(['unit', 'intervention'])
                predicted_df = predicted_df.loc[test_df.index]

                # compute the R2 score for each gene expression profile
                r2s = []
                for ((unit, iv), predicted_values), (_, true_values) in zip(predicted_df.iterrows(), test_df.iterrows()):
                    r2s.append({'unit': unit, 'intervention': iv, f'r2_{alg}': r2_score(true_values, predicted_values)})
                r2_df = pd.DataFrame(r2s)
                r2_df = r2_df.set_index(['unit', 'intervention'])
                r2_dfs.append(r2_df)

        return r2_dfs
