from evaluation.helpers.prediction_manager2 import PredictionManager
import pandas as pd
from sklearn.metrics import r2_score
import ipdb
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set()


class EvaluationManager:
    def __init__(self, prediction_manager: PredictionManager):
        self.prediction_manager = prediction_manager

    def r2(self):
        alg2r2s = {}
        for alg, predicted_dfs in self.prediction_manager._predictions.items():
            print(f'[EvaluationManager.r2] computing r2 for {alg}')
            r2_dfs = []
            for test_ixs, predicted_df in zip(self.prediction_manager.fold_test_ixs, predicted_dfs):
                # get test data that was held out in this fold
                test_df = self.prediction_manager.gene_expression_df.iloc[test_ixs]
                test_df = test_df.sort_index()

                # take the predicted df at the corresponding indices
                predicted_df = predicted_df.sort_index()
                # predicted_df = predicted_df.loc[test_df.index]

                # compute the R2 score for each gene expression profile
                r2s = []
                for ((unit, iv), predicted_values), (_, true_values) in zip(predicted_df.iterrows(), test_df.iterrows()):
                    r2s.append({'unit': unit, 'intervention': iv, f'r2_{alg}': r2_score(true_values, predicted_values)})
                r2_df = pd.DataFrame(r2s)
                r2_df = r2_df.set_index(['unit', 'intervention'])
                r2_dfs.append(r2_df)
            alg2r2s[alg] = r2_dfs

        return alg2r2s

    def mse(self):
        alg2mse = dict()

    def boxplot(self):
        alg2r2s = self.r2()
        algs = list(alg2r2s.keys())
        full_dfs = [pd.concat(r2_dfs) for alg, r2_dfs in alg2r2s.items()]
        plt.clf()
        plt.boxplot([full_df.values.flatten() for full_df in full_dfs], labels=algs)
        plt.ylim([0, 1])
        plt.legend()
        os.makedirs('evaluation/plots', exist_ok=True)
        plt.savefig(f'evaluation/plots/boxplot_{self.prediction_manager.result_string}.png')
