import itertools as itr
import random
import numpy as np
from tqdm import tqdm
from utils import DonorFinder
import ipdb
import pandas as pd
from scipy.stats import iqr


def compute_normalized_rmse_iqr_matrix(true_values, predicted_values):
    rmse = compute_rmse_matrix(true_values, predicted_values)
    iqrs = iqr(true_values, axis=1)
    normalized_rmse = rmse / iqrs
    return normalized_rmse


def compute_r2_matrix(true_values, predicted_values):
    """Faster implementation of sklearn.metrics.r2_score for matrices"""
    true_means = true_values.mean(axis=1)
    baseline_error = np.sum((true_values - true_means[:, None])**2, axis=1)
    true_error = np.sum((true_values - predicted_values)**2, axis=1)
    return 1 - true_error/baseline_error


def compute_rmse_matrix(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values)**2, axis=1))


def compute_cosine_sim(true_vectors, predicted_vectors):
    numerators = np.sum(true_vectors * predicted_vectors, axis=1)
    denominators = np.linalg.norm(true_vectors, axis=1) * np.linalg.norm(predicted_vectors, axis=1)
    return numerators/denominators


class VaryAvailabilityManager:
    def __init__(self, df, min_donor=10, min_training=5):
        self.df = df
        self.min_donor = min_donor
        self.min_training = min_training
        donor_finder = DonorFinder(df)
        self.pairs_to_predict = []
        self.pairs2training_contexts = dict()
        self.pairs2donor_actions = dict()
        for target_context, target_action in df.index:
            training_contexts, donor_actions = donor_finder.get_donors(target_context, target_action)
            num_training, num_donors = len(training_contexts), len(donor_actions)
            if num_training > min_training and num_donors > min_donor:
                self.pairs_to_predict.append((target_context, target_action))
                self.pairs2training_contexts[(target_context, target_action)] = training_contexts
                self.pairs2donor_actions[(target_context, target_action)] = donor_actions
        print(f"[VaryAvailabilityManager.__init__] {len(self.pairs_to_predict)} pairs")

    def true_values(self):
        tasks = list(itr.product(range(1, self.min_donor), range(1, self.min_training), self.pairs_to_predict))
        values = np.empty((len(tasks), self.df.shape[1]))
        for ix, (num_donors, num_training, target_pair) in enumerate(tqdm(tasks)):
            target_context, target_action = target_pair
            values[ix] = df[
                (df.index.get_level_values("unit") == target_context) &
                (df.index.get_level_values("intervention") == target_action)
            ]

        df_index = pd.MultiIndex.from_tuples(tasks, names=["num_donor", "num_training", "pair"])
        value_df = pd.DataFrame(data=values, index=df_index)

        return value_df

    def predict(self):
        np.random.seed(12312)
        random.seed(12312)

        tasks = list(itr.product(range(1, self.min_donor), range(1, self.min_training), self.pairs_to_predict))
        predictions = np.empty((len(tasks), self.df.shape[1]))
        for ix, (num_donors, num_training, target_pair) in enumerate(tqdm(tasks)):
            target_context, target_action = target_pair

            candidate_training_contexts = self.pairs2training_contexts[target_pair]
            candidate_donor_actions = self.pairs2donor_actions[target_pair]
            training_contexts = random.sample(list(candidate_training_contexts), num_training)
            donor_actions = random.sample(list(candidate_donor_actions), num_donors)

            # === SUBSAMPLE DF
            subdf = df[
                df.index.get_level_values("unit").isin(set(training_contexts) | {target_context}) &
                df.index.get_level_values("intervention").isin(set(donor_actions) | {target_action}) &
                ~((df.index.get_level_values("unit") == target_context) & (df.index.get_level_values("intervention") == target_action))
            ]

            # === PREDICT
            targets = pd.MultiIndex.from_tuples([target_pair], names=["unit", "intervention"])
            prediction, _ = predict_synthetic_intervention_ols(subdf, targets, num_desired_donors=None)
            predictions[ix] = prediction.values

        df_index = pd.MultiIndex.from_tuples(tasks, names=["num_donor", "num_training", "pair"])
        prediction_df = pd.DataFrame(data=predictions, index=df_index)

        return prediction_df

    def r2_df(self):
        predicted_values = self.predict()
        true_values = self.true_values()
        r2_vals = compute_r2_matrix(true_values.values, predicted_values.values)
        r2_df = pd.DataFrame(data=r2_vals, index=predicted_values.index)
        return r2_df

    def rmse_df(self):
        predicted_values = self.predict()
        true_values = self.true_values()
        vals = compute_rmse_matrix(true_values.values, predicted_values.values)
        rmse_df = pd.DataFrame(data=vals, index=predicted_values.index)
        return rmse_df

    def nrmse_df(self):
        predicted_values = self.predict()
        true_values = self.true_values()
        vals = compute_normalized_rmse_iqr_matrix(true_values.values, predicted_values.values)
        nrmse_df = pd.DataFrame(data=vals, index=predicted_values.index)
        return nrmse_df

    # def cos_df(self):
    #     predicted_values = self.predict()
    #     true_values = self.true_values()
    #     vals = compute_cosine_sim(true_values.values, predicted_values.values)
    #     cos_df = pd.DataFrame(data=vals, index=predicted_values.index)
    #     return cos_df


if __name__ == '__main__':
    from evaluation.helpers.get_data_block import get_data_block
    from src.algorithms import predict_synthetic_intervention_ols
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    df, _, _, _ = get_data_block(
        num_cells=None,
        num_perts=100,
        name='level2'
    )
    min_donor = 10
    min_training = 5
    vam = VaryAvailabilityManager(df)


    xs = list(range(1, min_donor))

    # === R2
    r2_df = vam.r2_df()
    mean_df = r2_df.groupby(level=["num_donor", "num_training"]).mean()
    std_df = r2_df.groupby(level=["num_donor", "num_training"]).std()
    plt.clf()
    for num_training, color in zip(range(1, min_training), sns.color_palette()):
        means = mean_df[mean_df.index.get_level_values("num_training") == num_training].values.flatten()
        stds = .1 * std_df[std_df.index.get_level_values("num_training") == num_training].values.flatten()
        plt.plot(xs, means, label=num_training, color=color)
        plt.fill_between(xs, means-stds, means+stds, alpha=.3, color=color)
    plt.legend(title="Number of training contexts")
    plt.xlabel("Number of donor actions")
    plt.ylabel("Average $R^2$")
    plt.tight_layout()
    plt.savefig("visuals/figures/num-donors-num-training-r2.png")

    # === RMSE
    rmse_df = vam.rmse_df()
    mean_df = rmse_df.groupby(level=["num_donor", "num_training"]).mean()
    std_df = rmse_df.groupby(level=["num_donor", "num_training"]).std()
    plt.clf()
    for num_training, color in zip(range(1, min_training), sns.color_palette()):
        means = mean_df[mean_df.index.get_level_values("num_training") == num_training].values.flatten()
        stds = .1 * std_df[std_df.index.get_level_values("num_training") == num_training].values.flatten()
        plt.plot(xs, means, label=num_training, color=color)
        plt.fill_between(xs, means - stds, means + stds, alpha=.3, color=color)
    plt.legend(title="Number of training contexts")
    plt.xlabel("Number of donor actions")
    plt.ylabel("Average RMSE")
    plt.tight_layout()
    plt.savefig("visuals/figures/num-donors-num-training-rmse.png")

    # === NRMSE
    nrmse_df = vam.nrmse_df()
    mean_df = nrmse_df.groupby(level=["num_donor", "num_training"]).mean()
    std_df = nrmse_df.groupby(level=["num_donor", "num_training"]).std()
    plt.clf()
    for num_training, color in zip(range(1, min_training), sns.color_palette()):
        means = mean_df[mean_df.index.get_level_values("num_training") == num_training].values.flatten()
        stds = .1 * std_df[std_df.index.get_level_values("num_training") == num_training].values.flatten()
        plt.plot(xs, means, label=num_training, color=color)
        plt.fill_between(xs, means - stds, means + stds, alpha=.3, color=color)
    plt.legend(title="Number of training contexts")
    plt.xlabel("Number of donor actions")
    plt.ylabel("Average Normalized RMSE")
    plt.tight_layout()
    plt.savefig("visuals/figures/num-donors-num-training-nrmse.png")

