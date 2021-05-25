from tqdm import tqdm
import numpy as np
import pandas as pd
import ipdb
import operator as op
from collections import Counter, defaultdict
from src.algorithms.hsvt_regressor2 import HSVTRegressor2


class NoContextsWithTargetAction(UserWarning):
    def __init__(self, target_action):
        self.target_action = target_action
        super().__init__()

    def __str__(self):
        return f"The action \"{self.target_action}\" has never been seen in the training data."


class NoActionsWithTargetContext(UserWarning):
    def __init__(self, target_context):
        self.target_context = target_context
        super().__init__()

    def __str__(self):
        return f"The context \"{self.target_context}\" has never been seen in the training data."


def get_index_dict(df, key_level, value_level):
    d = defaultdict(set)
    for entry in df.index:
        d[entry[key_level]].add(entry[value_level])
    return d


class SyntheticInterventions:
    def __init__(
            self,
            regressor,
            regression_dim="intervention",
            context_dim="unit",
            num_donors=10,
    ):
        self.regressor = regressor
        self.regression_dim = regression_dim
        self.context_dim = context_dim
        self.num_donors = num_donors if num_donors is not None else float("inf")

        self.default_prediction = None
        self.training_df = None
        self.context2actions = None
        self.action2contexts = None

    def fit(self, training_df):
        df = training_df.reorder_levels([self.context_dim, self.regression_dim])
        df = df.sort_index(level=[self.regression_dim, self.context_dim])
        self.training_df = df
        self.context2actions = get_index_dict(df, 0, 1)
        self.action2contexts = get_index_dict(df, 1, 0)
        self.default_prediction = df.values.mean(axis=0)

    def _find_donors(self, target_context, target_action):
        # === FIND DONOR ACTIONS FOR THE TARGET CONTEXT
        remaining_actions = self.context2actions[target_context]
        if len(remaining_actions) == 0: raise NoActionsWithTargetContext(target_context)

        # === FIND CONTEXTS WHERE THE TARGET ACTION IS MEASURED
        current_contexts = self.action2contexts[target_action]
        if len(current_contexts) == 0: raise NoContextsWithTargetAction(target_action)

        # === GREEDILY PICK DONOR ACTIONS, UNTIL THE NUMBER OF TRAINING CONTEXTS WOULD BECOME ZERO _OR_
        # === WE HAVE ENOUGH DONOR ACTIONS AND DON'T WANT TO DECREASE THE NUMBER OF TRAINING_CONTEXTS
        donor_actions = set()
        while True:
            actions2new_contexts = {a: current_contexts & self.action2contexts[a] for a in remaining_actions}
            new_donor_action, new_contexts = max(actions2new_contexts.items(), key=op.itemgetter(1))
            if len(new_contexts) == 0: break
            if len(new_contexts) < len(current_contexts) and len(donor_actions) >= self.num_donors: break

            current_contexts = new_contexts
            donor_actions.add(new_donor_action)
            remaining_actions.remove(new_donor_action)
            if len(remaining_actions) == 0: break

        return current_contexts, donor_actions

    def _get_block(self, contexts, actions=None):
        df = self.training_df
        if isinstance(contexts, str):
            context_mask = df.index.get_level_values(0) == contexts
        elif isinstance(contexts, set):
            context_mask = df.index.get_level_values(0).isin(contexts)
        elif contexts is None:
            context_mask = True
        else:
            raise ValueError

        if isinstance(actions, str):
            action_mask = df.index.get_level_values(1) == actions
        elif isinstance(actions, set):
            action_mask = df.index.get_level_values(1).isin(actions)
        elif actions is None:
            action_mask = True
        else:
            raise ValueError

        block = df[context_mask & action_mask]
        if isinstance(contexts, str):
            vals = block.values.T
        elif isinstance(actions, str):
            vals = block.values.flatten()
        else:
            vals = block.values.reshape(-1, len(contexts)*df.shape[1]).T
        return vals

    def predict(self, targets, progress=False, statistics=False):
        targets = targets.reorder_levels([self.context_dim, self.regression_dim])
        iterator = enumerate(targets) if not progress else enumerate(tqdm(targets))

        num_features = self.training_df.shape[1]
        predicted_data = np.zeros((len(targets), num_features))
        statistic_data = np.zeros((len(targets), 7))
        statistic_data.fill(np.nan)
        for ix, (target_context, target_action) in iterator:
            try:
                training_contexts, donor_actions = self._find_donors(target_context, target_action)

                # === FIT
                train_x = self._get_block(training_contexts, donor_actions)
                train_y = self._get_block(training_contexts, target_action).flatten()
                assert train_x.shape == (num_features*len(training_contexts), len(donor_actions))
                assert train_y.shape == (num_features*len(training_contexts), )
                self.regressor.fit(train_x, train_y)

                # === PREDICT
                test_x = self._get_block(target_context, donor_actions)
                assert test_x.shape == (num_features, len(donor_actions))
                predicted_y = self.regressor.predict(test_x)
                predicted_data[ix] = predicted_y

                # === ADD RELEVANT STATISTICS
                statistic_data[ix, [2, 3]] = [(len(donor_actions), len(training_contexts))]
                statistic_data[ix, 4] = self.regressor.train_error
                if statistics:
                    pstat, rank_train, rank_test = self.regressor.projection_stat(train_x, test_x)
                    statistic_data[ix, 0] = pstat
                    statistic_data[ix, [5, 6]] = [rank_train, rank_test]
            except NoActionsWithTargetContext:
                predicted_data[ix] = self.default_prediction
            except NoContextsWithTargetAction:
                predicted_data[ix] = self._get_block(target_context).mean(axis=1)

        predicted_df = pd.DataFrame(predicted_data, index=targets, columns=self.training_df.columns)
        statistic_df = pd.DataFrame(statistic_data, index=targets, columns=["stat", "cv", "num_donors", "num_contexts", "train_error", "rank_train", "rank_test"])
        return predicted_df, statistic_df


def predict_synthetic_intervention_ols(
        df,
        targets,
        num_desired_donors,
        progress=False,
        donor_dim="intervention"
):
    regressor = HSVTRegressor2(energy=1)
    context_dim = list({"intervention", "unit"} - {donor_dim})[0]
    si = SyntheticInterventions(regressor, regression_dim=donor_dim, context_dim=context_dim, num_donors=num_desired_donors)
    si.fit(df)
    predicted_df, stats_df = si.predict(targets, progress=progress, statistics=False)
    predicted_df = predicted_df.reorder_levels(["unit", "intervention"])
    stats_df = stats_df.reorder_levels(["unit", "intervention"])
    return predicted_df, stats_df


if __name__ == '__main__':
    from evaluation.helpers.get_data_block import get_data_block
    from sklearn.linear_model import LinearRegression
    from src.algorithms.synthetic_interventions import predict_synthetic_intervention_ols as si1
    df, _, _, _ = get_data_block()
    targets = pd.MultiIndex.from_tuples([("SKB", "BRD-A76941896")], names=["unit", "intervention"])
    pred_df, stat_df = predict_synthetic_intervention_ols(df, targets, None)

    pred_df1, stat_df1 = si1(df, targets, None)


