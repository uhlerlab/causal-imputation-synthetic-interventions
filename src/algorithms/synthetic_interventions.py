import pandas as pd
import numpy as np
from collections import Counter
from utils import get_index_dict
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from functools import partial
from src.algorithms import HSVTRegressor, RejectionError
import ipdb


class NoContextsWithTargetAction(Exception):
    pass


class NoDonorActionsWithTargetContext(Exception):
    pass


def find_donors(
        sorted_donors,
        contexts2available_actions,
        target_context,
        target_action,
        num_desired_donors=None
):
    donor_actions_with_target_context = [
        donor for donor in sorted_donors
        if donor in contexts2available_actions[target_context]
    ]
    print(f"num donor actions: {len(donor_actions_with_target_context)}")
    if len(donor_actions_with_target_context) == 0:
        raise NoDonorActionsWithTargetContext

    # start with training data which has the target_donor_ix available
    current_training_contexts = [
        context for context, available_actions in contexts2available_actions.items()
        if target_action in available_actions
    ]
    print(f"num training contexts: {len(current_training_contexts)}")
    if len(current_training_contexts) == 0:
        raise NoContextsWithTargetAction

    donor_actions = set()
    while True:
        if len(donor_actions_with_target_context) == 0:
            break
        next_donor_action = donor_actions_with_target_context.pop(0)

        # stop adding donors if it would make the number of training dimensions zero
        new_training_contexts = [
            train_ix for train_ix in current_training_contexts
            if next_donor_action in contexts2available_actions[train_ix]
        ]
        if len(new_training_contexts) == 0:
            break

        # stop adding donors if we've reached the desired number of donors, and adding another
        # donor would decrease the number of training dimensions
        reached_num_desired = len(donor_actions) == num_desired_donors
        if reached_num_desired and len(new_training_contexts) < len(current_training_contexts):
            break

        # otherwise, it's fine to add another donor
        current_training_contexts = new_training_contexts
        donor_actions.add(next_donor_action)

    assert len(current_training_contexts) > 0
    assert len(donor_actions) > 0
    return current_training_contexts, donor_actions


def synthetic_intervention_inner(
        df,
        targets,
        regressor,
        predictor_no_training,
        default_prediction,
        regression_dim='intervention',
        num_desired_donors=None,
        progress=False
):
    assert regression_dim == 'intervention' or regression_dim == 'unit'
    context_dim = 'unit' if regression_dim == 'intervention' else 'intervention'

    df = df.sort_index(level=[regression_dim, context_dim])

    contexts2available_actions = get_index_dict(df, context_dim)
    counter = Counter(df.index.get_level_values(regression_dim))
    sorted_donors = [donor for donor, _ in counter.most_common()]

    num_features = df.shape[1]
    predicted_data = np.zeros((len(targets), num_features))
    statistic_data = np.zeros(len(targets))
    statistic_data.fill(np.nan)
    iterator = enumerate(targets) if not progress else enumerate(tqdm(targets))
    rejection_counter = 0
    for ix, (target_unit, target_intervention) in iterator:
        target_action = target_intervention if regression_dim == 'intervention' else target_unit
        target_context = target_unit if regression_dim == 'intervention' else target_intervention

        # no target data on which to learn the regression model
        try:
            # find donor units
            training_contexts, donor_actions = find_donors(
                sorted_donors,
                contexts2available_actions,
                target_context,
                target_action,
                num_desired_donors
            )

            # get donor source/target
            donor_x = df[
                df.index.get_level_values(context_dim).isin(training_contexts) &
                df.index.get_level_values(regression_dim).isin(donor_actions)
            ]
            # donor_source = donor_source.sort_index()  # 14%
            donor_x_values = donor_x.values.reshape(len(donor_actions), -1).T
            assert donor_x_values.shape == (num_features*len(training_contexts), len(donor_actions))
            donor_y = df[
                df.index.get_level_values(context_dim).isin(training_contexts) &
                (df.index.get_level_values(regression_dim) == target_action)
            ]
            # donor_target = donor_target.sort_index()  # 10%
            donor_y_values = donor_y.values.flatten()
            assert donor_y_values.shape == (num_features*len(training_contexts), )

            # perform regression
            regressor.fit(donor_x_values, donor_y_values)

            # predict
            target_x = df[
                (df.index.get_level_values(context_dim) == target_context) &
                (df.index.get_level_values(regression_dim).isin(donor_actions))
            ]
            # target_source = target_source.sort_index()  # 12%
            target_x_values = target_x.values.T

            if isinstance(regressor, HSVTRegressor):
                prediction, stat = regressor.predict(target_x_values)
            else:
                prediction = regressor.predict(target_x_values)

            predicted_data[ix] = prediction
        except (NoContextsWithTargetAction, RejectionError) as err:
            target_context_data = df[df.index.get_level_values(context_dim) == target_context].values
            prediction = predictor_no_training(target_context_data)
            predicted_data[ix] = prediction
            if isinstance(err, RejectionError):
                statistic_data[ix] = err.stat
            else:
                print("no training perturbations for target cell type")
            # ipdb.set_trace()
        except NoDonorActionsWithTargetContext as e:
            predicted_data[ix] = default_prediction
            print("no donor data")

    predicted_df = pd.DataFrame(predicted_data, index=targets, columns=df.columns)
    statistic_df = pd.DataFrame(statistic_data, index=targets, columns=['statistic'])
    return predicted_df, statistic_df


def predict_synthetic_intervention_ols(
        df,
        targets,
        num_desired_donors,
        progress=False,
        donor_dim='intervention'
):
    regressor = LinearRegression()

    predictor_no_training = partial(np.mean, axis=0)
    default_prediction = df.values.mean(axis=0)

    predicted_df, _ = synthetic_intervention_inner(
        df,
        targets,
        regressor,
        predictor_no_training,
        default_prediction,
        num_desired_donors=num_desired_donors,
        regression_dim=donor_dim,
        progress=progress
    )
    return predicted_df


def predict_synthetic_intervention_hsvt_ols(
        df,
        targets,
        num_desired_donors,
        progress=False,
        center=True,
        energy=.99,
        hypo_test=True,
        sig_level=.05,
        hypo_test_percent=None,
        donor_dim='intervention'
):
    regressor = HSVTRegressor(
        center=center,
        energy=energy,
        sig_level=sig_level,
        hypo_test=hypo_test,
        hypo_test_percent=hypo_test_percent
    )

    predictor_no_training = partial(np.mean, axis=0)
    default_prediction = df.values.mean(axis=0)

    predicted_df, statistic_df = synthetic_intervention_inner(
        df,
        targets,
        regressor,
        predictor_no_training,
        default_prediction,
        num_desired_donors=num_desired_donors,
        regression_dim=donor_dim,
        progress=progress
    )

    return predicted_df, statistic_df



