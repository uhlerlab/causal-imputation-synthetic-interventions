import pandas as pd
import numpy as np
from collections import Counter
from utils import get_index_dict
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from functools import partial
from src.algorithms import HSVTRegressor, RejectionError
import ipdb


class NoTrainingData(Exception):
    pass


class NoDonorData(Exception):
    pass


def find_donors(
        sorted_donors,
        training2available_donors,
        target_training_ix,
        target_donor_ix,
        num_desired_donors=None
):
    donors_available_for_target = [
        donor for donor in sorted_donors
        if donor in training2available_donors[target_training_ix]
    ]
    if len(donors_available_for_target) == 0:
        raise NoDonorData

    # start with training data which has the target_donor_ix available
    current_training_ixs = [
        t for t, available_donor_dims in training2available_donors.items()
        if target_donor_ix in available_donor_dims
    ]
    if len(current_training_ixs) == 0:
        raise NoTrainingData

    donors = set()
    while True:
        if len(donors_available_for_target):
            break
        next_donor = donors_available_for_target.pop()

        # stop adding donors if it would make the number of training dimensions zero
        new_training_ixs = [
            train_ix for train_ix in current_training_ixs
            if next_donor in training2available_donors[train_ix]
        ]
        if len(new_training_ixs) == 0:
            break

        # stop adding donors if we've reached the desired number of donors, and adding another
        # donor would decrease the number of training dimensions
        reached_num_desired = len(donors) == num_desired_donors
        if reached_num_desired and len(new_training_ixs) < len(current_training_ixs):
            break

        # otherwise, it's fine to add another donor
        current_training_ixs = new_training_ixs
        donors.add(next_donor)

    assert len(current_training_ixs) > 0
    assert len(donors) > 0
    return current_training_ixs, donors


def synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        predictor_no_training,
        default_prediction,
        donor_dim='intervention',
        num_desired_donors=None,
        progress=False
):
    assert donor_dim == 'intervention' or donor_dim == 'unit'
    training_dim = 'unit' if donor_dim == 'intervention' else 'unit'

    df = df.sort_index(level=[donor_dim, training_dim])

    training2available_donors = get_index_dict(df, training_dim)
    counter = Counter(df.index.get_level_values(donor_dim))
    sorted_donors = [donor for donor, _ in counter.most_common()]

    num_features = df.shape[1]
    predicted_data = np.zeros((len(targets), num_features))
    iterator = enumerate(targets) if not progress else enumerate(tqdm(targets))
    rejection_counter = 0
    for ix, (target_unit, target_intervention) in iterator:
        target_donor_ix = target_intervention if donor_dim == 'intervention' else target_unit
        target_training_ix = target_unit if donor_dim == 'intervention' else target_intervention

        # no target data on which to learn the regression model
        try:
            # find donor units
            training_ixs, donor_ixs = find_donors(
                sorted_donors,
                training2available_donors,
                target_training_ix,
                target_donor_ix,
                num_desired_donors
            )

            # get donor source/target
            donor_source = df[
                df.index.get_level_values(training_dim).isin(training_ixs) &
                df.index.get_level_values(donor_dim).isin(donor_ixs)
            ]
            # donor_source = donor_source.sort_index()  # 14%
            donor_source_values = donor_source.values.reshape(len(donor_ixs), -1).T
            assert donor_source_values.shape == (num_features*len(training_ixs), len(donor_ixs))
            donor_target = df[
                df.index.get_level_values(training_dim).isin(training_ixs) &
                (df.index.get_level_values(donor_dim) == target_donor_ix)
            ]
            # donor_target = donor_target.sort_index()  # 10%
            donor_target_values = donor_target.values.flatten()
            assert donor_target_values.shape == (num_features*len(training_ixs), )

            # perform regression
            regression_function.fit(donor_source_values, donor_target_values)

            # predict
            target_source = df[
                (df.index.get_level_values(training_dim) == target_training_ix) &
                (df.index.get_level_values(donor_dim).isin(donor_ixs))
            ]
            # target_source = target_source.sort_index()  # 12%
            target_source_values = target_source.values.T

            prediction = regression_function.predict(target_source_values)
            predicted_data[ix] = prediction
        except (NoTrainingData, RejectionError):
            target_source_values = df[df.index.get_level_values(training_dim) == target_training_ix].values
            prediction = predictor_no_training(target_source_values)
            predicted_data[ix] = prediction
        except NoDonorData:
            predicted_data[ix] = default_prediction

    print(f"Rejected: {rejection_counter} out of {len(targets)}")
    predicted_df = pd.DataFrame(predicted_data, index=targets, columns=df.columns)
    return predicted_df


def predict_synthetic_intervention_ols(
        df,
        targets,
        num_desired_donors,
        progress=False,
        donor_dim='intervention'
):
    regression_function = LinearRegression()

    predictor_no_training = partial(np.mean, axis=0)
    default_prediction = df.values.mean(axis=0)

    predicted_df = synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        predictor_no_training,
        default_prediction,
        num_desired_donors=num_desired_donors,
        donor_dim=donor_dim,
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
    regression_function = HSVTRegressor(
        center=center,
        energy=energy,
        sig_level=sig_level,
        hypo_test=hypo_test,
        hypo_test_percent=hypo_test_percent
    )

    predictor_no_training = partial(np.mean, axis=0)
    default_prediction = df.values.mean(axis=0)

    predicted_df = synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        predictor_no_training,
        default_prediction,
        num_desired_donors=num_desired_donors,
        donor_dim=donor_dim,
        progress=progress
    )
    return predicted_df



