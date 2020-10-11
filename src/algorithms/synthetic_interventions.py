import pandas as pd
import numpy as np
from collections import Counter
from utils import get_index_dict
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from functools import partial
from src.algorithms import HSVTRegressor, RejectionError


def find_donors(sorted_ivs, units2available_ivs, target_unit, target_intervention, num_desired_interventions=None):
    sorted_training_ivs = (iv for iv in sorted_ivs if iv in units2available_ivs[target_unit])
    current_donors = [unit for unit, available_ivs in units2available_ivs.items() if target_intervention in available_ivs]

    source_interventions = set()
    while True:
        next_intervention = next(sorted_training_ivs, None)
        if next_intervention is None:
            break

        # stop adding interventions if it would make the number of donor units zero
        new_donors = [donor_unit for donor_unit in current_donors if next_intervention in units2available_ivs[donor_unit]]
        if len(new_donors) == 0:
            break

        # stop adding interventions if we've reached the desired number of interventions and adding another
        # intervention would decrease the number of donor units
        reached_num_desired = len(source_interventions) == num_desired_interventions
        if reached_num_desired and len(new_donors) < len(current_donors):
            break

        # otherwise, it's fine to add another intervention
        current_donors = new_donors
        source_interventions.add(next_intervention)

    assert len(current_donors) > 0
    assert len(source_interventions) > 0
    return current_donors, source_interventions


def synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        default_predictor,
        donor_dim='intervention',
        num_desired_interventions=None,
        progress=False
):
    assert donor_dim == 'intervention' or donor_dim == 'unit'
    training_dim = 'unit' if donor_dim == 'intervention' else 'unit'

    df = df.sort_index(level=[donor_dim, training_dim])

    training2available_donors = get_index_dict(df, training_dim)
    c = Counter(df.index.get_level_values(donor_dim))
    sorted_donors = [donor for donor, _ in c.most_common()]

    num_features = df.shape[1]
    predicted_data = np.zeros((len(targets), num_features))
    iterator = enumerate(targets) if not progress else enumerate(tqdm(targets))
    rejection_counter = 0
    for ix, (target_unit, target_intervention) in iterator:
        target_donor_ix = target_intervention if donor_dim == 'intervention' else 'unit'
        target_training_ix = target_unit if donor_dim == 'intervention' else 'unit'

        if target_donor_ix not in sorted_donors:
            target_source_values = df[df.index.get_level_values(training_dim) == target_training_ix].values
            prediction = default_predictor(target_source_values)
            predicted_data[ix] = prediction
        else:
            # find donor units
            training_ixs, donor_ixs = find_donors(
                sorted_donors,
                training2available_donors,
                target_training_ix,
                target_donor_ix,
                num_desired_interventions
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

            try:
                prediction = regression_function.predict(target_source_values)
            except RejectionError:
                all_target_source_values = df[df.index.get_level_values(training_dim) == target_training_ix].values
                prediction = default_predictor(all_target_source_values)
                rejection_counter += 1

            predicted_data[ix] = prediction

    print(f"Rejected: {rejection_counter} out of {len(targets)}")
    predicted_df = pd.DataFrame(predicted_data, index=targets, columns=df.columns)
    return predicted_df


def predict_synthetic_intervention_ols(df, targets, num_desired_interventions, progress=False, donor_dim='intervention'):
    regression_function = LinearRegression()
    default_predictor = partial(np.mean, axis=0)
    predicted_df = synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        default_predictor,
        num_desired_interventions=num_desired_interventions,
        donor_dim=donor_dim,
        progress=progress
    )
    return predicted_df


def predict_synthetic_intervention_hsvt_ols(
        df,
        targets,
        num_desired_interventions,
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
    default_predictor = partial(np.mean, axis=0)
    predicted_df = synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        default_predictor,
        num_desired_interventions=num_desired_interventions,
        donor_dim=donor_dim,
        progress=progress
    )
    return predicted_df



