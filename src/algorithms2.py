import pandas as pd
import numpy as np
import itertools as itr
import ipdb
from collections import Counter
from utils import get_index_dict, get_top_available
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def fill_missing_means(df, overall_mean, missing_ixs):
    missing_df = pd.DataFrame(
        np.tile(overall_mean.values, (len(missing_ixs), 1)),
        index=missing_ixs,
        columns=overall_mean.index
    )
    return df.append(missing_df)


def impute_unit_mean(df, targets: pd.MultiIndex):
    targets = targets.sortlevel('unit')[0]
    units = Counter(targets.get_level_values('unit'))
    unit_means = df.groupby(level='unit').mean()
    unit_means = fill_missing_means(unit_means, df.mean(), set(units) - set(unit_means.index))

    imputed_data = np.zeros((len(targets), df.shape[1]))
    ix = 0
    for unit, num_ivs in units.items():
        imputed_data[ix:(ix+num_ivs), :] = unit_means.loc[unit]
        ix += num_ivs

    imputed_df = pd.DataFrame(imputed_data, index=targets, columns=df.columns)
    imputed_df.update(df)

    imputed_df.sort_index(inplace=True)
    return imputed_df


def impute_intervention_mean(df, targets: pd.MultiIndex):
    targets = targets.sortlevel('intervention')[0]
    interventions = Counter(targets.get_level_values('intervention'))
    iv_means = df.groupby(level='intervention').mean()
    iv_means = fill_missing_means(iv_means, df.mean(), set(interventions) - set(iv_means.index))

    imputed_data = np.zeros((len(targets), df.shape[1]))
    ix = 0
    for iv, num_units in interventions.items():
        imputed_data[ix:(ix+num_units), :] = iv_means.loc[iv]
        ix += num_units

    imputed_df = pd.DataFrame(imputed_data, index=targets, columns=df.columns)
    imputed_df.update(df)

    imputed_df.sort_index(inplace=True)
    return imputed_df


def impute_two_way_mean(df, targets, lam=.5):
    unit_mean_df = impute_unit_mean(df, targets)
    intervention_mean_df = impute_intervention_mean(df, targets)
    imputed_df = lam*unit_mean_df + (1 - lam)*intervention_mean_df
    return imputed_df


def predict_intervention_fixed_effect(df, targets, control_intervention):
    control_df = df.query('intervention == @control_intervention')
    control_df.reset_index(level='intervention', drop=True, inplace=True)
    intervention_effects = df.subtract(control_df, level='unit')
    average_intervention_effects = intervention_effects.groupby('intervention').mean()

    target_units, target_ivs = zip(*targets)
    predicted_data = control_df.loc[list(target_units)].values + average_intervention_effects.loc[list(target_ivs)].values
    predicted_df = pd.DataFrame(predicted_data, index=targets, columns=control_df.columns)

    return predicted_df


def predict_unit_fixed_effect(df, targets, control_unit):
    pass


def synthetic_control_unit_inner(df, targets, num_desired_interventions, regression_function, progress=False):
    # make a dictionary mapping each unit to the `num_desired_interventions` most popular interventions, which will
    # be used as the "donor" interventions for learning weights

    units2available_ivs = get_index_dict(df, 'unit')
    sorted_ivs = df.groupby('intervention').size().sort_values(ascending=False)
    units2training_ivs = get_top_available(sorted_ivs.index, units2available_ivs, num_desired_interventions)

    num_features = df.shape[1]
    predicted_data = np.zeros((len(targets), num_features))
    iterator = enumerate(targets) if not progress else enumerate(tqdm(targets))
    for ix, (target_unit, target_intervention) in iterator:
        # find donor units
        source_interventions = units2training_ivs[target_unit]
        desired_interventions = source_interventions | {target_intervention}
        donor_units = [unit for unit, available_ivs in units2available_ivs.items() if desired_interventions <= available_ivs]

        # get donor source/target
        donor_source = df.loc[[(unit, iv) for iv in source_interventions for unit in donor_units]]
        donor_source_values = donor_source.values.reshape(len(source_interventions), -1).T
        assert donor_source_values.shape == (num_features*len(donor_units), len(source_interventions))
        donor_target = df.loc[[(unit, target_intervention) for unit in donor_units]]
        donor_target_values = donor_target.values.flatten()
        assert donor_target_values.shape == (num_features*len(donor_units), )

        # perform regression
        regression_function.fit(donor_source_values, donor_target_values)

        # predict
        target_source = df.loc[[(target_unit, iv) for iv in source_interventions]]
        target_source_values = target_source.values.T
        prediction = regression_function.predict(target_source_values)
        predicted_data[ix] = prediction

    predicted_df = pd.DataFrame(predicted_data, index=targets, columns=df.columns)
    return predicted_df


def predict_synthetic_control_unit(df, targets, num_desired_interventions, progress=False):
    regression_function = LinearRegression()
    predicted_df = synthetic_control_unit_inner(df, targets, num_desired_interventions, regression_function, progress=progress)
    return predicted_df
