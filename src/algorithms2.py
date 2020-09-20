import pandas as pd
import numpy as np
import itertools as itr
import ipdb
from collections import Counter


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


def predict_unit_fixed_effect(df, units, interventions, control_unit):
    pass


def predict_synthetic_control_unit(df, targets):
    # for each target, find other units which have received the desired intervention, and fit coefficients using control data

    # then, predict the value using those coefficients
    pass
