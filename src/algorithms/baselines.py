import pandas as pd
import numpy as np
from collections import Counter


def fill_missing_means(df, missing_ixs):
    if len(missing_ixs) == 0:
        return df

    overall_mean = df.values.mean(axis=0)
    new_data = np.tile(overall_mean, (len(missing_ixs), 1))
    new_df = pd.DataFrame(
        np.vstack((new_data, df.values)),
        index=list(missing_ixs)+list(df.index),
        columns=df.columns
    )
    return new_df


def impute_unit_mean(df, targets: pd.MultiIndex):
    targets = targets.sortlevel('unit')[0]
    units = Counter(targets.get_level_values('unit'))

    unit_means = df.groupby(level='unit').mean()
    # if any target units *don't* have any samples in `df`, then use the average over all cells
    unit_means = fill_missing_means(unit_means, set(units) - set(unit_means.index))

    # build an array to hold the correct number of copies of each unit mean
    imputed_data = np.zeros((len(targets), df.shape[1]))
    ix = 0
    for unit, num_ivs in units.items():
        imputed_data[ix:(ix+num_ivs), :] = unit_means.loc[unit]
        ix += num_ivs

    imputed_df = pd.DataFrame(imputed_data, index=targets, columns=df.columns)

    imputed_df.sort_index(inplace=True)
    return imputed_df


def impute_intervention_mean(df, targets: pd.MultiIndex):
    targets = targets.sortlevel('intervention')[0]
    interventions = Counter(targets.get_level_values('intervention'))
    iv_means = df.groupby(level='intervention').mean()
    iv_means = fill_missing_means(iv_means, set(interventions) - set(iv_means.index))

    imputed_data = np.zeros((len(targets), df.shape[1]))
    ix = 0
    for iv, num_units in interventions.items():
        imputed_data[ix:(ix+num_units), :] = iv_means.loc[iv]
        ix += num_units

    imputed_df = pd.DataFrame(imputed_data, index=targets, columns=df.columns)
    # imputed_df.update(df)

    imputed_df.sort_index(inplace=True)
    return imputed_df


def impute_two_way_mean(df, targets, lam=.5):
    unit_mean_df = impute_unit_mean(df, targets)
    intervention_mean_df = impute_intervention_mean(df, targets)
    imputed_df = lam*unit_mean_df + (1 - lam)*intervention_mean_df
    return imputed_df


def predict_intervention_fixed_effect(df, targets, control_intervention):
    control_df = df[df.index.get_level_values('intervention') == control_intervention]
    control_df.reset_index(level='intervention', drop=True, inplace=True)

    target_units, target_ivs = zip(*targets)
    target_iv_set = set(target_ivs)
    intervention_effects = df[df.index.get_level_values('intervention').isin(target_iv_set)].subtract(control_df, level='unit')
    average_intervention_effects = intervention_effects.groupby('intervention').mean()
    missing_interventions = target_iv_set - set(average_intervention_effects.index.get_level_values('intervention'))
    if len(missing_interventions) > 0:
        missing_intervention_df = pd.DataFrame(
            np.zeros((len(missing_interventions), average_intervention_effects.shape[1])),
            index=list(missing_interventions),
            columns=average_intervention_effects.columns
        )
        average_intervention_effects = pd.concat((average_intervention_effects, missing_intervention_df))
    assert average_intervention_effects.shape == (len(target_iv_set), control_df.shape[1])

    predicted_data = control_df.loc[list(target_units)].values + average_intervention_effects.loc[list(target_ivs)].values
    predicted_df = pd.DataFrame(predicted_data, index=targets, columns=control_df.columns)

    return predicted_df
