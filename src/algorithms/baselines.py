import pandas as pd
import numpy as np
from collections import Counter
import ipdb
import miceforest as mf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from tensorly.decomposition import parafac
from scipy.sparse import csr_matrix
from itertools import chain


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


def df_to_tensor(df, targets):
    full_index = df.index.tolist() + targets.tolist()
    units, interventions = zip(*full_index)
    units, interventions = list(set(units)), list(set(interventions))
    unit2ix, iv2ix = {unit: ix for ix, unit in enumerate(units)}, {iv: ix for ix, iv in enumerate(interventions)}
    tensor = np.zeros((len(units), len(interventions), df.shape[1]))
    for ix, (unit, iv) in enumerate(df.index):
        tensor[(unit2ix[unit], iv2ix[iv])] = df.values[ix]
    return tensor, unit2ix, iv2ix


def tensor_to_df(tensor, targets, unit2ix, iv2ix):
    vals = np.zeros((len(targets), tensor.shape[2]))
    tensor2 = tensor.to_tensor()
    # TODO: Converting the whole matrix and only computing the desired fiber gives the same answer
    # TODO: no difference in speed for `p` small, but should check for larger `p`.
    for ix, (unit, iv) in enumerate(targets):
        unit_ix, iv_ix = unit2ix[unit], iv2ix[iv]
        val1 = tensor2[unit_ix, iv_ix]
        # A = tensor.factors[0][unit_ix]
        # B = tensor.factors[1][iv_ix]
        # C = tensor.factors[2]
        # val2 = (C * A * B).sum(axis=1)
        # print(np.isclose(val1, val2))
        vals[ix] = val1
    return pd.DataFrame(data=vals, index=targets)


def df_to_missing(df, targets):
    df = pd.concat((df, pd.DataFrame(index=targets)))
    df['unit'] = pd.Categorical(df.index.get_level_values("unit"))
    df['intervention'] = pd.Categorical(df.index.get_level_values("intervention"))
    return df


def df_to_missing2(df, targets):
    df = pd.concat((df, pd.DataFrame(index=targets)))
    unit_dummies = pd.get_dummies(df.index.get_level_values("unit"))
    unit_dummies.index = df.index
    iv_dummies = pd.get_dummies(df.index.get_level_values("intervention"))
    iv_dummies.index = df.index
    return pd.concat((df, unit_dummies, iv_dummies), axis=1)


def impute_mice(df, targets: pd.MultiIndex):
    new_df = df_to_missing2(df, targets)
    imp = IterativeImputer()
    completed_data = imp.fit_transform(new_df)
    return pd.DataFrame(completed_data[:, :df.shape[1]], index=new_df.index)


def impute_missforest(df, targets: pd.MultiIndex):
    new_df = df_to_missing2(df, targets)
    imp = MissForest()
    completed_data = imp.fit_transform(new_df, cat_vars=list(range(df.shape[1], new_df.shape[1])))
    return pd.DataFrame(completed_data[:, :df.shape[1]], index=new_df.index)


def impute_miceforest(df, targets: pd.MultiIndex):  # works with categorical
    new_df = df_to_missing(df, targets)
    kds = mf.KernelDataSet(new_df)
    kds.mice(3)
    completed_data = kds.complete_data()
    return completed_data.iloc[:, :df.shape[1]]


def impute_als(df, targets: pd.MultiIndex, rank=20):
    tensor, unit2ix, iv2ix = df_to_tensor(df, targets)
    res = parafac(tensor, rank=rank)
    imputed_df = tensor_to_df(res, targets, unit2ix, iv2ix)
    return imputed_df


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
