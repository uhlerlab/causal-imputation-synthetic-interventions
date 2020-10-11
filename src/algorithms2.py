import pandas as pd
import numpy as np
from numpy.linalg import pinv
import itertools as itr
import ipdb
from collections import Counter, defaultdict
from utils import get_index_dict, get_top_available
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from functools import partial
from sklearn.decomposition import TruncatedSVD
from tensorly import partial_svd
import itertools as itr
from numpy import sqrt, log


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


def predict_unit_fixed_effect(df, targets, control_unit):
    pass


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
            assert donor_target_values.shape == (num_features*len(training_dim), )

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


def synthetic_control_unit_inner(df, targets, regression_function, default_predictor, num_desired_interventions=None, progress=False):
    # make a dictionary mapping each unit to the `num_desired_interventions` most popular interventions, which will
    # be used as the "donor" interventions for learning weights

    df = df.sort_index(level=['intervention', 'unit'])

    units2available_ivs = get_index_dict(df, 'unit')
    c = Counter(df.index.get_level_values('intervention'))
    sorted_ivs = [iv for iv, _ in c.most_common()]

    num_features = df.shape[1]
    predicted_data = np.zeros((len(targets), num_features))
    iterator = enumerate(targets) if not progress else enumerate(tqdm(targets))
    rejection_counter = 0
    for ix, (target_unit, target_intervention) in iterator:
        if target_intervention not in sorted_ivs:
            target_source_values = df[df.index.get_level_values('unit') == target_unit].values
            prediction = default_predictor(target_source_values)
            predicted_data[ix] = prediction
        else:
            # find donor units
            donor_units, source_interventions = find_donors(sorted_ivs, units2available_ivs, target_unit, target_intervention, num_desired_interventions)

            # get donor source/target
            donor_source = df[
                df.index.get_level_values('unit').isin(donor_units) &
                df.index.get_level_values('intervention').isin(source_interventions)
            ]
            # donor_source = donor_source.sort_index()  # 14%
            donor_source_values = donor_source.values.reshape(len(source_interventions), -1).T
            assert donor_source_values.shape == (num_features*len(donor_units), len(source_interventions))
            donor_target = df[
                df.index.get_level_values('unit').isin(donor_units) &
                (df.index.get_level_values('intervention') == target_intervention)
            ]
            # donor_target = donor_target.sort_index()  # 10%
            donor_target_values = donor_target.values.flatten()
            assert donor_target_values.shape == (num_features*len(donor_units), )

            # perform regression
            regression_function.fit(donor_source_values, donor_target_values)
            # print('donor source')
            # print(donor_source)
            # print('donor target')
            # print(donor_target)

            # predict
            target_source = df[
                (df.index.get_level_values('unit') == target_unit) &
                (df.index.get_level_values('intervention').isin(source_interventions))
            ]
            # target_source = target_source.sort_index()  # 12%
            target_source_values = target_source.values.T

            try:
                prediction = regression_function.predict(target_source_values)
            except RejectionError:
                all_target_source_values = df[df.index.get_level_values('unit') == target_unit].values
                prediction = default_predictor(all_target_source_values)
                rejection_counter += 1

            predicted_data[ix] = prediction
            # print('target source')
            # print(target_source)

    print(f"Rejected: {rejection_counter} out of {len(targets)}")
    predicted_df = pd.DataFrame(predicted_data, index=targets, columns=df.columns)
    return predicted_df


def predict_synthetic_control_unit_ols(df, targets, num_desired_interventions, progress=False):
    regression_function = LinearRegression()
    default_predictor = partial(np.mean, axis=0)
    predicted_df = synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        default_predictor,
        num_desired_interventions,
        progress=progress
    )
    return predicted_df


def predict_synthetic_control_unit_hsvt_ols(
        df,
        targets,
        num_desired_interventions,
        progress=False,
        center=True,
        energy=.99,
        hypo_test=True,
        sig_level=.05,
        hypo_test_percent=None
):
    regression_function = HSVTRegressor(center=center, energy=energy, sig_level=sig_level, hypo_test=hypo_test, hypo_test_percent=hypo_test_percent)
    default_predictor = partial(np.mean, axis=0)
    predicted_df = synthetic_intervention_inner(
        df,
        targets,
        regression_function,
        default_predictor,
        num_desired_interventions,
        progress=progress
    )
    return predicted_df


def approximate_rank(spectra, energy):
    spectra_sq = spectra ** 2
    spectra_total_energy = spectra_sq.cumsum()
    percent_energy = spectra_total_energy / spectra_total_energy[-1]
    is_above = percent_energy > energy
    rank = next((ix for ix, val in enumerate(is_above) if val)) + 1
    captured_energy = spectra_total_energy[rank-1]
    return rank, spectra_total_energy[-1], captured_energy


def hsvt(values, energy):
    u, spectra, v = np.linalg.svd(values, full_matrices=False)
    rank, total_energy, captured_energy = approximate_rank(spectra, energy)
    uncaptured_energy = total_energy - captured_energy
    return u[:, :rank], spectra[:rank], v[:rank], uncaptured_energy


def critical_value(sigma, r1, r2, num_donor_units, num_dimensions1, num_dimensions2, alpha):
    n_d = num_donor_units
    t0 = num_dimensions1
    t1 = num_dimensions2
    term1 = sigma**2 * r1 * r2 * (sqrt(t0) + sqrt(n_d) + sqrt(log(1/alpha)))**2 / n_d / t0
    term2 = sigma**2 * r2**2 * (sqrt(t1) + sqrt(n_d) + sqrt(log(1/alpha)))**2 / n_d / t1
    term3 = sigma * sqrt(r1) * r2 * (sqrt(t1) + sqrt(n_d) + sqrt(log(1/alpha))) / sqrt(n_d * t0)
    return term1 + term2 + term3


# check if row space of X2 lies within row space of X1 (look at right singular vectors)
def projection_stat(v1, v2):
    # training projection matrix
    P = v1.T @ v1

    # gap in subspaces
    delta = v2 @ P - v2
    return np.linalg.norm(delta, 'fro') ** 2


class HSVTRegressor:
    def __init__(self, center=True, energy=.95, sig_level=.05, hypo_test=True, hypo_test_percent=None):
        self.center = center
        self.energy = energy
        self.coef_ = None
        self.bias = None

        # parameters needed for hypothesis test
        self.sig_level = sig_level
        self.hypo_test = hypo_test
        self.hypo_test_percent = hypo_test_percent
        self.vmat_source = None
        self.num_shared_ivs = None
        self.t0 = None
        self.sigma = None

    def fit(self, source_values, target_values):
        # each column should correspond to a single intervention
        if self.center:
            source_values = source_values - source_values.mean(axis=0)
            self.bias = target_values.mean()
            target_values = target_values - self.bias

        u_mat, spectra, v_mat, uncaptured_energy = hsvt(source_values, self.energy)
        if self.hypo_test:
            self.vmat_source = v_mat
            self.num_shared_ivs = source_values.shape[1]
            self.t0 = source_values.shape[0]
            self.sigma = np.sqrt(1 / self.t0 / self.num_shared_ivs * uncaptured_energy)

        inv_source = (v_mat.T / spectra) @ u_mat.T
        self.coef_ = inv_source @ target_values

    def predict(self, source_values):
        if self.center:
            source_values = source_values - source_values.mean(axis=0)
        u_mat, spectra, v_mat, _ = hsvt(source_values, self.energy)

        if self.hypo_test:
            stat = projection_stat(self.vmat_source, v_mat)
            num_dimensions = source_values.shape[0]
            r1 = self.vmat_source.shape[0]
            r2 = v_mat.shape[0]

            if self.hypo_test_percent is None:
                critval = critical_value(self.sigma, r1, r2, self.num_shared_ivs, self.t0, num_dimensions, self.sig_level)
            else:
                critval = r2 * self.hypo_test_percent
            print(stat, critval)
            print(f"stat={stat}, critval={critval}, sigma={self.sigma}, r1={r1}, r2={r2}, n_d={self.num_shared_ivs}, t0={self.t0}, t1={num_dimensions}")
            if stat > critval:
                raise RejectionError

        source_values = (u_mat*spectra) @ v_mat
        res = source_values @ self.coef_
        if self.center:
            res += self.bias
        return res


class RejectionError(Exception):
    pass
