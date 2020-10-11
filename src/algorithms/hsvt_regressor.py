import numpy as np
from numpy.linalg import pinv
from numpy import sqrt, log


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
