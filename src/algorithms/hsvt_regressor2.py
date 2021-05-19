import numpy as np
from numpy.linalg import pinv
from numpy import sqrt, log
# from fancyimpute import SoftImpute
from scipy.optimize import linprog
import ipdb


def approximate_rank(spectra, energy):
    spectra_sq = spectra ** 2
    spectra_total_energy = spectra_sq.cumsum()
    percent_energy = spectra_total_energy / spectra_total_energy[-1]
    is_above = percent_energy > energy
    rank = next((ix for ix, val in enumerate(is_above) if val)) + 1
    captured_energy = spectra_total_energy[rank-1]
    return rank, spectra_total_energy[-1], captured_energy


def hsvt(values, energy, max_rank=None):
    u, spectra, v = np.linalg.svd(values, full_matrices=False)
    if energy == 1:
        return u, spectra, v, 0

    rank, total_energy, captured_energy = approximate_rank(spectra, energy)
    if max_rank is not None:
        if rank > max_rank:
            print(rank, max_rank)
            rank = max_rank
            captured_energy = spectra[:rank].sum()
    uncaptured_energy = total_energy - captured_energy
    return u[:, :rank], spectra[:rank], v[:rank], uncaptured_energy


def find_cutoff(spectra, desired_energy):
    p = len(spectra)
    c = -1*np.hstack((np.ones(p), [0]))

    a_ub = np.zeros([2*p + 1, p+1])
    a_ub[:p, :p] = np.eye(p)
    a_ub[p:2*p, :p] = np.eye(p)
    a_ub[p:2*p, -1] = -1
    a_ub[-1, :-1] = 1

    b_ub = np.zeros(2*p+1)
    b_ub[:p] = spectra
    b_ub[-1] = 1-desired_energy
    print(c)
    print(a_ub)
    print(b_ub)

    res = linprog(c, a_ub, b_ub)
    return res.x


def soft_threshold(spectra, t):
    new_spectra = np.maximum(spectra - t, 0)
    new_rank = next((ix for ix, val in enumerate(new_spectra) if val == 0), len(new_spectra))
    return new_spectra, new_rank


def ssvt(values, energy):
    u, spectra, v = np.linalg.svd(values, full_matrices=False)
    total_energy = spectra.sum()
    # cutoff = find_cutoff(spectra, desired_energy=energy)
    cutoff = total_energy*(1 - energy) / len(spectra)
    new_spectra, new_rank = soft_threshold(spectra, cutoff)
    print(spectra, new_spectra)
    return u[:, :new_rank], new_spectra[:new_rank], v[:new_rank]


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


class MERegressor:
    def __init__(
        self,
        center=True,
        matrix_estimator=None
    ):
        self.center = center
        self.matrix_estimator = matrix_estimator
        self.bias = None
        self.coef_ = None

    def fit(self, donor_x, donor_y):
        if self.center:
            donor_x = donor_x - donor_x.mean(axis=0)
            self.bias = donor_y.mean()
            donor_y = donor_y - self.bias

        donor_umat, donor_spectra, donor_vmat = self.matrix_estimator(donor_x)
        inv_donor = (donor_vmat.T/donor_spectra) @ donor_umat.T
        self.coef_ = inv_donor @ donor_y

    def predict(self, target_x):
        if self.center:
            target_x = target_x - target_x.mean(axis=0)

        res = None
        if self.center:
            res += self.bias
        return res


class HSVTRegressor2:
    def __init__(
            self,
            energy: float = 0.95,
            compute_stats=False
    ):
        self.energy = energy
        self.compute_stats = compute_stats

        self.intercept_ = None
        self.coef_ = None
        self.vmat_train = None

    def fit(self, train_x, train_y):
        x_mean = train_x.mean(axis=0)
        y_mean = train_y.mean()
        train_x = train_x - x_mean
        train_y = train_y - y_mean
        if self.energy == 1:
            self.coef_ = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
            self.intercept_ = y_mean - np.sum(self.coef_ * x_mean)
        else:
            umat, spectra, vmat, _ = hsvt(train_x, energy=self.energy)
            self.vmat_train = vmat

            inv_source = (vmat.T / spectra) @ umat.T
            self.coef_ = inv_source @ train_y
            self.intercept_ = y_mean - np.sum(self.coef_ * x_mean)

    def predict(self, test_x):
        return test_x @ self.coef_ + self.intercept_

    def compute_statistics(self, train_x, test_x):
        umat_test, spectra_test, vmat_test, _ = hsvt(test_x, self.energy)
        if self.energy == 1:
            umat_train, spectra_train, vmat_train, _ = hsvt(train_x, energy=self.energy)
            self.vmat_train = vmat_train
        proj_stat = projection_stat(self.vmat_train, vmat_test)
        print(proj_stat)


class HSVTRegressor:
    def __init__(self, center=True, energy=.95, sig_level=.05, hypo_test=True, hypo_test_percent=None, verbose=True, equal_rank=False, hypo_test_override=False):
        self.center = center
        self.energy = energy
        self.coef_ = None
        self.bias = None
        self.r1 = None
        self.verbose = verbose
        self.equal_rank = equal_rank
        self.hypo_test_override = hypo_test_override

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
        self.r1 = len(spectra)
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

        if self.equal_rank:
            u_mat, spectra, v_mat, _ = hsvt(source_values, self.energy, max_rank=self.r1)
        else:
            u_mat, spectra, v_mat, _ = hsvt(source_values, self.energy)

        if self.hypo_test:
            stat = projection_stat(self.vmat_source, v_mat)
            num_dimensions = source_values.shape[0]
            r2 = v_mat.shape[0]

            if self.hypo_test_percent is None:
                critval = critical_value(self.sigma, self.r1, r2, self.num_shared_ivs, self.t0, num_dimensions, self.sig_level)
            else:
                critval = r2 * self.hypo_test_percent

            if stat > critval:
                print(f"stat={stat}, critval={critval}, sigma={self.sigma}, r1={self.r1}, r2={r2}, n_d={self.num_shared_ivs}, t0={self.t0}, t1={num_dimensions}")
            if stat > critval and self.hypo_test_override:
                raise RejectionError(stat, critval)

        source_values = (u_mat*spectra) @ v_mat
        res = source_values @ self.coef_
        if self.center:
            res += self.bias

        if self.hypo_test:
            return res, stat, critval
        else:
            return res


class RejectionError(Exception):
    def __init__(self, stat, critval):
        self.stat = stat
        self.critval = critval


if __name__ == '__main__':
    from functools import partial
    # regressor = MERegressor(matrix_estimator=partial(ssvt, energy=.95))
    # regressor.fit(np.random.random((5, 5)), np.random.random(5))

    import numpy as np
    from sklearn.linear_model import LinearRegression

    x = np.random.normal(size=(100, 3))
    y = x @ np.array([1,2,3]) + 2

    lr = LinearRegression()
    lr.fit(x, y)
    print(lr.coef_)
    print(lr.intercept_)

    lr2 = LinearRegression(fit_intercept=False)
    y_mean = y.mean()
    x_mean = x.mean(axis=0)
    lr.fit(x - x_mean, y - y_mean)
    print(lr.coef_)
    xy_mean = np.sum(lr.coef_ * x_mean)
    print(y_mean - xy_mean)

    hs = HSVTRegressor2(energy=1)
    hs.fit(x, y)
    print(hs.coef_)
    print(hs.intercept_)

    ypred = hs.predict(x)
