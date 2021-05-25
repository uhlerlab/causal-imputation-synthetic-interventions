import numpy as np
from numpy.linalg import pinv
from numpy import sqrt, log
# from fancyimpute import SoftImpute
from scipy.optimize import linprog
import ipdb


def get_spectral_alignment(svd_train, svd_test, energies=None):
    if energies is None:
        energies = np.linspace(.01, 1, 100)
    umat_train, spectra_train, vmat_train = svd_train
    umat_test, spectra_test, vmat_test = svd_test

    stats = np.empty(len(energies))
    for ix, energy in enumerate(energies):
        umat_test_trunc, spectra_test_trunc, vmat_test_trunc, _ = hsvt_from_svd(umat_test, spectra_test, vmat_test, energy)
        umat_train_trunc, spectra_train_trunc, vmat_train_trunc, _ = hsvt_from_svd(umat_train, spectra_train, vmat_train, energy)
        # proj_stat = projection_stat(vmat_train_trunc, vmat_test) / vmat_test.shape[0]
        proj_stat_trunc = projection_stat(vmat_train_trunc, vmat_test_trunc) / vmat_test_trunc.shape[0]
        stats[ix] = proj_stat_trunc
    return stats


def spectra2energy_percent(spectra):
    spectra_sq = spectra ** 2
    spectra_total_energy = spectra_sq.cumsum()
    percent_energy = spectra_total_energy / spectra_total_energy[-1]
    return percent_energy


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
    return hsvt_from_svd(u, spectra, v, energy, max_rank=max_rank)


def hsvt_from_svd(u, spectra, v, energy, max_rank=None):
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


class HSVTRegressor:
    def __init__(
            self,
            energy: float = 0.95,
            compute_stats=False
    ):
        self.energy = energy
        self.compute_stats = compute_stats

        self.intercept_ = None
        self.coef_ = None
        self.train_error = None

        self.umat_train = None
        self.spectra_train = None
        self.vmat_train = None

    def fit(self, train_x, train_y):
        x_mean = train_x.mean(axis=0)
        y_mean = train_y.mean()
        train_x = train_x - x_mean
        train_y = train_y - y_mean
        if self.energy == 1:
            self.coef_ = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
            self.intercept_ = y_mean - np.sum(self.coef_ * x_mean)
            self.train_error = np.sqrt(np.sum((train_y - train_x @ self.coef_ - self.intercept_)**2)) / train_x.shape[0]
        else:
            umat, spectra, vmat, _ = hsvt(train_x, energy=self.energy)
            self.umat_train = umat
            self.spectra_train = spectra
            self.vmat_train = vmat

            inv_source = (vmat.T / spectra) @ umat.T
            self.coef_ = inv_source @ train_y
            self.intercept_ = y_mean - np.sum(self.coef_ * x_mean)

    def predict(self, test_x):
        return test_x @ self.coef_ + self.intercept_

    def get_train_svd(self, train_x):
        if self.umat_train is None:
            u, s, v = np.linalg.svd(train_x, full_matrices=False)
            self.umat_train, self.spectra_train, self.vmat_train = u, s, v
        return self.umat_train, self.spectra_train, self.vmat_train

    def projection_stat(self, train_x, test_x, energy=.99):
        umat_train_trunc, spectra_train_trunc, vmat_train_trunc, _ = hsvt_from_svd(*self.get_train_svd(train_x), energy)
        umat_test, spectra_test, vmat_test = np.linalg.svd(test_x, full_matrices=False)
        umat_test_trunc, spectra_test_trunc, vmat_test_trunc, _ = hsvt_from_svd(umat_test, spectra_test, vmat_test, energy)
        proj_stat = projection_stat(vmat_train_trunc, vmat_test) / vmat_test.shape[0]
        proj_stat_trunc = projection_stat(vmat_train_trunc, vmat_test_trunc) / vmat_test_trunc.shape[0]
        # print(spectra2energy_percent(self.spectra_train))
        # print(proj_stat, proj_stat_trunc)
        rank_train = len(spectra_train_trunc)
        rank_test = len(spectra_test_trunc)
        align = get_spectral_alignment(self.get_train_svd(train_x), (umat_test, spectra_test, vmat_test))
        # print(np.mean(align))
        return np.mean(align), rank_train, rank_test
        # ipdb.set_trace()


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

