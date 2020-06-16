import numpy as np
import statsmodels.api as sm
from src.matrix import approximate_rank


# check if row space of X2 lies within row space of X1 (look at right singular vectors)
def regression_test(v1, v2, alpha=0.05):
    for i in range(v2.shape[1]):
        model = sm.OLS(v2[:, i], v1)
        results = model.fit()
        pvalues = results.pvalues
        result = True in (pvalue < alpha for pvalue in pvalues)
        if not result:
            return False
    return True


# check if row space of X2 lies within row space of X1 (look at right singular vectors)
# incomplete...
def energy_test(v1, v2, alpha=0.05):
    P = v1.dot(v1.T)
    delta = v2 - P.dot(v2)
    return np.linalg.norm(delta, 'fro') ** 2


# diagnostic test
def diagnostic_test(pre_df, post_df, unit_ids, metric, iv, t=0.99, alpha=0.05):
    columns = ['unit', 'intervention', 'metric']

    # get dimensions
    N = len(unit_ids)
    M = int(pre_df.loc[pre_df.unit.isin(unit_ids)].shape[0] / N)
    T0 = pre_df.drop(columns=columns).shape[1]
    T1 = post_df.drop(columns=columns).shape[1]

    # pre-int data
    X1 = pre_df.loc[pre_df.unit.isin(unit_ids)]
    X1 = X1.drop(columns=columns).values.reshape(N, M * T0).T

    # post-int data
    X2 = post_df.loc[(post_df.unit.isin(unit_ids)) & (post_df.intervention == iv) & (post_df.metric == metric)]
    X2 = X2.drop(columns=columns).values.T

    # compute row spaces of X1 and X2 (top right singular vectors)
    k1 = approximate_rank(X1, t=t)
    k2 = approximate_rank(X2, t=t)
    _, _, v1 = np.linalg.svd(X1, full_matrices=False)
    _, _, v2 = np.linalg.svd(X2, full_matrices=False)
    v1 = v1[:k1, :].T
    v2 = v2[:k2, :].T

    # estimate sigma
    # beta = linear_regression(X1, y1, rcond=rcond)

    # perform regression test
    regression_rslt = regression_test(v1, v2, alpha=alpha)

    # perform energy test
    inner = (k1 + k2) / N + (k1 / T0 + k2 / T1) * (1 + np.log(1 / alpha) / N)
    # t = 8*k2*sigma**2*(inner)
    energy_rslt = energy_test(v1, v2, alpha=alpha)
    return regression_rslt, energy_rslt
