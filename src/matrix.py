import numpy as np


# compute spectral energy (squared singular values)
def spectral_energy(s):
    return 100 * (s ** 2).cumsum() / (s ** 2).sum()


# compute approximate rank of matrix
def approximate_rank(X, t=0.99):
    """
    Input:
        X: donor data (#samples x #donor units)
        t: percentage of spectral energy to retain 

    Output:
        rank: approximate rank of X at t% of spectral energy 
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    total_energy = (s ** 2).cumsum() / (s ** 2).sum()
    try:
        rank = list((total_energy > t)).index(True) + 1
        return rank
    except:
        print(X)
        print(total_energy)


def approximate_rank2(X, t=0):
    """
    Input:
        X: donor data (#samples x #donor units)
        t: percentage of spectral energy to retain 

    Output:
        rank: approximate rank of X
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    b = X.shape[1] / X.shape[0]
    omega = 0.56 * b ** 3 - 0.95 * b ** 2 + 1.43 + 1.82 * b
    thre = omega * np.median(s)
    rank = len(s[s > thre])
    return rank


