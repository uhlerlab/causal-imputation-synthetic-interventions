import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

num_latent = 3
num_actions = 10
num_genes = 100
num_celltypes = 2
vs = np.random.uniform(size=[num_actions, num_latent])
U = np.random.uniform(size=[num_celltypes, num_genes, num_latent])
M = np.tensordot(U, vs, axes=(2, 1))
noise = np.random.normal(scale=.1, size=M.shape)
noisy_M = M + noise

lr = LinearRegression(fit_intercept=False)
lr.fit(noisy_M[0, :, 1:], noisy_M[0, :, 0])
est1 = lr.coef_
print(est1)
print(est1 @ vs[1:] - vs[0])

g0 = noisy_M[0, :, 0]
g_init = noisy_M[0, :, 0] - noisy_M[0, :, 1:] @ est1
Ainv = np.outer(g_init, g_init)
A = inv(Ainv)
G = noisy_M[0, :, 1:]
est2 = inv(G.T @ A @ G) @ G.T @ A @ g0
D = np.diag(np.arange(1, 101))
coef3 = inv(G.T @ D @ G) @ G.T @ D  # if we use a diagonal A
est3 = coef3 @ g0
print(est2 @ vs[1:] - vs[0])
print(est3 @ vs[1:] - vs[0])

# TODO: USE A J-TEST?
