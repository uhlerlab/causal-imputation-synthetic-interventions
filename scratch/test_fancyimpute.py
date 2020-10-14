import numpy as np
from fancyimpute import SoftImpute, IterativeSVD, IterativeImputer, MatrixFactorization, NuclearNormMinimization

a = np.random.rand(10, 2)
b = np.random.rand(10, 2)
m_true = a @ b.T
m = m_true + np.random.normal(0, .01, size=(10, 10))

s = SoftImpute()
m_est = s.fit_transform(m)
print(np.linalg.norm(m_true - m, 'fro'))
print('SoftImpute', np.linalg.norm(m_true - m_est, 'fro'))

# s = MatrixFactorization(rank=2)
# m_est = s.fit_transform(m)
# print(np.linalg.norm(m_true - m, 'fro'))
# print('MatrixFactorization', np.linalg.norm(m_true - m_est, 'fro'))

s = IterativeSVD(rank=2)
m_est = s.fit_transform(m)
print(np.linalg.norm(m_true - m, 'fro'))
print('IterativeSVD', np.linalg.norm(m_true - m_est, 'fro'))

s = IterativeImputer()
m_est = s.fit_transform(m)
print(np.linalg.norm(m_true - m, 'fro'))
print('IterativeImputer', np.linalg.norm(m_true - m_est, 'fro'))

s = NuclearNormMinimization()
m_est = s.fit_transform(m)
print(np.linalg.norm(m_true - m, 'fro'))
print('NuclearNormMinimization', np.linalg.norm(m_true - m_est, 'fro'))
