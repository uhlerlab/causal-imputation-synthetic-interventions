import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.cvxRegression import ConvexRegression

# ordinary least squares 
def linear_regression(X, y, rcond=1e-15):
	"""
	Input:
		X: pre-int. donor data (#pre-int. samples x #donor units)
		y: pre-int. target data (#pre-int. samples x 1)

	Output:
		synthetic control (regression coefficients)
	"""
	return np.linalg.pinv(X, rcond=rcond).dot(y)

# convex regression 
def cvx_regression(X, y):
	cvxr = ConvexRegression(X, y)
	return cvxr.x

# ridge regression 
def ridge(X, y, alpha=1.0): 
	regr = Ridge(alpha=alpha)
	regr.fit(X, y)
	return regr.coef_
