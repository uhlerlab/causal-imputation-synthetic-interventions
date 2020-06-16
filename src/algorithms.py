import numpy as np
from src.matrix import center_data, approximate_rank,approximate_rank2, hsvt
from src.regression import linear_regression


# HSVT + OLS 
def hsvt_ols(X1, X2, y1, t=0.99, center=True, rcond=1e-15, alpha=0.05, include_pre=True, method = 'spectral_energy', return_coefficients = False):
	"""
	Input:
		X1: pre-int. donor data (#pre-int samples x #units)
		X2: post-int. donor data (#post-int samples x #units)
		y1: pre-int. target data (#pre-int samples x 1)
		t: level of spectral energy to retain
		center: binary value indicating whether to center pre-int. data

	Output:
		counterfactual estimate of target unit in post-int. regime
	"""
	# if there are no donors, then don't run method 
	if X1.shape[1] == 0:
		# return None
		# return (baseline(np.concatenate([X1, X2])),0) if include_pre else (baseline(X2),0)
		return baseline(X2) 

	# center training data
	c = np.zeros(y1.shape[1])
	if center: 
		X1, _ = center_data(X1) 
		y1, c = center_data(y1) 

	# ranks
	if method == 'Donoho':
		k1 = approximate_rank2(X1)
		k2 = approximate_rank2(X2)
	else:	
		k1 = approximate_rank(X1, t=t)
		k2 = approximate_rank(X2, t=t)
	# de-noise donor matrices
	X1 = hsvt(X1, rank=k1)
	X2 = hsvt(X2, rank=k2)

	# learn synthetic control via linear regression
	beta = linear_regression(X1, y1, rcond=rcond)
	#print(beta.shape)
	# forecast counterfactuals
	y2 = X2.dot(beta).T
	X1_centered = X1.dot(beta).T + c[:, None]
	#print(X1_centered.shape, y2.shape)
	return y2.T
	yh = np.concatenate([X1_centered, y2]) if include_pre else y2
	if return_coefficients: return yh, beta
	return yh 


# baseline (simple average)
def baseline(X2):
	return X2.mean(axis=1)
