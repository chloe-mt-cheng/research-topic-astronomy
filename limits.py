import h5py
import numpy as np

def cum_post(D_cov, D_cov_limit, ks, ks_limit, sigma):
	"""Return the normalized y-axis and cumulative posterior distribution.
	
	Parameters
	----------
	elem_number : int
		The element being considered
	D_cov : tuple
		Array of numbers representing the covariance matrix summary statistics
	D_cov_limit : float
		Value of the covariance matrix statistic on which to cut
	ks : tuple
		Array of numbers representing the KS distance summary statistics
	ks_limit : float
		Value of the KS distance statistic on which to cut
	sigma : tuple
		Array of all sigma being considered
	
	Returns
	-------
	y_ax : tuple
		Array of numbers from 0 to 1 that is the same length as the cumulative posterior distribution
	cum_sigma : tuple
		Array of the sorted values of sigma within the summary statistic cuts
	"""
	
	#Get the indices of the summary statistics that are below the chosen cut 
	D_cov_inds = np.argwhere(D_cov <= D_cov_limit)
	ks_inds = np.argwhere(ks <= ks_limit)
	
	#Get the union of these indices to get the indices corresponding to where both summary statistics are close to zero
	all_inds = np.intersect1d(D_cov_inds, ks_inds)
	#Get the values of sigma at these indices
	wanted_sigma = sigma[all_inds]
	
	#Compute the cumulative distribution of these sigma
	cum_sigma = np.sort(wanted_sigma)
	y_ax = np.linspace(0, 1, len(cum_sigma))
	
	return y_ax, cum_sigma

def ABC(Dcov, ks, sigma):
	"""Return the 68% and 95% upper limits on the intrinsic abundance scatter in an open cluster.
	
	Parameters
	----------
	Dcov : tuple
		Array of all covariance matrix summary statistics being considered
	ks : tuple
		Array of all KS distance summary statistics being considered
	sigma : tuple
		Array of all sigma values being considered
	
	Returns
	-------
	limit_95 : float
		The 95% upper limit on the intrinsic abundance scatter for each element in the open cluster being considered
	limit_68 : float
		The 68% upper limit on the intrinsic abundance scatter for each element in the open cluster being considered
	"""
	
	#Sort the summary statistics from least to greatest
	Dcov_sorted = np.sort(Dcov)
	ks_sorted = np.sort(ks)
	
	#Take the 1000 points where both summary statistics are closest to zero
	D_cov_limit = Dcov_sorted[1000]
	ks_limit = ks_sorted[1000]
	
	#Get the cumulative posterior distribution function
	y_ax, cum_sigma = cum_post(Dcov, D_cov_limit, ks, ks_limit, sigma)
	
	#Get all of the sigma at the top 95% of the posterior PDF
	sigma_95 = cum_sigma[int(len(cum_sigma)*0.95):]
	#Get all of the sigma at the top 68% of the posterior PDF
	sigma_68 = cum_sigma[int(len(cum_sigma)*0.68):]
	
	#Get the limit at 95% confidence
	limit_95 = sigma_95[0]
	#Get the limit at 68% confidence 
	limit_68 = sigma_68[0]
	
	return limit_95, limit_68