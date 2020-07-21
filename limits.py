import h5py
import numpy as np

def cum_post(elem_number, D_cov, D_cov_limit, ks, ks_limit, sigma):
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
	
	D_cov_inds = np.argwhere(D_cov[:,elem_number] <= D_cov_limit)
	ks_inds = np.argwhere(ks[:,elem_number] <= ks_limit)
	all_inds = np.intersect1d(D_cov_inds, ks_inds)
	wanted_sigma = sigma[all_inds]
	cum_sigma = np.sort(wanted_sigma)
	y_ax = np.linspace(0, 1, len(cum_sigma))
	return y_ax, cum_sigma

def ABC(Dcov_all, ks_all, sigma_all):
	"""Return the 68% and 95% upper limits on the intrinsic abundance scatter in an open cluster.
	
	Parameters
	----------
	Dcov_all : tuple
		Array of all covariance matrix summary statistics being considered
	ks_all : tuple
		Array of all KS distance summary statistics being considered
	sigma_all : tuple
		Array of all sigma values being considered
	
	Returns
	-------
	limit_95 : tuple
		The 95% upper limits on the intrinsic abundance scatter for each element in the open cluster being considered
	limit_68 : float
		The 68% upper limits on the intrinsic abundance scatter for each element in the open cluster being considered
	"""
	
	elem_dict = {'element': ['C', 'N', 'O', 'NA', 'MG', 'AL', 'SI', 'S', 'K', 'CA', 'TI', 'V', 'MN', 'FE', 'NI']}
	
	#Sort Dcov and KS
	Dcov_sorted = np.zeros_like(Dcov_all)
	ks_sorted = np.zeros_like(ks_all)
	for i in range(len(elem_dict['element'])):
		Dcov_sorted[:,i] = np.sort(Dcov_all[:,i])
		ks_sorted[:,i] = np.sort(ks_all[:,i])
		
	#Cut on the 1000-smallest summary statistics
	D_cov_limit = Dcov_sorted[1000]
	ks_limit = ks_sorted[1000]
	
	#Get cumulative posterior distributions
	y_ax_all = []
	cum_sigma_all = []
	elem_numbers = np.arange(0, 15)
	for i in range(len(elem_dict['element'])):
		post_data = cum_post(elem_numbers[i], Dcov_all, D_cov_limit[i], ks_all, ks_limit[i], sigma_all)
		y_ax_all.append(post_data[0])
		cum_sigma_all.append(post_data[1])
	y_ax_all = np.array(y_ax_all)
	cum_sigma_all = np.array(cum_sigma_all)
	
	#Get 68% and 95% of sigma within the cuts
	sigma_95 = []
	sigma_68 = []
	for i in range(len(elem_dict['element'])):
		sigma_95.append(cum_sigma_all[i][int(len(cum_sigma_all[i])*0.95):])
		sigma_68.append(cum_sigma_all[i][int(len(cum_sigma_all[i])*0.68):])
	sigma_95 = np.array(sigma_95)
	sigma_68 = np.array(sigma_68)
	
	#Get 68% and 95% upper limits
	limit_95 = np.zeros(15)
	limit_68 = np.zeros(15)
	for i in range(len(elem_dict['element'])):
		limit_95[i] = sigma_95[i][0]
		limit_68[i] = sigma_68[i][0]
	
	return limit_95, limit_68