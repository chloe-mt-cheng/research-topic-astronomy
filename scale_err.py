"""Usage: scale_err.py [-h][--cluster=<arg>][--num_sigma=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
	Number of sigma: e.g. input --num_sigma='1000'

-h  Help file
--cluster=<arg>  Cluster name
--num_sigma=<arg> Number of sigma

"""

#Import project scripts
import occam_clusters_post_process as pp
import ABC
import occam_clusters_input as oc
#Import python modules
from docopt import docopt
import numpy as np
import h5py
import time

def run_everything(cluster, num_sigma):
	"""Return the covariance matrix statistics and KS distances for every element in APOGEE in the desired cluster,
	for every simulation run, with errors scaled by chosen scaling factor for each simulation.  Function also saves 
	all final summary statistics and values of sigma to file.
	
	Parameters
	----------
	cluster : str
		Name of the desired cluster (e.g. 'NGC 2682')
	num_sigma : int
		Number of simulations to run 
	
	Returns
	-------
	D_cov_all : tuple
		All covariance matrix summary statistics for all simulations
	ks_all : tuple
		All KS distances for all simulations
	"""
	
	#Create cluster directory, if it doesn't exist already
	cluster_dir = oc.make_directory(cluster)
	#Get APOGEE and spectral data
	apogee_cluster_data, spectra, spectra_errs, T, num_elem, num_stars, bitmask = ABC.get_cluster_data(cluster, 1)
	
	#Draw scaling factors
	scaled_errs = []
	scale_factors = np.random.uniform(0.5, 2, int(num_sigma))
	for i in range(len(scale_factors)):
		scaled_errs.append(spectra_errs*scale_factors[i])
	scaled_errs = np.array(scaled_errs)
	
	#Get real data and fits for each scaling factor
	real_res = []
	real_err = []
	real_nanless_res = []
	real_nanless_err = []
	real_weights = []
	y_ax_real = []
	real_cdists = []
	real_used_elems = []
	real_skipped_elems = []
	new_num_elem = []
	for i in range(len(scale_factors)):
		real_dat = ABC.real_data(num_elem, cluster, spectra, scaled_errs[i], T)
		real_res.append(real_dat[0])
		real_err.append(real_dat[1])
		real_nanless_res.append(real_dat[2])
		real_nanless_err.append(real_dat[3])
		real_weights.append(real_dat[4])
		y_ax_real.append(real_dat[5])
		real_cdists.append(real_dat[6])
		real_used_elems.append(real_dat[7])
		real_skipped_elems.append(real_dat[8])
		new_num_elem.append(real_dat[9])
	real_res = np.array(real_res)
	real_err = np.array(real_err)
	real_nanless_res = np.array(real_nanless_res)
	real_nanless_err = np.array(real_nanless_err)
	real_weights = np.array(real_weights)
	y_ax_real = np.array(y_ax_real)
	real_cdists = np.array(real_cdists)
	real_used_elems = np.array(real_used_elems)
	real_skipped_elems = np.array(real_skipped_elems)
	new_num_elem = np.array(new_num_elem)
	
	#Create synthetic spectra for each value of sigma and fit
	sigma_vals = np.random.uniform(0, 0.1, int(num_sigma))
	fake_res = []
	fake_err = []
	y_ax_psm = []
	psm_cdists = []
	fake_nanless_res = []
	for i in range(len(scale_factors)):
		fake_dat = ABC.psm_data(num_elem, num_stars, apogee_cluster_data, sigma_vals[i], T, cluster, spectra, scaled_errs[i])
		fake_res.append(fake_dat[0])
		fake_err.append(fake_dat[1])
		y_ax_psm.append(fake_dat[2])
		psm_cdists.append(fake_dat[3])
		fake_nanless_res.append(fake_dat[4])
		
	fake_res = np.array(fake_res)
	fake_err = np.array(fake_err)
	y_ax_psm = np.array(y_ax_psm)
	psm_cdists = np.array(psm_cdists)
	fake_nanless_res = np.array(fake_nanless_res)
	
	#Create arrays of real data in the same shape as the simulated data
	all_y_ax_real = np.full_like(y_ax_psm, y_ax_real)
	all_real_cdists = np.full_like(psm_cdists, real_cdists)
	all_real_res = np.full_like(psm_cdists, real_res)
	all_real_err = np.full_like(fake_err, real_err)
	all_real_weights = np.full_like(fake_nanless_res, real_weights)
	
	#Calculate summary statistics
	D_cov_all = []
	ks_all = []
	for i in range(len(scale_factors)):
		D_cov_all.append(ABC.d_cov_all(cluster, new_num_elem[i], all_real_weights[i], all_real_res[i], all_real_err[i], fake_res[i], fake_err[i], num_stars, sigma_vals[i]))
		ks_all.append(ABC.KS_all(cluster, new_num_elem[i], all_y_ax_real[i], all_real_cdists[i], y_ax_psm[i], psm_cdists[i], sigma_vals[i]))
	
	D_cov_all = np.array(D_cov_all)
	ks_all = np.array(ks_all)
	
	#Save to file
	timestr = time.strftime("%Y%m%d_%H%M%S") #Date and time by which to identify file
	name_string = str(cluster).replace(' ','') #Remove spaces in cluster name
	#Personal path
	#path = '/Users/chloecheng/Personal/' + name_string + '/' + name_string + '_' + 'scaled_err' + '_' + timestr + '.hdf5'
	#Server path
	path = '/geir_data/scr/ccheng/AST425/Personal/' + name_string + '/' + name_string + '_' + 'scaled_err' + '_' + timestr + '.hdf5'
	file = h5py.File(path, 'w')
	file['D_cov'] = D_cov_all
	file['KS'] = ks_all
	file['sigma'] = sigma_vals
	file['scale_factors'] = scale_factors
	file.close()
	return D_cov_all, ks_all
	
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	D_cov_all, ks_all = run_everything(arguments['--cluster'], arguments['--num_sigma'])