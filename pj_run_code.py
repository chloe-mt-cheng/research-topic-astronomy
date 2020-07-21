"""Usage: pj_run_code.py [-h][--cluster=<arg>][--red_clump=<arg>][--num_sigma=<arg>][--run_number=<arg>]

Examples:
	Cluster name: e.g. input --cluster='PJ_26'
	Red clump: e.g. input --red_clump='True'
	Number of sigma: e.g. input --num_sigma='1000'
	Run number: e.g. input --run_number='$i'

-h  Help file
--cluster=<arg>  Cluster name
--red_clump=<arg> Whether to exclude red clump stars in rcsample or not
--num_sigma=<arg> Number of sigma
--run_number=<arg> Number of iteration

"""

#Imports
#project scripts
import pj_ABC as ABC
import pj_clusters_input as pj
#basic math and plotting
from docopt import docopt
import numpy as np
import h5py
import time
import os

def run_everything(cluster, num_sigma, red_clump, run_number):
	"""Return the covariance matrix statistics and KS distances for every element in APOGEE in the desired cluster,
	for every simulation run.  Function also saves all final summary statistics and values of sigma to file.
	
	Parameters
	----------
	cluster : str
		Name of the desired cluster (e.g. 'PJ_26')
	num_sigma : int
		Number of simulations to run 
	red_clump : bool
		If the red clump stars in rcsample are to be removed, set to True.  If all stars are to be used,
		set to False.
	run_number : int
		Number of the run by which to label files.
	
	Returns
	-------
	D_cov_all : tuple
		All covariance matrix summary statistics for all simulations
	ks_all : tuple
		All KS distances for all simulations
	"""
	
	#Create cluster directory, if doesn't exist already
	cluster_dir = pj.make_directory(cluster) 
	#Get APOGEE and spectral data
	apogee_cluster_data, spectra, spectra_errs, T, num_elem, num_stars, bitmask = ABC.get_cluster_data(cluster, red_clump)

	#Create synthetic spectra for each value of sigma and fit
	sigma_vals = np.random.uniform(0, 0.1, int(num_sigma))
	fake_res = []
	fake_err = []
	y_ax_psm = []
	psm_cdists = []
	fake_nanless_res = []
	final_real_spectra = []
	final_real_spectra_err = []
	for i in range(len(sigma_vals)):
		fake_dat = ABC.psm_data(num_elem, num_stars, apogee_cluster_data, sigma_vals[i], T, cluster, spectra, spectra_errs, run_number)
		fake_res.append(fake_dat[0])
		fake_err.append(fake_dat[1])
		y_ax_psm.append(fake_dat[2])
		psm_cdists.append(fake_dat[3])
		fake_nanless_res.append(fake_dat[4])
		final_real_spectra.append(fake_dat[7])
		final_real_spectra_err.append(fake_dat[8])
		
	fake_res = np.array(fake_res)
	fake_err = np.array(fake_err)
	y_ax_psm = np.array(y_ax_psm)
	psm_cdists = np.array(psm_cdists)
	fake_nanless_res = np.array(fake_nanless_res)
	final_real_spectra = np.array(final_real_spectra)
	final_real_spectra_err = np.array(final_real_spectra_err)
	
	#Fit the data
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
	for i in range(len(sigma_vals)):
		real_dat = ABC.real_data(num_elem, cluster, final_real_spectra[i], final_real_spectra_err[i], T, run_number)
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
	
	all_real_res = np.array(real_res)
	all_real_err = np.array(real_err)
	all_real_nanless_res = np.array(real_nanless_res)
	all_real_nanless_err = np.array(real_nanless_err)
	all_real_weights = np.array(real_weights)
	all_y_ax_real = np.array(y_ax_real)
	all_real_cdists = np.array(real_cdists)
	all_real_used_elems = np.array(real_used_elems)
	all_real_skipped_elems = np.array(real_skipped_elems)
	all_new_num_elem = np.array(new_num_elem)
	
	#Calculate summary statistics
	D_cov_all_elems = []
	ks_all_elems = []
	for i in range(len(sigma_vals)):
		D_cov_all_elems.append(ABC.d_cov_all(cluster, new_num_elem[i], all_real_weights[i], all_real_res[i], all_real_err[i],  fake_res[i], fake_err[i], num_stars, sigma_vals[i], run_number))
		ks_all_elems.append(ABC.KS_all(cluster, new_num_elem[i], all_y_ax_real[i], all_real_cdists[i], y_ax_psm[i], psm_cdists[i], sigma_vals[i], run_number))
	D_cov_all_elems = np.array(D_cov_all_elems)
	ks_all_elems = np.array(ks_all_elems)
	
	#If elements are missing, insert NaNs into the summary statistic arrays 
	nan_inds = [[] for i in range(len(D_cov_all_elems))]
	number_inds = [[] for i in range(len(D_cov_all_elems))]
	for i in range(len(D_cov_all_elems)):
		for j in range(num_elem):
			if all_real_skipped_elems[i][j] == 'nan':
				nan_inds[i].append(j)
			else:
				number_inds[i].append(j)
				
	D_cov_all = np.zeros((len(D_cov_all_elems), num_elem))
	ks_all = np.zeros((len(ks_all_elems), num_elem))
	D_cov_all[:] = np.nan
	ks_all[:] = np.nan
	for i in range(len(D_cov_all_elems)):
		D_cov_all[i][number_inds[i]] = D_cov_all_elems[i]
		ks_all[i][number_inds[i]] = ks_all_elems[i]
	
	#Write to file
	timestr = time.strftime("%Y%m%d_%H%M%S") #Date and time by which to identify file
	name_string = str(cluster).replace(' ','') 	#Remove spaces from name of cluster
	pid = str(os.getpid())
	#path = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' #Personal path - REMOVE FOR FINAL VERSION
	path = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string + '/' + name_string + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' #Server path
	file = h5py.File(path, 'w')
	file['D_cov'] = D_cov_all
	file['KS'] = ks_all
	file['sigma'] = sigma_vals
	file.close()
	return D_cov_all, ks_all
	
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	D_cov_all, ks_all = run_everything(arguments['--cluster'], arguments['--num_sigma'], arguments['--red_clump'], arguments['--run_number'])