"""Usage: pj_ABC.py [-h][--cluster=<arg>][--red_clump=<arg>][--sigma=<arg>][--location=<arg>][--elem=<arg>]

Examples:
	Cluster name: e.g. input --cluster='PJ_26'
	Red clump: e.g. input --red_clump='True'
	Sigma choice: e.g. input --sigma=0.1
	Location: e.g. input --location='personal'
	Element: e.g. input --elem='AL'

-h  Help file
--cluster=<arg>  Cluster name
--red_clump=<arg> Whether to exclude red clump stars in rcsample or not
--sigma=<arg> Sigma value choice
--location=<arg> Machine where the code is being run
--elem=<arg> Element to examine

"""

#Imports
#project scripts
import pj_clusters_input as pj
import occam_clusters_post_process as pp
#basic math and plotting
import numpy as np
from docopt import docopt
from scipy.interpolate import interp1d
import h5py
import glob
import os
import time
#apogee package
from apogee.tools import toApStarGrid
from apogee.tools import toAspcapGrid
from apogee.tools import apStarInds
#PSM code
import psm

def psm_data(num_elem, num_stars, apogee_cluster_data, sigma, T, cluster, spectra, spectra_errs, run_number, location, elem):
    """Return the residuals (with and without NaNs), errors (with and without NaNs), cumulative distributions,
    and skipped elements for the simulated spectra.

    This function generates synthetic spectra using PSM and a specified sigma value, then fits the simulated
    spectra in the same way as the data.

    Parameters
    ----------
    num_elem : int
        The number of elements in APOGEE
    num_stars : int
        The number of stars in the desired cluster
    apogee_cluster_data : structured array
        All cluster data from APOGEE
    sigma : float
        The value of sigma to create the simulated spectra
    T : tuple
        Array of floats representing the effective temperature of each star in the cluster
    cluster : str
        Name of the desired cluster (e.g. 'PJ_26')
    spectra : tuple
        Array of floats representing the spectra of the desired cluster
    spectra_errs : tuple
        Array of floats representing the spectral uncertainties of the desired cluster
    run_number : int
        Number of the run by which to label files
    location : str
        If running locally, set to 'personal'.  If running on the server, set to 'server'.
    elem : str
        Element being examined (e.g. 'AL')

    Returns
    -------
    fake_res : tuple
        Array of floats representing the residuals of the quadratic fit 
    fake_err : tuple
        Array of floats representing the spectral errors corresponding to the residuals
    y_ax_psm : tuple
        One-dimensional array containing values from 0 to 1, the same size as cdist
    psm_cdists : tuple
        One-dimensional array containing the sorted, normalized fit residuals
    fake_nanless_res : tuple
        Array of floats representing the residuals of the quadratic fit, with NaNs removed 
        (doesn't return fake_nanless_err because they are the same as real_nanless_err)
    fake_used_elems : tuple
        Array of str representing the elements used in the cluster (some elements may be omitted due to 
        lack of data)
    fake_skipped_elems : tuple
        Array of str representing the elements skipped due to lack of data 
    final_real_spectra : tuple
        Array of observed spectra masked in the same way as the simulated spectra 
    final_real_spectra_err : tuple
        Array of observed spectral errors masked in the same way as the simulated spectra
    """
    
    #Abundances WRT H
    num_elem = 15
    num_stars = len(spectra)
    fe_abundance_dict = {'element': ['C_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE', 'SI_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'V_FE', 'MN_FE', 'NI_FE', 'FE_H']}
    cluster_xh = np.zeros((num_elem, num_stars))
    for i in range(num_elem):
    	for j in range(num_stars):
    		if fe_abundance_dict['element'][i] == 'FE_H':
    			cluster_xh[i] = apogee_cluster_data['FE_H']
    		else:
    			cluster_xh[i] = apogee_cluster_data[fe_abundance_dict['element'][i]] + apogee_cluster_data['FE_H']
    cluster_avg_abundance = np.mean(cluster_xh, axis=1)
    
    cluster_logg = apogee_cluster_data['LOGG']
    elem_number_dict = {'C': 0,
                       'N': 1,
                       'O': 2,
                       'NA': 3,
                       'MG': 4,
                       'AL': 5,
                       'SI': 6,
                       'S': 7,
                       'K': 8,
                       'CA': 9,
                       'TI': 10,
                       'V': 11,
                       'MN': 12,
                       'FE': 13,
                       'NI': 14}
    cluster_fake_abundance = np.copy(cluster_xh)
    cluster_fake_abundance[elem_number_dict[elem]] = np.random.normal(loc = cluster_avg_abundance[elem_number_dict[elem]], scale = float(sigma), size = num_stars)
    
    cluster_gen_spec = np.zeros((num_stars, 7214))
    for i in range(len(spectra)):
        cluster_gen_spec[i] = psm.generate_spectrum(Teff = T[i]/1000, logg = cluster_logg[i], vturb = psm.vturb, 
                                                   ch = cluster_fake_abundance[elem_number_dict['C']][i], 
                                                   nh = cluster_fake_abundance[elem_number_dict['N']][i], 
                                                   oh = cluster_fake_abundance[elem_number_dict['O']][i],
                                                   nah = cluster_fake_abundance[elem_number_dict['NA']][i], 
                                                   mgh = cluster_fake_abundance[elem_number_dict['MG']][i], 
                                                   alh = cluster_fake_abundance[elem_number_dict['AL']][i], 
                                                   sih = cluster_fake_abundance[elem_number_dict['SI']][i], 
                                                   sh = cluster_fake_abundance[elem_number_dict['S']][i], 
                                                   kh = cluster_fake_abundance[elem_number_dict['K']][i],
                                                   cah = cluster_fake_abundance[elem_number_dict['CA']][i], 
                                                   tih = cluster_fake_abundance[elem_number_dict['TI']][i], 
                                                   vh = cluster_fake_abundance[elem_number_dict['V']][i], 
                                                   mnh = cluster_fake_abundance[elem_number_dict['MN']][i], 
                                                   nih = cluster_fake_abundance[elem_number_dict['NI']][i], 
                                                   feh = cluster_fake_abundance[elem_number_dict['FE']][i], 
                                                   c12c13 = psm.c12c13)

    
    #Mask spectra outside of boundaries of DR12 detectors
    apStar_cluster_gen_spec = toApStarGrid(cluster_gen_spec, dr='12')
    dr12_d1_left = apStarInds['12']['blue'][0]
    dr12_d1_right = apStarInds['12']['blue'][-1]
    dr12_d2_left = apStarInds['12']['green'][0]
    dr12_d2_right = apStarInds['12']['green'][-1]
    dr12_d3_left = apStarInds['12']['red'][0]
    dr12_d3_right = apStarInds['12']['red'][-1]
    for i in range(len(apStar_cluster_gen_spec)):
    	for j in range(len(apStar_cluster_gen_spec.T)):
    		if j < dr12_d1_left or (dr12_d1_right < j < dr12_d2_left) or (dr12_d2_right < j < dr12_d3_left) or j > dr12_d3_right:
    			apStar_cluster_gen_spec[i][j] = np.nan
    
    #Pad psm spectra with zeros to make appropriate size for DR14
    cluster_padded_spec = toAspcapGrid(apStar_cluster_gen_spec, dr='14')

    #Create array of nans to mask the psm in the same way as the spectra
    masked_psm = np.empty_like(spectra)
    masked_psm[:] = np.nan

    #Mask the spectra
    for i in range(len(spectra)):
        for j in range(7514):
            if ~np.isnan(spectra[i][j]):
                masked_psm[i][j] = cluster_padded_spec[i][j]

    #Read in repeats residuals 
    if location == 'personal':
        file = h5py.File('/Users/chloecheng/Personal/repeats_dr14.hdf5', 'r') 
    elif location == 'server':
        file = h5py.File('/geir_data/scr/ccheng/AST425/Personal/repeats_dr14.hdf5', 'r')
    repeat_res = file['residuals'][()]
    file.close()

    #Cut out gaps between detectors for DR14
    repeats_dr14 = toAspcapGrid(repeat_res, dr='14')
    #Calculate 6sigma for repeats
    repeats_mean = np.nanmean(repeats_dr14)
    repeats_std = np.nanstd(repeats_dr14)
    repeats_6sigma_pos = repeats_mean + repeats_std*6
    repeats_6sigma_neg = repeats_mean - repeats_std*6

    #Create fake noise to add to the psm
    selected_repeats = []
    for i in range(0, num_stars): 
        #Select a random star from the repeats residuals by which to multiply the spectra errors
        random_repeat = np.random.choice(np.arange(0, len(repeats_dr14)))
        selected_repeats.append(repeats_dr14[random_repeat])
    selected_repeats = np.array(selected_repeats)

    #Mask individual |repeats| that are > 6sigma
    for i in range(len(selected_repeats)):
    	for j in range(len(selected_repeats.T)):
    		if (selected_repeats[i][j] > repeats_6sigma_pos) or (selected_repeats[i][j] < repeats_6sigma_neg):
    		#if np.abs(selected_repeats[i][j]) > repeats_6sigma_pos:
    			selected_repeats[i][j] = np.nan

    #Multiply the repeats by the spectral errors
    cluster_fake_errs = spectra_errs*selected_repeats
    #Correct the fake errors with zeroes in the same places as the PSM spectra
    cluster_fake_errs[masked_psm == 0] = 0.0

    #Add the noise to the psm 
    noise_fake_spec = masked_psm + cluster_fake_errs
    #Mask the real spectra and spectra errors in the same way as the fake spectra
    masked_real_spectra = np.copy(spectra)
    masked_real_spectra_err = np.copy(spectra_errs)
    masked_real_spectra[np.isnan(noise_fake_spec)] = np.nan
    masked_real_spectra_err[np.isnan(noise_fake_spec)] = np.nan

    #Remove empty spectra - assertion
    final_fake_spec = []
    final_real_spectra = []
    final_real_spectra_err = []
    for i in range(len(noise_fake_spec)):
        if any(noise_fake_spec[i,:] != 0):
            final_fake_spec.append(noise_fake_spec[i])
            final_real_spectra.append(masked_real_spectra[i])
            final_real_spectra_err.append(masked_real_spectra_err[i])
    final_fake_spec = np.array(final_fake_spec)
    final_real_spectra = np.array(final_real_spectra)
    final_real_spectra_err = np.array(final_real_spectra_err)
    
    #Run fitting function on synthetic spectra
    fake_res, fake_err, fake_points, fake_temp, fake_a, fake_b, fake_c, fake_nanless_res, fake_nanless_err, fake_nanless_T, fake_nanless_points, fake_normed_weights = pj.fit_func(elem, cluster, final_fake_spec, final_real_spectra_err, T, dat_type='sim', run_number = run_number, location = location, sigma_val = sigma)
    
    #Cumulative distributions
    y_ax_psm, psm_cdists  = pp.cum_dist(fake_nanless_res, fake_nanless_err)
    return fake_res, fake_err, y_ax_psm, psm_cdists, fake_nanless_res, final_real_spectra, final_real_spectra_err

def cov_matrix(res, err, num_stars):
	"""Return the covariance matrix of the normalized fit residuals.
	
	Parameters
	----------
	res : tuple
		Array of floats representing the residuals of the quadratic fits
	err : tuple
		Array of floats representing the spectral errors corresponding to the residuals
	num_stars : int
		The number of stars in the cluster
	
	Returns 
	-------
	covariance_matrix : tuple
		The covariance matrix of the normalized residuals
	"""
	
	normalized_res = res/err
	covariance_matrix = np.zeros((len(normalized_res), len(normalized_res)))
	
	#Calculate means and sums
	pixel_means = np.nanmean(normalized_res, axis=1)
	tiled_means = np.tile(pixel_means, (num_stars, 1)).T
	diffs = normalized_res - tiled_means
	row_sums = np.sum(~np.isnan(normalized_res), axis=1)
	
	#Create covariance matrix
	for pixel in range(len(normalized_res)):
		rowdiff = diffs[pixel]
		tiled_row = np.tile(rowdiff, (len(normalized_res), 1))
		covariance_matrix[pixel] = np.nansum(diffs*tiled_row, axis=1)/(row_sums[pixel]-1)
	return covariance_matrix

def d_cov(cluster, weights, data_res, data_err, simulated_res, simulated_err, num_stars, sigma, elem, location, run_number):
	"""Return the covariance matrix summary statistic, as computed in Bovy 2016.
	
	Parameters
	----------
	cluster : str
		The name of the cluster being analyzed (e.g. 'PJ_26')
	weights : tuple
		Array of floats representing the weights of each pixel in the element
	data_res : tuple
		Array of floats representing the residuals of the quadratic fits of the data
	data_err : tuple
		Array of floats representing the spectral errors corresponding to the residuals of the data
	simulated_res : tuple
		Array of floats representing the residuals of the quadratic fits of the simulation
	simulated_err : tuple
		Array of floats representing the spectral errors corresponding to the residuals of the simulation
	num_stars : int
		The number of stars in the cluster
	sigma : float
		Value of sigma used for the simulation
	elem : str
		The desired element to analyze (e.g. 'AL')
	location : str
		If running locally, set to 'personal'.  If running on the server, set to 'server'
	run_number : int
		Number of the run by which to label files
	
	Returns
	-------
	D_cov : float
		The covariance matrix summary statistic
	"""
	
	#Compute the covariance matrices
	data_cov = cov_matrix(data_res, data_err, num_stars)
	sim_cov = cov_matrix(simulated_res, simulated_err, num_stars)
	
	#Compute the summary statistic
	stat = np.zeros_like(data_cov)
	for i in range(len(data_cov)):
		for j in range(len(data_cov)):
			stat[i][j] = np.sqrt(weights[i]*weights[j])*((data_cov[i][j] - sim_cov[i][j])**2)
	D_cov = np.sqrt(np.sum(stat))
	
	#Save data to file
	name_string = str(cluster).replace(' ','') #Remove spaces from cluster name
	timestr = time.strftime("%Y%m%d_%H%M%S")
	pid = str(os.getpid())
	if location == 'personal':
		path = '/Users/chloecheng/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'D_cov' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' 
	elif location == 'server':
		path = '/geir_data/scr/ccheng/AST425/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'D_cov' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' #Server path
	#If file exists, append to file
	if glob.glob(path):
		file = h5py.File(path, 'a')
		grp = file.create_group(str(sigma))
		grp['D_cov'] = D_cov
		file.close()
	#If file does not exist, create a new file
	else:
		file = h5py.File(path, 'w')
		grp = file.create_group(str(sigma))
		grp['D_cov'] = D_cov
		file.close()
	return D_cov
    
def KS(cluster, data_yax, data_cdist, sim_yax, sim_cdist, sigma, elem, location, run_number):
    """Return the KS distance summary statistic.
    
    Parameters
    ----------
    cluster : str
		The name of the cluster being analyzed (e.g. 'PJ_26')
    data_yax : tuple
		One-dimensional array containing values from 0 to 1, the same size as cdist, for the data
	data_cdist : tuple
		One-dimensional array containing the sorted, normalized fit residuals for the data
	sim_yax : tuple
		One-dimensional array containing values from 0 to 1, the same size as cdist, for the simulation
	sim_cdist : tuple
		One-dimensional array containing the sorted, normalized fit residuals for the simulation
	sigma : float
    	The value of sigma used to create the simulation
	elem : str
		The desired element to analyze (e.g. 'AL')
	location : str
		If running locally, set to 'personal'.  If running on the server, set to 'server'.
	run_number : int
		Number of the run by which to label files
		
	Returns
	-------
	dist : float
		The KS distance between the simulation and the data
    """
    
    #Interpolate the cumulative distributions for subtraction
    real_interp = interp1d(data_cdist, data_yax)
    fake_interp = interp1d(sim_cdist, sim_yax)
    lower_bound = np.max((np.min(data_cdist), np.min(sim_cdist)))
    upper_bound = np.min((np.max(data_cdist), np.max(sim_cdist)))
    xnew = np.linspace(lower_bound, upper_bound, 1000)
    
    #Compute the KS distance
    dist = np.max(np.abs(real_interp(xnew) - fake_interp(xnew)))
    
    #Save data to file
    name_string = str(cluster).replace(' ','') #Remove spaces from cluster name
    timestr = time.strftime("%Y%m%d_%H%M%S")
    pid = str(os.getpid())
    if location == 'personal':
    	path = '/Users/chloecheng/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'KS' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' 
    elif location == 'server':
    	path = '/geir_data/scr/ccheng/AST425/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string + '/' + name_string + '_'  + str(elem) + '_' + 'KS' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' 
    #If file exists, append to file
    if glob.glob(path):
    	file = h5py.File(path, 'a')
    	grp = file.create_group(str(sigma))
    	grp['KS'] = dist
    	file.close()
    #Else create a new file
    else:
    	file = h5py.File(path, 'w')
    	grp = file.create_group(str(sigma))
    	grp['KS'] = dist
    	file.close()
    return dist
    

if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	apogee_cluster_data, spectra, spectra_errs, T, bitmask = pj.get_spectra(arguments['--cluster'], arguments['--red_clump'], ['--location'])
	num_elem = 15
	num_stars = len(spectra)
	fake_res, fake_err, y_ax_psm, psm_cdists, fake_nanless_res, final_real_spectra, final_real_spectra_err = psm_data(num_elem, num_stars, apogee_cluster_data, arguments['--sigma'], T, arguments['--cluster'], spectra, spectra_errs, run_number, arguments['--location'], arguments['--elem'])
	real_res, real_err, real_points, real_temp, real_a, real_b, real_c, real_nanless_res, real_nanless_err, real_nanless_T, real_nanless_points, real_normed_weights = pj.fit_func(arguments['--elem'], arguments['--cluster'], final_real_spectra, final_real_spectra_err, T, arguments['--dat_type'], run_number, arguments['--location'], arguments['--sigma'])
	y_ax_real, real_cdists = pp.cum_dist(real_nanless_res, real_nanless_err)
	D_cov = d_cov(arguments['--cluster'], real_weights, real_res, real_err, fake_res, fake_err, num_stars, arguments['--sigma'], arguments['--elem'], arguments['--location'], run_number)
	ks = KS(arguments['--cluster'], y_ax_real, real_cdists, y_ax_psm, psm_cdists, arguments['--sigma'], arguments['--elem'], arguments['--location'], run_number)