"""Usage: ABC.py [-h][--cluster=<arg>][--red_clump=<arg>][--sigma=<arg>][--location=<arg>][--elem=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
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
import occam_clusters_input as oc ###data reading and fitting
import occam_clusters_post_process as pp ###cumulative distributions
#basic math and plotting 
import numpy as np ###Numpy
from docopt import docopt ###Docopt for executing in terminal 
from scipy.interpolate import interp1d ###Interpolation for KS distance
import h5py ###File writing (for KS and Dcov)
import glob ###Checking if file exists (for KS and Dcov)
import os ###getting PID for file labelling
import time ###Labelling files 
#apogee package
from apogee.tools import toApStarGrid ###Changing between DR12 and DR14 for fake spectra
from apogee.tools import toAspcapGrid ###Changing between DR12 and DR14 for fake spectra
#PSM code
import psm ###Generate fake spectra

    
###Function for generating the synthetic spectra and fitting it and calculating the cumulative distributions
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
        Name of the desired cluster (e.g. 'NGC 2682')
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
    num_elem = 15 ###Number of elements in APOGEE 
    num_stars = len(spectra) ###Number of stars in the cluster
    ###Dictionary for the names of the FE abundances in the allStar file 
    fe_abundance_dict = {'element': ['C_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE', 'SI_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'V_FE', 'MN_FE', 'NI_FE', 'FE_H']}
    cluster_xh = np.zeros((num_elem, num_stars)) ###Make an empty array to add all of the cluster abundances to
    for i in range(num_elem): ###Iterate through the elements
        for j in range(num_stars): ###Iterate through the stars
        	###Get all of the [X/H] abundances for all of the elements X and all of the stars in the cluster by multiplying each [X/Fe] abundance by [Fe/H]
            cluster_xh[i] = apogee_cluster_data[fe_abundance_dict['element'][i]]*apogee_cluster_data['FE_H'] 
    cluster_avg_abundance = np.mean(cluster_xh, axis=1) ###Get the average abundances of all the elements in the cluster (so should be just 1 number for each element)
    
    cluster_logg = apogee_cluster_data['LOGG'] ###Get the surface gravities of each star from the allStar file
    elem_number_dict = {'C': 0, ###Create a dictionary to match element names to their order number 
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
    cluster_fake_abundance = np.copy(cluster_xh) ###Create a copy of the array of all abundances in the cluster to use to simulate abundances
    ###Simulate the abundances of the DESIRED ELEMENT (ONE ELEMENT ONLY) by drawing from a random normal distribution centred about the mean of THAT ONE 
    ###ELEMENT with a scatter of the chosen value of sigma.  The rest of the abundances in this array will remain the same as the data 
    cluster_fake_abundance[elem_number_dict[elem]] = np.random.normal(loc = cluster_avg_abundance[elem_number_dict[elem]], scale = float(sigma), size = num_stars)
    
    cluster_gen_spec = np.zeros((num_stars, 7214)) ###Create an empty array to add the fake spectra
    for i in range(len(spectra)): ###Iterate through all of the stars - change this to range(num_stars)
    	###Use PSM to make a fake spectrum, using the array of abundances created above (with the element in question varied by the value of sigma and the
    	###remaining values of abundances the same as the data), using the photometric Teffs calculated previously, the logg of each star, the default 
    	###PSM vturb value, and the default psm c12c13 value.
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

    #Pad psm spectra with zeros to make appropriate size for DR14
    apStar_cluster_gen_spec = toApStarGrid(cluster_gen_spec, dr='12') ###Put the fake spectra onto the DR12 grid
    cluster_padded_spec = toAspcapGrid(apStar_cluster_gen_spec, dr='14') ###Put the fake spectra onto the DR14 grid.  This will pad the spectra with zeroes
    ### to make it the right shape for DR14

    #Create array of nans to mask the psm in the same way as the spectra
    masked_psm = np.empty_like(spectra) ###Create an empty array that is the same shape as the spectra
    masked_psm[:] = np.nan ###Fill the array with NaNs to mask it

    #Mask the spectra
    for i in range(len(spectra)): ###Iterate through the stars - change this to num_stars
        for j in range(7514): ###Iterate through the wavelength range - change to len(spectra.T) or make a variable = 7514 so I can stop hardcoding
            if ~np.isnan(spectra[i][j]): ###If the entry in the real spectra is not a NaN
                masked_psm[i][j] = cluster_padded_spec[i][j] ###Make the value in the masked fake spectra the corresponding entry in the fake spectra

    #Read in repeats residuals 
    if location == 'personal': ###If running on Mac
        file = h5py.File('/Users/chloecheng/Personal/repeats_dr14.hdf5', 'r') ###Path to repeats file that I made
    elif location == 'server': ###If running on server
        file = h5py.File('/geir_data/scr/ccheng/AST425/Personal/repeats_dr14.hdf5', 'r') ###Path to repeats file that I made
    repeat_res = file['residuals'][()] ###Get the repeats
    file.close() ###Close the file

    #Cut out gaps between detectors for DR14
    repeats_dr14 = toAspcapGrid(repeat_res, dr='14') ###Cut the gaps between the detectors in the repeats
    #Calculate 6sigma for repeats
    repeats_mean = np.nanmean(repeats_dr14) ###Get the mean of the repeats, avoiding the masked areas that are NaNs
    repeats_std = np.nanstd(repeats_dr14) ###Get the standard deviation of the repeats, avoiding the masked areas that are NaNs
    repeats_6sigma = repeats_mean + repeats_std*6 ###Get the value for 6 sigma of the repeats from the mean (check that this is correct)

    #Create fake noise to add to the psm
    selected_repeats = [] ###Empty list to append the random repeats that I will use to multiply the spectral errors
    for i in range(0, num_stars):  ###Iterate through the stars
        #Select a random star from the repeats residuals by which to multiply the spectra errors
        random_repeat = np.random.choice(np.arange(0, len(repeats_dr14))) ###Randomly select one of the stars in the repeats
        selected_repeats.append(repeats_dr14[random_repeat]) ###Get all of the repeats residuals for this star
    selected_repeats = np.array(selected_repeats) ###Turn the list into an array

    #Mask individual |repeats| that are > 6sigma
    for i in range(len(selected_repeats)): ###Iterate through the number of selected repeats (number of stars)
        for j in range(len(selected_repeats.T)): ###Iterate through the wavelength range 
            if np.abs(selected_repeats[i][j]) > repeats_6sigma: ###If the absolute value of a one of the entries in the selected repeats is greater than 6sigma
                selected_repeats[i][j] = np.nan ###Mask it out

    #Multiply the repeats by the spectral errors
    cluster_fake_errs = spectra_errs*selected_repeats ###Multiply the spectral errors by these randomly selected repeats.  The spectral errors will become masked in the same way as the repeats
    #Pad the fake errors with zeroes in the same places as the PSM spectra 
    cluster_fake_errs[masked_psm == 0] = 0.0 ###Pad the fake errors with zeroes in the same places as the fake spectra from modifying the DR12 wavelength range to fit DR14

    #Add the noise to the psm 
    noise_fake_spec = masked_psm + cluster_fake_errs ###Add the fake errors that I've created to the fake spectra as fake noise to make more realistic
    #Mask the real spectra and spectra errors in the same way as the fake spectra
    masked_real_spectra = np.copy(spectra) ###Make a copy of the observed spectra to mask in the same way as the fake spectra
    masked_real_spectra_err = np.copy(spectra_errs) ###Make a copy of the observed spectral errors to mask in the same way as the fake spectra
    masked_real_spectra[np.isnan(noise_fake_spec)] = np.nan ###Mask the observed spectra in the same way as the fake spectra
    masked_real_spectra_err[np.isnan(noise_fake_spec)] = np.nan ###Mask the observed spectral errors in the same way as the fake spectra

    #Remove empty spectra ###I'm not sure if this chunk is necessary but I wrote it in just in case
    final_fake_spec = [] ###Empty list to append the final set of fake spectra
    final_real_spectra = [] ###Empty list to append the final set of real spectra
    final_real_spectra_err = [] ###Empty list to append the final set of real spectral errors 
    for i in range(len(noise_fake_spec)): ###Iterate through the fake spectra
        if any(noise_fake_spec[i,:] != 0): ###If there are rows that are not completely filled with zeroes
            final_fake_spec.append(noise_fake_spec[i]) ###Append those fake spectra
            final_real_spectra.append(masked_real_spectra[i]) ###Append those real spectra
            final_real_spectra_err.append(masked_real_spectra_err[i]) ###Append those real spectral errors
    final_fake_spec = np.array(final_fake_spec) ###Make into array
    final_real_spectra = np.array(final_real_spectra) ###Make into array
    final_real_spectra_err = np.array(final_real_spectra_err) ###Make into array
    
    #Run fitting function on synthetic spectra 
    fake_res, fake_err, fake_points, fake_temp, fake_a, fake_b, fake_c, fake_nanless_res, fake_nanless_err, fake_nanless_T, fake_nanless_points, fake_normed_weights = oc.fit_func(elem, cluster, final_fake_spec, final_real_spectra_err, T, dat_type='sim', run_number = run_number, location = location, sigma_val = sigma)
    
    #Cumulative distributions
    y_ax_psm, psm_cdists  = pp.cum_dist(fake_nanless_res, fake_nanless_err)
    return fake_res, fake_err, y_ax_psm, psm_cdists, fake_nanless_res, final_real_spectra, final_real_spectra_err

def cov_matrix(res, err, num_stars): ###Function to compute the covariance matrix of a set of residuals, tailored to take NaNs into account so everything stays the same shape
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
	
	normalized_res = res/err ###Normalize the residuals by the errors
	covariance_matrix = np.zeros((len(normalized_res), len(normalized_res))) ###Create an empty array to append the covariance matrix 
	
	#Calculate means and sums
	pixel_means = np.nanmean(normalized_res, axis=1) ###Calculate the mean of the normalized residuals across the pixels, taking NaNs into account 
	tiled_means = np.tile(pixel_means, (num_stars, 1)).T ###Tile the means to make them the right shape
	diffs = normalized_res - tiled_means ###Subtract the means from the residuals
	row_sums = np.sum(~np.isnan(normalized_res), axis=1) ###Take the sum of the residuals over the pixels, ignoring NaNs
	
	#Create covariance matrix
	for pixel in range(len(normalized_res)): ###Iterate through the normalized residuals
		rowdiff = diffs[pixel] ###Get the value of the differences of the means and residuals at each pixel
		tiled_row = np.tile(rowdiff, (len(normalized_res), 1)) ###Tile this difference 
		covariance_matrix[pixel] = np.nansum(diffs*tiled_row, axis=1)/(row_sums[pixel]-1) ###Compute the covariance matrix at that pixel 
	return covariance_matrix

###Function to compute the delta covariance summary statistic
def d_cov(cluster, weights, data_res, data_err, simulated_res, simulated_err, num_stars, sigma, elem, location, run_number):
	"""Return the covariance matrix summary statistic, as computed in Bovy 2016.
	
	Parameters
	----------
	cluster : str
		The name of the cluster being analyzed (e.g. 'NGC 2682')
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
	data_cov = cov_matrix(data_res, data_err, num_stars) ###Covariance matrix of the data
	sim_cov = cov_matrix(simulated_res, simulated_err, num_stars) ###Covariance matrix of the simulation
	
	#Compute the summary statistic
	stat = np.zeros_like(data_cov) ###Create an empty array to add the statistic to for each pixel
	for i in range(len(data_cov)): ###Iterate through the pixels
		for j in range(len(data_cov)): ###Iterate through the pixels
			stat[i][j] = np.sqrt(weights[i]*weights[j])*((data_cov[i][j] - sim_cov[i][j])**2) ###Compute the stat before summing and sqrting
	D_cov = np.sqrt(np.sum(stat)) ###Sum and square root 
	
	#Save data to file
	name_string = str(cluster).replace(' ','') #Remove spaces from cluster name ###Name of the cluster
	timestr = time.strftime("%Y%m%d_%H%M%S") ###Date and time
	pid = str(os.getpid()) ###PID to label to file
	if location == 'personal': ###If running on Mac
		path = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'D_cov' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' 
	elif location == 'server': ###If running on server
		path = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'D_cov' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' #Server path
	#If file exists, append to file
	if glob.glob(path): ###If the file exists
		file = h5py.File(path, 'a') ###Append to the file
		grp = file.create_group(str(sigma)) ###Create a group named after the value of sigma being examined
		grp['D_cov'] = D_cov 
		file.close()
	#If file does not exist, create a new file
	else: ###IF the file does not exist
		file = h5py.File(path, 'w') ###Write a new file
		grp = file.create_group(str(sigma)) ###Create a group named after the value of sigma being examined
		grp['D_cov'] = D_cov
		file.close()
	return D_cov
    
def KS(cluster, data_yax, data_cdist, sim_yax, sim_cdist, sigma, elem, location, run_number): ###Function to compute the KS distance summary statistic
    """Return the KS distance summary statistic.
    
    Parameters
    ----------
    cluster : str
		The name of the cluster being analyzed (e.g. 'NGC 2682')
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
    real_interp = interp1d(data_cdist, data_yax) ###Interpolate the data cumulative distribution
    fake_interp = interp1d(sim_cdist, sim_yax) ###Interpolate the simulation cumulative distribution 
    lower_bound = np.max((np.min(data_cdist), np.min(sim_cdist))) ###Lower bound of the interpolation
    upper_bound = np.min((np.max(data_cdist), np.max(sim_cdist))) ###Upper bound of the interpolation
    xnew = np.linspace(lower_bound, upper_bound, 1000) ###Xaxis for the interpolated cumulative distributions
    
    #Compute the KS distance
    dist = np.max(np.abs(real_interp(xnew) - fake_interp(xnew))) ###Compute the maximum of the absolute difference between the two interpolated cdists
    
    #Save data to file
    name_string = str(cluster).replace(' ','') #Remove spaces from cluster name ###Name of the cluster
    timestr = time.strftime("%Y%m%d_%H%M%S") ###Date and time
    pid = str(os.getpid()) ###PID for file labelling
    if location == 'personal': ###If running on Mac
    	path = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'KS' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' 
    elif location == 'server': ###If running on server
    	path = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string + '/' + name_string + '_'  + str(elem) + '_' + 'KS' + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' 
    #If file exists, append to file
    if glob.glob(path): ###If the file exists
    	file = h5py.File(path, 'a') ###Append to it
    	grp = file.create_group(str(sigma)) ###Create a group named after the value of sigma being examined
    	grp['KS'] = dist
    	file.close()
    #Else create a new file
    else: ###If the file does not exist
    	file = h5py.File(path, 'w') ###Write it
    	grp = file.create_group(str(sigma)) ###Create a group named after the value of sigma being examined
    	grp['KS'] = dist
    	file.close()
    return dist
    

if __name__ == '__main__': ###Docopt stuff
	arguments = docopt(__doc__)
	
	###Get the allStar data and the spectra using the function from occam_clusters_input.py
	apogee_cluster_data, spectra, spectra_errs, T, bitmask = oc.get_spectra(arguments['--cluster'], arguments['--red_clump'], ['--location']) 
	num_elem = 15 ###Number of APOGEE elements
	num_stars = len(spectra) ###Number of stars
	###Make synthetic spectra and fit
	fake_res, fake_err, y_ax_psm, psm_cdists, fake_nanless_res, final_real_spectra, final_real_spectra_err = psm_data(num_elem, num_stars, apogee_cluster_data, arguments['--sigma'], T, arguments['--cluster'], spectra, spectra_errs, run_number, arguments['--location'], arguments['--elem'])
	###Get the real fits using the function from occam_clusters_input.py
	real_res, real_err, real_points, real_temp, real_a, real_b, real_c, real_nanless_res, real_nanless_err, real_nanless_T, real_nanless_points, real_normed_weights = oc.fit_func(arguments['--elem'], arguments['--cluster'], final_real_spectra, final_real_spectra_err, T, arguments['--dat_type'], run_number, arguments['--location'], arguments['--sigma'])
	###Get the real cumulative distributions using the function from occam_clusters_post_process.py
	y_ax_real, real_cdists = pp.cum_dist(real_nanless_res, real_nanless_err)
	###Compute the delta covariance
	D_cov = d_cov(arguments['--cluster'], real_weights, real_res, real_err, fake_res, fake_err, num_stars, arguments['--sigma'], arguments['--elem'], arguments['--location'], run_number)
	###Compute the KS distance
	ks = KS(arguments['--cluster'], y_ax_real, real_cdists, y_ax_psm, psm_cdists, arguments['--sigma'], arguments['--elem'], arguments['--location'], run_number)