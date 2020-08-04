"""Usage: occam_clusters_input.py [-h][--cluster=<arg>][--red_clump=<arg>][--element=<arg>][--type=<arg>][--location=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
	Red clump: e.g. input --red_clump='True'
	Element name: e.g. input --element='AL'
	Type: e.g. input --type='simulation'
	Location: e.g. input --location='personal'

-h  Help file
--cluster=<arg>  Cluster name
--red_clump=<arg> Whether to exclude red clump stars in rcsample or not
--element=<arg>  Element name
--type=<arg>  Data type 
--location=<arg> Machine where the code is being run

"""

#Imports
#apogee package 
import apogee.tools.read as apread
from apogee.tools.path import change_dr
from apogee.spec import window
from apogee.tools.read import rcsample
change_dr('14') #use DR14
#astropy helper functions
import astropy.io.fits as afits
#basic math and plotting
from docopt import docopt
import numpy as np
import pandas as pd
import h5py
import glob
import matplotlib.pyplot as plt
import os
import time

fs=16
plt.rc('font', family='serif',size=fs)

def photometric_Teff(apogee_cluster_data):
    """Return the photometric effective temperature of each star in a cluster, calculated via 
    Equation 10 from Hernandez and Bonifacio (2009).  Here, we use the median Ak and [Fe/H].
    
    The function takes apogee_cluster_data for the desired cluster and extracts the frequencies
    J and K and corrects them for extinction before feeding them into the effective temperature
    function.
    
    Parameters
    ----------
    apogee_cluster_data : structured array
        allStar data from APOGEE for the desired cluster
    
    Returns
    -------
    Teff : tuple
        Array of photometric effective temperatures
    """
    
    aktarg = apogee_cluster_data['AK_TARG']
    #Exception for unlikely AK_TARG numbers
    for i in range(len(aktarg)):
        if aktarg[i] <= -50.:
            aktarg[i] = np.nan
    
    #Correct J and K for median extinction
    med_aktarg = np.nanmedian(aktarg)
    aj = med_aktarg*2.5
    J0 = apogee_cluster_data['J'] - aj
    K0 = apogee_cluster_data['K'] - med_aktarg
    
    #Get numbers needed for Teff calculation
    colour = J0 - K0
    metallicity = np.nanmedian(apogee_cluster_data['FE_H'])
    b = np.array((0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013)) #Coefficients from Hernandez and Bonifacio (2009)
    
    #Calculate photometric Teff
    Teff = 5040/(b[0] + b[1]*colour + b[2]*colour**2 + b[3]*colour*metallicity + b[4]*metallicity
                + b[5]*metallicity**2)
    
    return Teff

def get_spectra(name, red_clump, location):
	"""Return cluster data, spectra, spectral errors, photometric Teffs, and bitmask from APOGEE.
	
	If the data file for the specified cluster already exists locally, 
	import the data from the file (cluster data, spectra, spectral errors, bitmask).
	If the data file does not exist, obtain the APOGEE spectra from a specified cluster 
	from the allStar catalogue, replacing ASPCAP abundances with astroNN abundances.
	
	Parameters
	----------
	name : str
		Name of desired cluster (i.e. 'NGC 2682') 
	red_clump : str
		If the red clump stars in rcsample are to be removed, set to 'True'.  If all stars are to be used,
		set to 'False'.
	location : str
		If running locally, set to 'personal'.  If running on the server, set to 'server'.
	
	Returns
	-------
	apogee_cluster_data (all stars) or apogee_cluster_data_final (red clumps removed) : structured array
		All cluster data from APOGEE
	spectra_50 (all stars) or spectra_final (red clumps removed) : tuple
		Array of floats representing the cleaned-up fluxes in the APOGEE spectra with red clump stars removed
	spectra_err_50 (all stars) or spectra_err_final (red clumps removed) : tuple
		Array of floats representing the cleaned-up spectral errors from the APOGEE spectra with red clump stars 
		removed
	good_T (all stars) or T_final (red clumps removed) : tuple
		Array of floats representing the effective temperatures of the stars in the cluster
		between 4000K and 5000K
	full_bitmask (all stars) or bitmask_final (red clumps removed) : tuple
		Array of ints (1 or 0), cleaned in the same way as the spectra, representing the bad pixels 
		in the APOGEE_PIXMASK bitmask
	"""
	
	#Path, strip spaces in cluster name
	if location == 'personal':
		path = '/Users/chloecheng/Personal/' + str(name).replace(' ', '') + '.hdf5'
	elif location == 'server':
		path = '/geir_data/scr/ccheng/AST425/Personal/' + str(name).replace(' ', '') + '.hdf5' 
		
	#If the data file for this cluster exists, save the data to variables
	if glob.glob(path):
		if red_clump == 'False':
			file = h5py.File(path, 'r')
			apogee_cluster_data = file['apogee_cluster_data'][()]
			spectra_50 = file['spectra'][()]
			spectra_err_50 = file['spectra_errs'][()]
			good_T = file['T'][()]
			full_bitmask = file['bitmask'][()]
			file.close()
			print(name, ' complete.')
			return apogee_cluster_data, spectra_50, spectra_err_50, good_T, full_bitmask
		
		elif red_clump == 'True':
			file = h5py.File(path, 'r')
			apogee_cluster_data_final = file['apogee_cluster_data'][()]
			spectra_final = file['spectra'][()]
			spectra_err_final = file['spectra_errs'][()]
			T_final = file['T'][()]
			bitmask_final = file['bitmask'][()]
			file.close()
			print(name, ' complete.')
			return apogee_cluster_data_final, spectra_final, spectra_err_final, T_final, bitmask_final
		
	#If the file does not exist, get the data from APOGEE
	else:
		#Get red clump stars from rcsample
		rc_data = rcsample(dr='14')
		rc_stars = []
		for i in range(len(rc_data)):
			#rc_stars.append(rc_data[i][2]) - REMOVE IN FINAL VERSION
			rc_stars.append(rc_data[i][2].decode('UTF-8'))
		rc_stars = np.array(rc_stars)
	
		#Read in APOGEE catalogue data, removing duplicated stars and replacing ASPCAP with astroNN abundances
		apogee_cat = apread.allStar(use_astroNN_abundances=True)
		unique_apoids,unique_inds = np.unique(apogee_cat['APOGEE_ID'],return_index=True)
		apogee_cat = apogee_cat[unique_inds]
		
		#Read in overall cluster information
		cls = afits.open('occam_cluster-DR14.fits')
		cls = cls[1].data
		
		#Read in information about cluster members
		members = afits.open('occam_member-DR14.fits')
		members = members[1].data
		
		#Select all members of a given cluster
		cluster_members = (members['CLUSTER']==name) & (members['MEMBER_FLAG']=='GM') #second part of the mask indicates to only use giant stars
		member_list = members[cluster_members]
		
		#Find APOGEE entries for that cluster
		#numpy.in1d finds the 1D intersection between two lists. 
		#In this case we're matching using the unique APOGEE ID assigned to each star
		#The indices given by numpy.in1d are for the first argument, so in this case the apogee catalogue
		cluster_inds = np.in1d((apogee_cat['APOGEE_ID']).astype('U100'),member_list['APOGEE_ID'])
		apogee_cluster_data = apogee_cat[cluster_inds]
		T = photometric_Teff(apogee_cluster_data)
		
		#Mark red clump stars in the members of the cluster as NaNs
		cluster_stars = member_list['APOGEE_ID']
		cluster_marked = np.copy(cluster_stars)
		for i in range(len(cluster_stars)):
			for j in range(len(rc_stars)):
				if cluster_stars[i] == rc_stars[j]:
					cluster_marked[i] = np.nan
		
		#Get spectra, spectral errors, and bitmask for each star - apStar
		#We can use the APOGEE package to read each star's spectrum
		#We'll read in the ASPCAP spectra, which have combined all of the visits for each star and removed the spaces between the spectra
		number_of_members = len(member_list)
		spectra = np.zeros((number_of_members, 7514))
		spectra_errs = np.zeros((number_of_members, 7514))
		bitmask = np.zeros((number_of_members, 7514))
		for s,star in enumerate(apogee_cluster_data):
			spectra[s] = apread.aspcapStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=1,header=False,dr='14',aspcapWavegrid=True)
			spectra_errs[s] = apread.aspcapStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=2,header=False,dr='14',aspcapWavegrid=True)
			bitmask[s] = apread.apStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=3,header=False,dr='14', aspcapWavegrid=True)[1]
		
		#Set all entries in bitmask to integers	
		bitmask = bitmask.astype(int)
		bitmask_flip = np.zeros_like(bitmask)
		for i in range(len(spectra)):
			for j in range(7514):
				if bitmask[i][j] == 0:
					bitmask_flip[i][j] = 1
				else:
					bitmask_flip[i][j] = 0
					
		#Remove empty spectra
		full_spectra = []
		full_spectra_errs = []
		full_bitmask = []
		full_T = [] 
		full_stars = [] 
		for i in range(len(spectra)):
			if any(spectra[i,:] != 0):
				full_spectra.append(spectra[i])
				full_spectra_errs.append(spectra_errs[i])
				full_bitmask.append(bitmask_flip[i])
				full_T.append(T[i]) 
				full_stars.append(i) 
		full_spectra = np.array(full_spectra)
		full_spectra_errs = np.array(full_spectra_errs)
		full_bitmask = np.array(full_bitmask)
		full_T = np.array(full_T) 
		full_stars = np.array(full_stars) 
		full_marked_stars = cluster_marked[full_stars] 
		
		#Create array of NaNs to replace flagged values in spectra
		masked_spectra = np.empty_like(full_spectra)
		masked_spectra_errs = np.empty_like(full_spectra_errs)
		masked_spectra[:] = np.nan
		masked_spectra_errs[:] = np.nan
		
		#Mask the spectra
		for i in range(len(full_spectra)):
			for j in range(7514):
				if full_bitmask[i][j] != 0:
					masked_spectra[i][j] = full_spectra[i][j]
					masked_spectra_errs[i][j] = full_spectra_errs[i][j]
					
		#Cut stars that are outside of the temperature limits 
		good_T_inds = (full_T > 4000) & (full_T < 5000)
		final_spectra = masked_spectra[good_T_inds]
		final_spectra_errs = masked_spectra_errs[good_T_inds]
		good_T = full_T[good_T_inds]
		apogee_cluster_data = apogee_cluster_data[good_T_inds]
		full_bitmask = full_bitmask[good_T_inds]
		final_stars = full_marked_stars[good_T_inds] 
		rgs = (final_stars != 'nan') #Get indices for final red giant stars to be used
		
		#Want an SNR of 200 so set those errors that have a larger SNR to have an SNR of 200
		spectra_err_200 = np.zeros_like(final_spectra_errs)
		for i in range(len(final_spectra)):
			for j in range(7514):
				if final_spectra[i][j]/final_spectra_errs[i][j] <= 200:
					spectra_err_200[i][j] = final_spectra_errs[i][j]
				else:
					spectra_err_200[i][j] = final_spectra[i][j]/200
					
		#Cut errors with SNR of less than 50
		spectra_50 = np.copy(final_spectra)
		spectra_err_50 = np.copy(spectra_err_200)
		
		for i in range(len(final_spectra)):
			for j in range(7514):
				if final_spectra[i][j]/spectra_err_200[i][j] <= 50:
					spectra_50[i][j] = np.nan
					spectra_err_50[i][j] = np.nan
		
		#Cut red clumps
		logg = apogee_cluster_data['LOGG']
		apogee_cluster_data_final = apogee_cluster_data[rgs]
		spectra_final = spectra_50[rgs]
		spectra_err_final = spectra_err_50[rgs]
		T_final = good_T[rgs]
		bitmask_final = full_bitmask[rgs]
		
		if red_clump == 'False':
			#Write to file
			file = h5py.File(path, 'w')
			file['apogee_cluster_data'] = apogee_cluster_data
			file['spectra'] = spectra_50
			file['spectra_errs'] = spectra_err_50
			file['T'] = good_T
			file['bitmask'] = full_bitmask
			file.close()
			print(name, 'complete')
			
			return apogee_cluster_data, spectra_50, spectra_err_50, good_T, full_bitmask
		
		elif red_clump == 'True':
			#Write to file 
			file = h5py.File(path, 'w')
			file['apogee_cluster_data'] = apogee_cluster_data_final
			file['spectra'] = spectra_final
			file['spectra_errs'] = spectra_err_final
			file['T'] = T_final
			file['bitmask'] = bitmask_final
			file.close()
			print(name, 'complete')
			
			return apogee_cluster_data_final, spectra_final, spectra_err_final, T_final, bitmask_final

def weight_lsq(data, temp, error):
	"""Return the quadratic fit parameters for a data set using the weighted least-squares method from Hogg 2015. 
	
	Parameters
	----------
	data : tuple
		Array of floats representing the fluxes of a particular element for all stars in a cluster
	temp : tuple
		Array of floats representing the effective temperature of each star, obtained from APOGEE
	error : tuple
		Array of floats representing the spectral uncertainties for data
	
	Returns
	-------
	a : float
		Represents the fit parameter for the quadratic term in the fit
	b : float
		Represents the fit parameter for the linear term in the fit
	c : float
		Represents the fit parameter for the constant term in the fit
	"""
	
	try:
		#Run the fitting algorithm on the data
		Y = data.T #Data vector
		ones_column = np.ones(len(temp)) #Constant column
		A = np.column_stack((temp**2, temp, ones_column)) #Temperature matrix
		C = np.zeros((len(data), len(data))) #Covariance matrix
		np.fill_diagonal(C, error**2) #Fill covariance matrix
		C_inv = np.linalg.inv(C) #Invert covariance matrix
		
		#Perform the matrix multiplication
		step1 = np.dot(A.T, C_inv)
		step2 = np.dot(step1, A)
		step3 = np.dot(A.T, C_inv)
		step4 = np.dot(step3, Y)
		
		#Calculate the parameters
		parameters = np.dot(np.linalg.inv(step2), step4) 
		#Isolate the parameters
		a = parameters[0]
		b = parameters[1]
		c = parameters[2]
		return a, b, c
		
	except np.linalg.LinAlgError as e:
		#Catch if covariance matrix is non-diagonal
		plt.figure()
		plt.imshow(C)
		plt.colorbar()
		print(e)
        
def residuals(data, fit):
	"""Return the residuals from a fit.
	
	Parameters
	----------
	data : tuple
		Array of floats representing the fluxes of a particular element for all stars in a cluster
	fit : tuple
		Array of floats containing the line of best fit for the data
	
	Returns
	-------
	data - fit : tuple
		The residuals of the fit
	"""
	
	return data - fit

def make_directory(name):
	"""Create a new directory for a cluster, if it does not already exist.
	
	Parameters
	----------
	name : str
		Name of desired cluster (i.e. 'NGC 2682') 
	"""
	
	#Strip spaces from cluster name
	name_string = str(name).replace(' ', '')
	
	#If directory exists, do nothing
	if glob.glob(name_string):
		return None
	#If directory does not exist, make new directory
	else:
		os.mkdir(name_string)

def fit_func(elem, name, spectra, spectra_errs, T, dat_type, run_number, location, sigma_val=None):
    """Return fit residuals from quadratic fit, spectral errors for desired element, fluxes for desired element,
    an appropriately-sized array of effective temperatures, the quadratic fitting parameters, the residuals, 
    errors, temperatures, and fluxes with NaNs removed, and the normalized elemental weights.
    
    Functions:
        Reads in the DR14 windows.
        Obtains the indices of pixels of the absorption lines and saves the flux value and uncertainty for 
        each star in these pixels.
        Performs the quadratic fit on each pixel using weight_lsq() and computes the residuals using residuals().
        Obtains the flux values, uncertainties, fits, residuals, and temperatures with NaNs removed.
        Writes the residuals and fit parameters to .hdf5 files.
    
    Parameters
    ----------
    elem : str
    	Element name (i.e. 'AL')
    name : str
    	Name of desired cluster (i.e. 'NGC 2682')
    spectra : tuple
    	Array of floats representing the spectra of the desired cluster
    spectra_errs : tuple
    	Array of floats representing the spectral uncertainties of the desired cluster
    T : tuple
    	Array of floats representing the effective temperature of each star in the cluster
    dat_type : str
    	Indicates whether the data being examined is the data or a simulation
     run_number : int
		Number of the run by which to label files
	location : str
		If running locally, set to 'personal'.  If running on the server, set to 'server'.
    sigma_val : float, optional
    	Indicates the value of sigma being used for the simulation in question, if applicable (default is None)

    Returns
    -------
    elem_res : tuple
    	Array of floats representing the fit residuals, with original positioning of points maintained
    final_err : tuple
    	Array of floats representing the spectral uncertainties from the lines of the desired element,
    	with original positioning of points maintained
    final_points : tuple
    	Array of floats representing the fluxes from the lines of the desired element, with original 
    	positioning of points maintained
    temp_array : tuple
    	Array of floats representing the effective temperature of each star in the cluster, with a row for
    	each pixel of the desired element
    elem_a : tuple
    	Array of floats representing the fitting parameters for the quadratic terms in the fits for each pixel of
    	the desired element
    elem_b : tuple
    	Array of floats representing the fitting parameters for the linear terms in the fits for each pixel of
    	the desired element
    elem_c : tuple
    	Array of floats representing the fitting parameters for the constant terms in the fits for each pixel of
    	the desired element
    nanless_res : tuple
    	Array of floats representing the fit residuals, with NaNs removed
    nanless_T : tuple
    	Array of floats representing the effective temperature of each star in the cluster, with a row for 
    	each pixel of the desired element, with NaNs removed
    nanless_points : tuple
    	Array of floats representing the fluxes from the lines of the desired element, with NaNs removed
    normed_weights : tuple
    	Array of floats representing the weight of each elemental window, normalized to 1
    """
    
    change_dr('12')
    #Find the DR14 windows from the DR12 windows
    dr12_elem_windows = window.read(elem)
    change_dr('14')
    dr14_elem_windows_12 = np.concatenate((dr12_elem_windows[246:3274], dr12_elem_windows[3585:6080], dr12_elem_windows[6344:8335]))
    normalized_dr14_elem_windows_12 = (dr14_elem_windows_12 - np.nanmin(dr14_elem_windows_12))/(np.nanmax(dr14_elem_windows_12) - np.nanmin(dr14_elem_windows_12))
    
    #Get the indices of the lines 
    ind_12 = np.argwhere(normalized_dr14_elem_windows_12 > 0)
    ind_12 = ind_12.flatten()
    
    #Get the fluxes and errors from spectra
    len_spectra = len(spectra)
    elem_points_12 = np.zeros((len(ind_12), len_spectra))
    elem_err_12 = np.zeros((len(ind_12), len_spectra))
    for i in range(0, len(ind_12)):
    	for j in range(0, len_spectra):
    		elem_points_12[i][j] = spectra[j][ind_12[i]]
    		elem_err_12[i][j] = spectra_errs[j][ind_12[i]] #APOGEE measured errors
    		
    #Use only pixels with more than 5 points
    final_points_12 = []
    final_err_12 = []
    final_inds_12 = []
    for i in range(len(elem_points_12)):
    	if np.count_nonzero(~np.isnan(elem_points_12[i])) >= 5:
    		final_points_12.append(elem_points_12[i])
    		final_err_12.append(elem_err_12[i])
    		final_inds_12.append(ind_12[i])
    final_points_12 = np.array(final_points_12)
    final_err_12 = np.array(final_err_12)
    final_inds_12 = np.array(final_inds_12)
    if len(final_inds_12) == 0:
    	print('Warning: less than 5 points for every pixel, skipping ', elem)
    else:
    	dr12_weights = normalized_dr14_elem_windows_12[final_inds_12]
    	sorted_dr12_weights = np.sort(dr12_weights)
    	
    	#Get windows
    	if location == 'personal':
    		window_file = pd.read_hdf('/Users/chloecheng/Personal/dr14_windows.hdf5', 'window_df') 
    	elif location == 'server':
    		window_file = pd.read_hdf('/geir_data/scr/ccheng/AST425/Personal/dr14_windows.hdf5', 'window_df')
    		
    	dr14_elem_windows_14 = window_file[elem].values
    	normalized_dr14_elem_windows_14 = (dr14_elem_windows_14 - np.min(dr14_elem_windows_14))/(np.max(dr14_elem_windows_14) - np.min(dr14_elem_windows_14))
    	
    	#Get the indices of the lines 
    	if elem == 'C' or elem == 'N' or elem == 'FE':
    		ind = np.argwhere(normalized_dr14_elem_windows_14 > np.min(sorted_dr12_weights[int(len(sorted_dr12_weights)*0.7):]))
    	else:
    		ind = np.argwhere(normalized_dr14_elem_windows_14 > 0)
    	ind = ind.flatten()
    	
    	#Get the fluxes and errors from spectra
    	elem_points = np.zeros((len(ind), len_spectra))
    	elem_err = np.zeros((len(ind), len_spectra))
    	for i in range(0, len(ind)):
    		for j in range(0, len_spectra):
    			elem_points[i][j] = spectra[j][ind[i]]
    			elem_err[i][j] = spectra_errs[j][ind[i]] #APOGEE measured errors
    	
    	#Use only pixels with more than 5 points
    	final_points = []
    	final_err = []
    	final_inds = []
    	for i in range(len(elem_points)):
    		if np.count_nonzero(~np.isnan(elem_points[i])) >= 5:
    			final_points.append(elem_points[i])
    			final_err.append(elem_err[i])
    			final_inds.append(ind[i])
    	final_points = np.array(final_points)
    	final_err = np.array(final_err)
    	final_inds = np.array(final_inds)
    	
    	if len(final_points) == 0:
    		print('Warning: less than 5 points for every pixel, skipping ', elem)
    	else:
    	
    		#Create an appropriately-sized array of temperatures to mask as well
    		temp_array = np.full((final_points.shape), T)
    		for i in range(0, len(final_points)):
    			for j in range(0, len_spectra):
    				if np.isnan(final_points[i][j]):
    					temp_array[i][j] = np.nan
    					
    		#Do fits with non-nan numbers
    		nanless_inds = np.isfinite(final_points)
    		fits = []
    		for i in range(len(final_points)):
    			fits.append(weight_lsq(final_points[i][nanless_inds[i]], temp_array[i][nanless_inds[i]], final_err[i][nanless_inds[i]]))
    		for i in range(len(fits)):
    			fits[i] = np.array(fits[i])
    		fits = np.array(fits)
    		elem_a = fits[:,0]
    		elem_b = fits[:,1]
    		elem_c = fits[:,2]
    		
    		elem_fits = np.zeros_like(final_points)
    		for i in range(0, len(final_points)):
    			elem_fits[i] = elem_a[i]*temp_array[i]**2 + elem_b[i]*temp_array[i] + elem_c[i]
    			
    		#Calculate residuals
    		elem_res = residuals(final_points, elem_fits)
    		
    		#Remove nans from fits, residuals, errors, and temperatures for plotting and cumulative distribution 
    		#calculation purposes
    		nanless_fits = []
    		nanless_res = []
    		nanless_err = []
    		nanless_T = []
    		nanless_points = []
    		for i in range(len(final_points)):
    			nanless_fits.append(elem_fits[i][nanless_inds[i]])
    			nanless_res.append(elem_res[i][nanless_inds[i]])
    			nanless_err.append(final_err[i][nanless_inds[i]])
    			nanless_T.append(temp_array[i][nanless_inds[i]])
    			nanless_points.append(final_points[i][nanless_inds[i]])
    		for i in range(len(final_points)):
    			nanless_fits[i] = np.array(nanless_fits[i])
    			nanless_res[i] = np.array(nanless_res[i])
    			nanless_err[i] = np.array(nanless_err[i])
    			nanless_T[i] = np.array(nanless_T[i])
    			nanless_points[i] = np.array(nanless_points[i])
    		nanless_fits = np.array(nanless_fits)
    		nanless_res = np.array(nanless_res)
    		nanless_err = np.array(nanless_err)
    		nanless_T = np.array(nanless_T)
    		nanless_points = np.array(nanless_points)
    		
    		#Get the weights for later
    		weights = normalized_dr14_elem_windows_14[final_inds]
    		normed_weights = weights/np.sum(weights)
    		
    		#File-saving 
    		#If we are looking at the data
    		timestr = time.strftime("%Y%m%d_%H%M%S")
    		name_string = str(name).replace(' ', '')
    		pid = str(os.getpid())
    		if sigma_val == None:
    			if location == 'personal':
    				path_dat = '/Users/chloecheng/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    			elif location == 'server':
    				path_dat = '/geir_data/scr/ccheng/AST425/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    	
    			#If the file exists, output the desired variables
    			if glob.glob(path_dat):
    				return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
    			#If the file does not exist, create file and output the desired variables
    			else:
    				file = h5py.File(path_dat, 'w')
    				file['points'] = final_points
    				file['residuals'] = elem_res
    				file['err_200'] = final_err
    				file['a_param'] = elem_a
    				file['b_param'] = elem_b
    				file['c_param'] = elem_c
    				file.close()
    				return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
    		#If we are looking at simulations
    		else:
    			if location == 'personal':
    				path_sim = '/Users/chloecheng/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    			elif location == 'server':
    				path_sim = '/geir_data/scr/ccheng/AST425/Personal/run_files_' + name_string + '_' + str(elem) + '/' + name_string  + '/' + name_string  + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    	
    			#If the file exists, append to the file
    			if glob.glob(path_sim):
    				file = h5py.File(path_sim, 'a')
    				#If the group for the particular value of sigma exists, don't do anything
    				if glob.glob(str(sigma_val)):
    					file.close()
    				#If not, append a new group to the file for the particular value of sigma
    				else:
    					grp = file.create_group(str(sigma_val))
    					grp['points'] = final_points
    					grp['residuals'] = elem_res
    					grp['err_200'] = final_err
    					grp['a_param'] = elem_a
    					grp['b_param'] = elem_b
    					grp['c_param'] = elem_c
    					file.close()
    			#If the file does not exist, create a new file
    			else:
    				file = h5py.File(path_sim, 'w')
    				grp = file.create_group(str(sigma_val))
    				grp['points'] = final_points
    				grp['residuals'] = elem_res
    				grp['err_200'] = final_err
    				grp['a_param'] = elem_a
    				grp['b_param'] = elem_b
    				grp['c_param'] = elem_c
    				file.close()
    			return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
    
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	apogee_cluster_data, spectra, spectra_errs, T, bitmask = get_spectra(arguments['--cluster'], arguments['--red_clump'], arguments['--location'])
	cluster_dir = make_directory(arguments['--cluster'])
	elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights = fit_func(arguments['--element'], arguments['--cluster'], spectra, spectra_errs, T, arguments['--type'], run_number, arguments['--location'], sigma_val)