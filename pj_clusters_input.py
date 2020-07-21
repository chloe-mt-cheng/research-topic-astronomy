"""Usage: pj_clusters_input.py [-h][--cluster=<arg>][--red_clump=<arg>][--element=<arg>][--type=<arg>]

Examples:
	Cluster name: e.g. input --cluster='PJ_26'
	Red clump: e.g. input --red_clump='True'
	Element name: e.g. input --element='AL'
	Type: e.g. input --type='simulation'

-h  Help file
--cluster=<arg>  Cluster name
--red_clump=<arg> Whether to exclude red clump stars in rcsample or not
--element=<arg>  Element name
--type=<arg>  Data type 

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
    Equation 10 from Hernandez and Bonifacio (2009).
    
    The function takes apogee_cluster_data for the desired cluster and extracts the frequencies
    J and K and corrects them for extinction before feeding them into the effective temperature
    function.
    
    Parameters
    ----------
    apogee_cluster_data : structured array
        All cluster data from APOGEE
    
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
    
    #Correct J and K for extinction
    aj = aktarg*2.5
    J0 = apogee_cluster_data['J'] - aj
    K0 = apogee_cluster_data['K'] - aktarg
    
    #Get numbers needed for Teff calculation
    colour = J0 - K0
    metallicity = apogee_cluster_data['FE_H']
    b = np.array((0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013)) #Coefficients from Hernandez and Bonifacio (2009)
    
    #Calculate photometric Teff
    Teff = 5040/(b[0] + b[1]*colour + b[2]*colour**2 + b[3]*colour*metallicity + b[4]*metallicity
                + b[5]*metallicity**2)
    
    return Teff

def get_spectra(name, red_clump):
	"""Return cluster data, spectra, spectral errors, photometric Teffs, and bitmask from APOGEE.
	
	If the data file for the specified cluster already exists locally, 
	import the data from the file (cluster data, spectra, spectral errors, bitmask).
	If the data file does not exist, obtain the APOGEE spectra from a specified cluster 
	from the allStar catalogue, replacing ASPCAP abundances with astroNN abundances.
	
	Parameters
	----------
	name : str
		Name of desired cluster (i.e. 'PJ_26') 
	red_clump : bool
		If the red clump stars in rcsample are to be removed, set to True.  If all stars are to be used,
		set to False.
	
	Returns
	-------
	cluster_data_full (all stars) or cluster_data (red clumps removed) : structured array
		All cluster data from APOGEE
	cluster_spectra_full (all stars) or cluster_spectra (red clumps removed) : tuple
		Array of floats representing the cleaned-up fluxes in the APOGEE spectra with red clump stars removed
	cluster_spectra_errs_full (all stars) or cluster_spectra_errs (red clumps removed) : tuple
		Array of floats representing the cleaned-up spectral errors from the APOGEE spectra with red clump stars 
		removed
	cluster_T_full (all stars) or cluster_T (red clumps removed) : tuple
		Array of floats representing the effective temperatures of the stars in the cluster
		between 4000K and 5000K
	full_bitmask (all stars) or bitmask_final (red clumps removed) : tuple
		Array of ints (1 or 0), cleaned in the same way as the spectra, representing the bad pixels 
		in the APOGEE_PIXMASK bitmask
	"""
	
	#path = '/Users/chloecheng/Personal/' + str(name) + '.hdf5' #Personal path - REMOVE FOR FINAL VERSION
	path = '/geir_data/scr/ccheng/AST425/Personal/' + str(name) + '.hdf5' #Server path
	
	
	#If the data file for this cluster exists, save the data to variables
	if glob.glob(path):
		if red_clump == False:
			file = h5py.File(path, 'r')
			apogee_cluster_data = file['apogee_cluster_data'][()]
			spectra_50 = file['spectra'][()]
			spectra_err_50 = file['spectra_errs'][()]
			good_T = file['T'][()]
			full_bitmask = file['bitmask'][()]
			file.close()
			print(name, ' complete.')
			return apogee_cluster_data, spectra_50, spectra_err_50, good_T, full_bitmask
		
		else:
			file = h5py.File(path, 'r')
			apogee_cluster_data_final = file['apogee_cluster_data'][()]
			spectra_final = file['spectra'][()]
			spectra_err_final = file['spectra_errs'][()]
			T_final = file['T'][()]
			bitmask_final = file['bitmask'][()]
			file.close()
			print(name, ' complete.')
			return apogee_cluster_data_final, spectra_final, spectra_err_final, T_final, bitmask_final
	
	#If the data file for this cluster exists, save the data to variables
	if glob.glob(path):
		if red_clump == False:
			file = h5py.File(path, 'r')
			cluster_data_full = file['apogee_cluster_data'][()]
			cluster_spectra_full = file['spectra'][()]
			cluster_spectra_errs_full = file['spectra_errs'][()]
			cluster_T_full = file['T'][()]
			full_bitmask = file['bitmask'][()]
			file.close()
			print(name, ' complete.')
			return cluster_data_full, cluster_spectra_full, cluster_spectra_errs_full, cluster_T_full, full_bitmask
		
		else:
			file = h5py.File(path, 'r')
			cluster_data = file['apogee_cluster_data'][()]
			cluster_spectra = file['spectra'][()]
			cluster_spectra_errs = file['spectra_errs'][()]
			cluster_T = file['T'][()]
			bitmask_final = file['bitmask'][()]
			file.close()
			print(name, ' complete.')
			return cluster_data, cluster_spectra, cluster_spectra_errs, cluster_T, bitmask_final
		
	#If the file does not exist
	else:
		#Get red clump stars from rcsample
		rc_data = rcsample(dr='14')
		rc_stars = []
		for i in range(len(rc_data)):
			#rc_stars.append(rc_data[i][2]) - REMOVE IN FINAL VERSION
			rc_stars.append(rc_data[i][2].decode('UTF-8'))
		rc_stars = np.array(rc_stars)
		
		#Read in PJ catalogue data
		#apogee_cluster_data = np.load('/Users/chloecheng/Personal/published_clusters.npy') #Personal path - REMOVE FOR FINAL VERSION
		apogee_cluster_data = np.load('/geir_data/scr/ccheng/AST425/Personal/published_clusters.npy') #Server path

		#Get temperatures
		T = photometric_Teff(apogee_cluster_data)
		
		#Get spectra for each star
		number_of_members = 360
		spectra = np.zeros((number_of_members, 7514))
		spectra_errs = np.zeros((number_of_members, 7514))
		bitmask = np.zeros((number_of_members, 7514))
		missing_spectra = []
		stars = []
		for s,star in enumerate(apogee_cluster_data):
			loc = star['FIELD'].decode('utf-8')
			apo = star['APOGEE_ID'].decode('utf-8')
			stars.append(apo)
			try:
				spectra[s] = apread.aspcapStar(loc,apo,ext=1,header=False,dr='16',aspcapWavegrid=True,telescope=star['TELESCOPE'].decode('utf-8'))
				spectra_errs[s] = apread.aspcapStar(loc,apo,ext=2,header=False,dr='16',aspcapWavegrid=True,telescope=star['TELESCOPE'].decode('utf-8'))
				bitmask[s] = apread.apStar(loc,apo,ext=3,header=False,dr='16', aspcapWavegrid=True,telescope=star['TELESCOPE'].decode('utf-8'))[1]
			#If the spectrum is missing, set bitmask to value that will be removed
			except OSError:
				bitmask[s] = -1.0
				missing_spec.append(s)
				print('missing ',star['APOGEE_ID'].decode("utf-8"))
				
		#Mark red clump stars
		PJ_stars = np.array(stars)
		PJ_marked = np.copy(PJ_stars)
		for i in range(len(PJ_stars)):
			for j in range(len(rc_stars)):
				if PJ_stars[i] in rc_stars[j]:
					PJ_marked[i] = np.nan
		
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
		full_stars = []
		full_T = []
		for i in range(len(spectra)):
			if any(spectra[i,:] != 0):
				full_spectra.append(spectra[i])
				full_spectra_errs.append(spectra_errs[i])
				full_bitmask.append(bitmask_flip[i])
				full_stars.append(i)
				full_T.append(T[i])
		full_spectra = np.array(full_spectra)
		full_spectra_errs = np.array(full_spectra_errs)
		full_bitmask = np.array(full_bitmask)
		full_stars = np.array(full_stars)
		full_T = np.array(full_T)
		full_marked_stars = PJ_marked[full_stars]
		
		#Create array of nans to replace flagged values in spectra
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
		final_stars = full_marked_stars[good_T_inds] #ADDED
		rgs = (final_stars != 'nan') #ADDED
		
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
		
		#Separate out individual clusters
		cluster_ids = apogee_cluster_data['CLUSTER_ID']
		PJ_26 = []
		PJ_95 = []
		PJ_471 = []
		PJ_162 = []
		PJ_398 = []
		PJ_151 = []
		PJ_230 = []
		PJ_939 = []
		PJ_262 = []
		PJ_289 = []
		PJ_359 = []
		PJ_396 = []
		PJ_899 = []
		PJ_189 = []
		PJ_574 = []
		PJ_641 = []
		PJ_679 = []
		PJ_1976 = []
		PJ_88 = []
		PJ_1349 = []
		PJ_1811 = []
		
		for i in range(len(apogee_cluster_data)):
			if cluster_ids[i] == 26:
				PJ_26.append(i)
			elif cluster_ids[i] == 95:
				PJ_95.append(i)
			elif cluster_ids[i] == 471:
				PJ_471.append(i)
			elif cluster_ids[i] == 162:
				PJ_162.append(i)
			elif cluster_ids[i] == 398:
				PJ_398.append(i)
			elif cluster_ids[i] == 151:
				PJ_151.append(i)
			elif cluster_ids[i] == 230:
				PJ_230.append(i)
			elif cluster_ids[i] == 939:
				PJ_939.append(i)
			elif cluster_ids[i] == 262:
				PJ_262.append(i)
			elif cluster_ids[i] == 289:
				PJ_289.append(i)
			elif cluster_ids[i] == 359:
				PJ_359.append(i)
			elif cluster_ids[i] == 396:
				PJ_396.append(i)
			elif cluster_ids[i] == 899:
				PJ_899.append(i)
			elif cluster_ids[i] == 189:
				PJ_189.append(i)
			elif cluster_ids[i] == 574:
				PJ_574.append(i)
			elif cluster_ids[i] == 641:
				PJ_641.append(i)
			elif cluster_ids[i] == 679:
				PJ_679.append(i)
			elif cluster_ids[i] == 1976:
				PJ_1976.append(i)
			elif cluster_ids[i] == 88:
				PJ_88.append(i)
			elif cluster_ids[i] == 1349:
				PJ_1349.append(i)
			elif cluster_ids[i] == 1811:
				PJ_1811.append(i)
				
		cluster_dict = {'PJ_26': PJ_26, 'PJ_95': PJ_95, 'PJ_471': PJ_471, 'PJ_162': PJ_162, 'PJ_398': PJ_398, 'PJ_151': PJ_151,
                'PJ_230': PJ_230, 'PJ_939': PJ_939, 'PJ_262': PJ_262, 'PJ_289': PJ_289, 'PJ_359': PJ_359,
                'PJ_396': PJ_396, 'PJ_899': PJ_899, 'PJ_189': PJ_189, 'PJ_574': PJ_574, 'PJ_641': PJ_641,
                'PJ_679': PJ_679, 'PJ_1976': PJ_1976, 'PJ_88': PJ_88, 'PJ_1349': PJ_1349, 'PJ_1811': PJ_1811}
				
		cluster_data_full = apogee_cluster_data[cluster_dict[name]]
		cluster_spectra_full = spectra_50[cluster_dict[name]]
		cluster_spectra_errs_full = spectra_err_50[cluster_dict[name]]
		cluster_T_full = good_T[cluster_dict[name]]
		
		#Cut red clump stars
		cluster_rgs = rgs[cluster_dict[name]]
		cluster_data = cluster_data_full[cluster_rgs]
		cluster_spectra = cluster_spectra_full[cluster_rgs]
		cluster_spectra_errs = cluster_spectra_errs_full[cluster_rgs]
		cluster_T = cluster_T_full[cluster_rgs]
		bitmask_final = full_bitmask[rgs]
		
		if red_clump == False: 	
			#Write to file
			file = h5py.File(path, 'w')
			file['apogee_cluster_data'] = cluster_data_full
			file['spectra'] = cluster_spectra_full
			file['spectra_errs'] = cluster_spectra_errs_full
			file['T'] = cluster_T_full
			file['bitmask'] = full_bitmask
			file.close()
			print(name, 'complete')
			
			return cluster_data_full, cluster_spectra_full, cluster_spectra_errs_full, cluster_T_full, full_bitmask
		
		else:
			#Write to file
			file = h5py.File(path, 'w')
			file['apogee_cluster_data'] = cluster_data
			file['spectra'] = cluster_spectra
			file['spectra_errs'] = cluster_spectra_errs
			file['T'] = cluster_T
			file['bitmask'] = bitmask_final
			file.close()
			print(name, 'complete')
			
			return cluster_data, cluster_spectra, cluster_spectra_errs, cluster_T, bitmask_final

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
		Name of desired cluster (i.e. 'PJ_26') 
	"""
	
	#If directory exists, do nothing
	if glob.glob(name):
		return None
	#If directory does not exist, make new directory
	else:
		os.mkdir(name)

def fit_func(elem, name, spectra, spectra_errs, T, dat_type, run_number, sigma_val=None):
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
    	Name of desired cluster (i.e. 'PJ_26')
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
    elem_err_200_12 = np.zeros((len(ind_12), len_spectra))
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
    	#window_file = pd.read_hdf('/Users/chloecheng/Personal/dr14_windows.hdf5', 'window_df') #Personal path - REMOVE FOR FINAL VERSION
    	window_file = pd.read_hdf('/geir_data/scr/ccheng/AST425/Personal/dr14_windows.hdf5', 'window_df') #Server path
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
    	elem_err_200 = np.zeros((len(ind), len_spectra))
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
    			#Personal path - REMOVE FOR FINAL VERSION
    			#path_dat = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    			#Server path
    			path_dat = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    	
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
    			#Personal path - REMOVE FOR FINAL VERSION
    			#path_sim = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    			#Server path
    			path_sim = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string  + '/' + name_string  + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    	
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
	
	apogee_cluster_data, spectra, spectra_errs, T, bitmask = get_spectra(arguments['--cluster'], arguments['--red_clump'])
	cluster_dir = make_directory(arguments['--cluster'])
	elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights = fit_func(arguments['--element'], arguments['--cluster'], spectra, spectra_errs, T, arguments['--type'], run_number, sigma_val)