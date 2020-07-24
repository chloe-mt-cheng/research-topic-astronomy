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
import apogee.tools.read as apread ###Read in the allStar data
from apogee.tools.path import change_dr ###Change the data-release
from apogee.spec import window ###Windows in fitting function for DR12
from apogee.tools.read import rcsample ###Remove red clumps
change_dr('14') #use DR14 ###Change to DR14
#astropy helper functions
import astropy.io.fits as afits ###Read OCCAM file
#basic math and plotting
from docopt import docopt ###Run with docopt from terminal
import numpy as np ###Numpy
import pandas as pd ###Read DR14 windows file 
import h5py ###Read/write files
import glob ###See whether file/directory exists or not
import matplotlib.pyplot as plt ###Plot for weighted least-squares function if matrix is singular
import os ###Make directory for cluster
import time ###Label files with date and time

fs=16 ###Plot fontsize
plt.rc('font', family='serif',size=fs) ###Plot fonts

def photometric_Teff(apogee_cluster_data): ###Function to compute photometric effective temperatures.  Requires AK_TARG and FE_H from apogee_cluster_data
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
    
    aktarg = apogee_cluster_data['AK_TARG'] ###Get extinction values for each star from allStar data
    #Exception for unlikely AK_TARG numbers
    for i in range(len(aktarg)): ###For number in array of extinction values
        if aktarg[i] <= -50.: ###If the value is very small
            aktarg[i] = np.nan ###Set it to NaN to be ignored later
    
    #Correct J and K for median extinction
    med_aktarg = np.nanmedian(aktarg) ###Compute the median of all of the individual extinction values (nanmedian in case values get masked out above)
    aj = med_aktarg*2.5 ###Compute the extinction factor for J (from the apogee package)
    J0 = apogee_cluster_data['J'] - aj ###Compute extinction-corrected J
    K0 = apogee_cluster_data['K'] - med_aktarg ###Compute extinction-corrected K
    
    #Get numbers needed for Teff calculation
    colour = J0 - K0 ###Get the colour you want to use to compute the temperatures (J0 - Ks0 in this case)
    metallicity = np.nanmedian(apogee_cluster_data['FE_H']) ###Compute the median of all individual metallicities (for consistency with median AK_TARG)
    b = np.array((0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013)) #Coefficients from Hernandez and Bonifacio (2009)
    
    #Calculate photometric Teff
    Teff = 5040/(b[0] + b[1]*colour + b[2]*colour**2 + b[3]*colour*metallicity + b[4]*metallicity
                + b[5]*metallicity**2) ###This should be equation 10 from Hernandez 2009, isolated for Teff
    
    return Teff

def get_spectra(name, red_clump, location): ###Function to read the allStar file and get the spectra, correct spectra for 
###small and large uncertainties, remove red clump stars
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
	if location == 'personal': ###If running on my Mac
		path = '/Users/chloecheng/Personal/' + str(name).replace(' ', '') + '.hdf5'  ###Path to folder named after cluster
	elif location == 'server': ###If running on the server
		path = '/geir_data/scr/ccheng/AST425/Personal/' + str(name).replace(' ', '') + '.hdf5'  ###Path to cluster folder
		
	#If the data file for this cluster exists, save the data to variables and return them
	if glob.glob(path): ###If the file exists
		if red_clump == 'False': ###If we're keeping all of the stars, read in the data
			file = h5py.File(path, 'r')
			apogee_cluster_data = file['apogee_cluster_data'][()]
			spectra_50 = file['spectra'][()]
			spectra_err_50 = file['spectra_errs'][()]
			good_T = file['T'][()]
			full_bitmask = file['bitmask'][()]
			file.close()
			print(name, ' complete.') ###Notification that this function is done
			return apogee_cluster_data, spectra_50, spectra_err_50, good_T, full_bitmask
		
		elif red_clump == 'True': ###If we're removing the red clumps, read in the data
			file = h5py.File(path, 'r')
			apogee_cluster_data_final = file['apogee_cluster_data'][()]
			spectra_final = file['spectra'][()]
			spectra_err_final = file['spectra_errs'][()]
			T_final = file['T'][()]
			bitmask_final = file['bitmask'][()]
			file.close()
			print(name, ' complete.') ###Notification that this function is done
			return apogee_cluster_data_final, spectra_final, spectra_err_final, T_final, bitmask_final
		
	#If the file does not exist, get the data from APOGEE
	else: ###If the file does not exist
		#Get red clump stars from rcsample
		rc_data = rcsample(dr='14') ###Get the rcsample data for DR14
		rc_stars = [] ###Empty list for the stars
		for i in range(len(rc_data)): ###Iterate through the rcsample data
			if location == 'personal': ###If running on Mac
				rc_stars.append(rc_data[i][2]) ###Append just the names of the stars
			elif location == 'server': ###If running on server
				rc_stars.append(rc_data[i][2].decode('UTF-8')) ###Append just the names of the stars (decode because on server the names are bitwise for some reason)
		rc_stars = np.array(rc_stars) ###Make list of red clump star names into array
	
		#Read in APOGEE catalogue data, removing duplicated stars and replacing ASPCAP with astroNN abundances
		apogee_cat = apread.allStar(use_astroNN_abundances=True) ###Read the allStar file, using the astroNN abundances 
		unique_apoids,unique_inds = np.unique(apogee_cat['APOGEE_ID'],return_index=True) ###Get the APOGEE IDs
		apogee_cat = apogee_cat[unique_inds] ###Get the APOGEE IDs
		
		#Read in overall cluster information
		cls = afits.open('occam_cluster-DR14.fits') ###Read in the OCCAM data
		cls = cls[1].data ###Get the cluster information
		
		#Read in information about cluster members
		members = afits.open('occam_member-DR14.fits') ###Read in the OCCAM members data
		members = members[1].data ###Get the member information
		
		#Select all members of a given cluster
		cluster_members = (members['CLUSTER']==name) & (members['MEMBER_FLAG']=='GM') #second part of the mask indicates to only use giant stars
		member_list = members[cluster_members] ###Make a list of all member stars in the cluster
		
		#Find APOGEE entries for that cluster
		#numpy.in1d finds the 1D intersection between two lists. 
		#In this case we're matching using the unique APOGEE ID assigned to each star
		#The indices given by numpy.in1d are for the first argument, so in this case the apogee catalogue
		cluster_inds = np.in1d((apogee_cat['APOGEE_ID']).astype('U100'),member_list['APOGEE_ID']) ###Get the indices of the cluster members
		apogee_cluster_data = apogee_cat[cluster_inds] ###Get the allStar data for these members
		T = photometric_Teff(apogee_cluster_data) ###Compute the photometric effective temperature
		
		#Mark red clump stars in the members of the cluster as NaNs
		cluster_stars = member_list['APOGEE_ID'] ###Get a list of all the names of the member stars in the cluster
		cluster_marked = np.copy(cluster_stars) ###Create a copy of this list to mark which stars are red clumps
		for i in range(len(cluster_stars)): ###Iterate through all of the stars in the cluster
			for j in range(len(rc_stars)): ###Iterate through all of the rcsample stars
				if cluster_stars[i] in rc_stars[j]: ###If a cluster member is also a member of the rcsample stars
					cluster_marked[i] = np.nan ###Replace the name of that star with a NaN to ignore it
		
		#Get spectra, spectral errors, and bitmask for each star - apStar
		#We can use the APOGEE package to read each star's spectrum
		#We'll read in the ASPCAP spectra, which have combined all of the visits for each star and removed the spaces between the spectra
		number_of_members = len(member_list) ###Number of members in the cluster
		spectra = np.zeros((number_of_members, 7514)) ###Create an empty array to add the spectra
		spectra_errs = np.zeros((number_of_members, 7514)) ###Create an empty array to add the spectral errors
		bitmask = np.zeros((number_of_members, 7514)) ###Create an empty array to add the bitmask
		for s,star in enumerate(apogee_cluster_data): ###Iterate through the allStar data
			spectra[s] = apread.aspcapStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=1,header=False,dr='14',aspcapWavegrid=True) ###Get the spectra
			spectra_errs[s] = apread.aspcapStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=2,header=False,dr='14',aspcapWavegrid=True) ###Get the spectral errors
			bitmask[s] = apread.apStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=3,header=False,dr='14', aspcapWavegrid=True)[1] ###Get the bitmask
		
		#Set all entries in bitmask to integers	
		bitmask = bitmask.astype(int) ###Set all entries in the bitmask to integers
		bitmask_flip = np.zeros_like(bitmask) ###Create an empty array for the bitmask with flipped entries
		for i in range(len(spectra)): ###Iterate through the number of stars in the cluster
			for j in range(7514): ###Iterate through the wavelength range
				if bitmask[i][j] == 0: ###If the bitmask entry is set to 0
					bitmask_flip[i][j] = 1 ###Set it to 1
				else: ###If the bitmask entry is not set to 0
					bitmask_flip[i][j] = 0 ###Set it to 0
		###I do this part because the unmasked entries are always 0 in the original bitmask but I think before I was maybe adding in other values to include in the mask that may not have necessarily been 1 so I just set all masked bits to 0 and all unmasked bits to 1 (or maybe this just made more sense in my head for masked to be 0 and unmasked to be 1)
					
		#Remove empty spectra
		full_spectra = [] ###Empty list for the spectra sans empty ones, list not array because we don't know how many stars will be eliminated
		full_spectra_errs = [] ###Empty list for the spectral errors sans empty spectra
		full_bitmask = [] ###Empty list for bitmask sans empty spectra
		full_T = [] ###Empty list for temperatures sans empty spectra
		full_stars = []  ###Empty list for indices of stars sans empty spectra
		for i in range(len(spectra)): ###Iterate through the number of stars
			if any(spectra[i,:] != 0): ###For all of the rows whose entries are not all 0
				full_spectra.append(spectra[i]) ###Append those spectra
				full_spectra_errs.append(spectra_errs[i]) ###Append those spectral errors
				full_bitmask.append(bitmask_flip[i]) ###Append those bitmask rows
				full_T.append(T[i]) ###Append those temperatures
				full_stars.append(i) ###Append the indices of those stars
		full_spectra = np.array(full_spectra) ###Make list into array
		full_spectra_errs = np.array(full_spectra_errs) ###Make list into array
		full_bitmask = np.array(full_bitmask) ###Make list into array
		full_T = np.array(full_T) ###Make list into array
		full_stars = np.array(full_stars) ###Make list into array
		full_marked_stars = cluster_marked[full_stars] ###Use array of stars left to index marked stars so we know which ones are red clump stars
		
		#Create array of NaNs to replace flagged values in spectra
		masked_spectra = np.empty_like(full_spectra) ###Create an empty array that is the same shape as full_spectra
		masked_spectra_errs = np.empty_like(full_spectra_errs) ###Create an empty array that is the same shape as full_spectra_errs
		masked_spectra[:] = np.nan ###Set all of the entries to NaNs
		masked_spectra_errs[:] = np.nan ###Set all of the entries to NaNs
		
		#Mask the spectra
		for i in range(len(full_spectra)): ###Iterate through the number of stars
			for j in range(7514): ###Iterate through the wavelength range
				if full_bitmask[i][j] != 0: ###If the bitmask is not 0 (i.e. if the bit is unmasked)
					masked_spectra[i][j] = full_spectra[i][j] ###Retain the value of the unmasked spectra here
					masked_spectra_errs[i][j] = full_spectra_errs[i][j] ###Retain the value of the unmasked spectral errors here 
		###All of the masked bits that were not captured by this if statement will remain NaNs and will thus be ignored 
					
		#Cut stars that are outside of the temperature limits 
		good_T_inds = (full_T > 4000) & (full_T < 5000) ###Get the indices of the temperatures that are between 4000K and 5000K
		final_spectra = masked_spectra[good_T_inds] ###Index the spectra to only keep stars that are within the temperature limits
		final_spectra_errs = masked_spectra_errs[good_T_inds] ###Index the spectral errors to only keep stars within Teff limits
		good_T = full_T[good_T_inds] ###Index the temperatures to keep only stars within Teff limits
		apogee_cluster_data = apogee_cluster_data[good_T_inds] ###Index the allStar data to keep stars only within Teff limits
		full_bitmask = full_bitmask[good_T_inds] ###Index the bitmask to keep stars only within Teff limits
		final_stars = full_marked_stars[good_T_inds] ###Index the array of red-clump-marked stars to keep only those within Teff limits
		rgs = (final_stars != 'nan') #Get indices for final red giant stars to be used 
		
		#Want an SNR of 200 so set those errors that have a larger SNR to have an SNR of 200
		spectra_err_200 = np.zeros_like(final_spectra_errs) ###Create an empty array to add corrected spectral errors to - shape will not change, just altering values
		for i in range(len(final_spectra)): ###Iterate through the stars
			for j in range(7514): ###Iterate through wavelength range
				if final_spectra[i][j]/final_spectra_errs[i][j] <= 200: ###If errors are of a reasonable size
					spectra_err_200[i][j] = final_spectra_errs[i][j] ###Leave them as they are
				else: ###If errors are too small
					spectra_err_200[i][j] = final_spectra[i][j]/200 ###Make them a bit bigger
					
		#Cut errors with SNR of less than 50
		spectra_50 = np.copy(final_spectra) ###Create a copy of the spectra to cut large error pixels
		spectra_err_50 = np.copy(spectra_err_200) ###Create a copy of the spectral errors to cut large error pixels
		
		for i in range(len(final_spectra)): ###Iterate through stars
			for j in range(7514): ###Iterate through wavelength range
				if final_spectra[i][j]/spectra_err_200[i][j] <= 50: ###If an error is too big
					spectra_50[i][j] = np.nan ###Set the corresponding entry in the spectra to be a NaN, will be ignored
					spectra_err_50[i][j] = np.nan ###Set the corresponding entry in the spectral errors to be a NaN, will be ignored
		
		#Cut red clumps
		logg = apogee_cluster_data['LOGG'] ###Get the logg values for the cluster (all corrections have been applied)
		apogee_cluster_data_final = apogee_cluster_data[rgs] ###Get the allStar data for the RGB stars only (no red clumps)
		spectra_final = spectra_50[rgs] ###Get the spectra for the RGB stars only
		spectra_err_final = spectra_err_50[rgs] ###Get the spectral errors for the RGB stars only
		T_final = good_T[rgs] ###Get the temperatures for the RGB stars only
		bitmask_final = full_bitmask[rgs] ###Get the bitmask for the RGB stars only
		
		if red_clump == 'False': ###If we are looking at all of the stars, save all data before red clumps were cut to file
			#Write to file
			file = h5py.File(path, 'w')
			file['apogee_cluster_data'] = apogee_cluster_data
			file['spectra'] = spectra_50
			file['spectra_errs'] = spectra_err_50
			file['T'] = good_T
			file['bitmask'] = full_bitmask
			file.close()
			print(name, 'complete') ###Notification that this function is done
			
			return apogee_cluster_data, spectra_50, spectra_err_50, good_T, full_bitmask
		
		elif red_clump == 'True': ###If we are removing the red clump stars, save the data after red clumps cut to file
			#Write to file 
			file = h5py.File(path, 'w')
			file['apogee_cluster_data'] = apogee_cluster_data_final
			file['spectra'] = spectra_final
			file['spectra_errs'] = spectra_err_final
			file['T'] = T_final
			file['bitmask'] = bitmask_final
			file.close()
			print(name, 'complete') ###Notification that this function is done
			
			return apogee_cluster_data_final, spectra_final, spectra_err_final, T_final, bitmask_final

def weight_lsq(data, temp, error): ###Function to get the fit parameters using the weighted least-squares fitting method
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
	
	try: ###Try the fit
		#Run the fitting algorithm on the data
		Y = data.T #Data vector
		ones_column = np.ones(len(temp)) #Constant column
		A = np.column_stack((temp**2, temp, ones_column)) #Temperature matrix
		C = np.zeros((len(data), len(data))) #Covariance matrix
		np.fill_diagonal(C, error**2) #Fill covariance matrix
		C_inv = np.linalg.inv(C) #Invert covariance matrix
		
		#Perform the matrix multiplication ###checked this with the Hogg paper, seems okay
		step1 = np.dot(A.T, C_inv)
		step2 = np.dot(step1, A)
		step3 = np.dot(A.T, C_inv)
		step4 = np.dot(step3, Y)
		
		#Calculate the parameters
		parameters = np.dot(np.linalg.inv(step2), step4) 
		#Isolate the parameters ###I'm not sure if these are in the right order?
		a = parameters[0]
		b = parameters[1]
		c = parameters[2]
		return a, b, c 
		
	except np.linalg.LinAlgError as e: ###If matrix is not diagonal, catch
		#Catch if covariance matrix is non-diagonal
		plt.figure()
		plt.imshow(C)
		plt.colorbar()
		print(e)
        
def residuals(data, fit): ###Function to compute the fit residuals 
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
	
	return data - fit ###Subtract the fit from the data

def make_directory(name): ###Create a directory for the cluster being examined
	"""Create a new directory for a cluster, if it does not already exist.
	
	Parameters
	----------
	name : str
		Name of desired cluster (i.e. 'NGC 2682') 
	"""
	
	#Strip spaces from cluster name
	name_string = str(name).replace(' ', '') ###Replace spaces with nothing in name
	
	#If directory exists, do nothing
	if glob.glob(name_string): ###If the directory exists
		return None ###Exit
	#If directory does not exist, make new directory
	else: ###If the directory does not exist
		os.mkdir(name_string) ###Make a directory named after the cluster with no spaces in the name

def fit_func(elem, name, spectra, spectra_errs, T, dat_type, run_number, location, sigma_val=None): ###Fitting function 
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
    
    change_dr('12') ###Switch data-release to 12
    #Find the DR14 windows from the DR12 windows
    dr12_elem_windows = window.read(elem) ###Read in the DR12 windows for the element in question
    change_dr('14') ###Switch back to DR14
    dr14_elem_windows_12 = np.concatenate((dr12_elem_windows[246:3274], dr12_elem_windows[3585:6080], dr12_elem_windows[6344:8335])) ###Fit the dr12 windows to dr14 ("hacked" dr14 windows)
    normalized_dr14_elem_windows_12 = (dr14_elem_windows_12 - np.nanmin(dr14_elem_windows_12))/(np.nanmax(dr14_elem_windows_12) - np.nanmin(dr14_elem_windows_12)) ###Normalize the hacked dr14 windows to 1 
    
    #Get the indices of the lines 
    ind_12 = np.argwhere(normalized_dr14_elem_windows_12 > 0) ###Get the indices of all of the pixels of the absorption lines of the element in question
    ind_12 = ind_12.flatten() ###Get rid of the extra dimension produced by np.argwhere - might be able to streamline this by getting rid of argwhere
    
    #Get the fluxes and errors from spectra ###I probably don't need to do this part, I only really need the final indices for pixels with more than 5 points to cut out 70% of the dr12 windows from dr14 after the next if else statement - need points for >= 5 part, but don't need errors
    len_spectra = len(spectra) ###Number of stars
    elem_points_12 = np.zeros((len(ind_12), len_spectra)) ###Array for values of the points in the spectra that are at the indices of the elemental lines in DR12
    elem_err_12 = np.zeros((len(ind_12), len_spectra)) ###Array for values of the errors in the spectra that are at the indices of the elemental lines in DR12 
    for i in range(0, len(ind_12)): ###Iterate through the DR12 elemental indices
    	for j in range(0, len_spectra): ###Iterate through the stars
    		elem_points_12[i][j] = spectra[j][ind_12[i]] ###Get the values of the points in the spectra at these indices in DR12
    		elem_err_12[i][j] = spectra_errs[j][ind_12[i]] #APOGEE measured errors ###Get the values of the errors in the spectra at these indices in DR12
    		
    #Use only pixels with more than 5 points
    final_points_12 = [] ###Empty list for the final DR12 points 
    final_err_12 = [] ###Empty list for the final DR12 errors
    final_inds_12 = [] ###Empty list for the final DR12 elemental line indices
    for i in range(len(elem_points_12)): ###Iterate through the DR12 flux points
    	if np.count_nonzero(~np.isnan(elem_points_12[i])) >= 5: ###If the number of points in each pixel that are not NaNs is greater than or equal to 5
    		final_points_12.append(elem_points_12[i]) ###Append those points
    		final_err_12.append(elem_err_12[i]) ###Append those errors
    		final_inds_12.append(ind_12[i]) ###Append those indices
    final_points_12 = np.array(final_points_12) ###Make into array
    final_err_12 = np.array(final_err_12) ###Make into array
    final_inds_12 = np.array(final_inds_12) ###Make into array
    if len(final_inds_12) == 0: ###If no indices are left (i.e. if there are less than 5 points in every pixel)
    	print('Warning: less than 5 points for every pixel, skipping ', elem) ###Skip and don't finish this element
    else: ###If there are enough points left
    	dr12_weights = normalized_dr14_elem_windows_12[final_inds_12] ###Get all of the weights of the elemental pixels for DR12 
    	sorted_dr12_weights = np.sort(dr12_weights) ###Sort these weights from smallest to largest
    	
    	#Get windows
    	if location == 'personal': ###If running on Mac
    		window_file = pd.read_hdf('/Users/chloecheng/Personal/dr14_windows.hdf5', 'window_df') ###Get file I made for DR14 windows
    	elif location == 'server': ###If running on the server
    		window_file = pd.read_hdf('/geir_data/scr/ccheng/AST425/Personal/dr14_windows.hdf5', 'window_df') ###Get file I made for DR14 windows
    		
    	dr14_elem_windows_14 = window_file[elem].values ###Get the DR14 windows for the element in question
    	normalized_dr14_elem_windows_14 = (dr14_elem_windows_14 - np.min(dr14_elem_windows_14))/(np.max(dr14_elem_windows_14) - np.min(dr14_elem_windows_14)) ###Normalize these windows to 1 
    	
    	#Get the indices of the lines 
    	if elem == 'C' or elem == 'N' or elem == 'FE': ###If we're looking at one of the elements with order ~1000 pixels
    		ind = np.argwhere(normalized_dr14_elem_windows_14 > np.min(sorted_dr12_weights[int(len(sorted_dr12_weights)*0.7):])) ###Get rid of the smallest 70% of the DR12 pixels
    	else: ###For all of the other elements
    		ind = np.argwhere(normalized_dr14_elem_windows_14 > 0) ###Get all of the pixels
    	ind = ind.flatten() ###Get rid of the extra dimension from argwhere (try to streamline this)
    	
    	#Get the fluxes and errors from spectra
    	#Limits of DR12 detectors
    	dr12_d1_left = 322 ###Left limit of detector 1
    	dr12_d1_right = 3242 ###Right limit of detector 1
    	dr12_d2_left = 3648 ###Left limit of detector 2
    	dr12_d2_right = 6048 ###Right limit of detector 2
    	dr12_d3_left = 6412 ###Left limit of detector 3
    	dr12_d3_right = 8306 ###Right limit of detector 3
    	
    	elem_points = np.zeros((len(ind), len_spectra)) ###Make an empty array to hold the values of the spectra at the elemental indices
    	elem_err = np.zeros((len(ind), len_spectra)) ###Make an empty array to hold the values of the spectral errors at the elemental indices
    	for i in range(0, len(ind)): ###Iterate through the elemental indices
    		for j in range(0, len_spectra): ###Iterate through the number of stars
    			###If the indices are outside of the bounds of the DR12 detectors (these bounds should be right)
    			if ind[i] < dr12_d1_left or (dr12_d1_right < ind[i] < dr12_d2_left) or (dr12_d2_right < ind[i] < dr12_d3_left) or ind[i] > dr12_d3_right:
    				elem_points[i][j] = np.nan ###Set the point to NaN and ignore
    				elem_err[i][j] = np.nan ###Set the error to NaN and ignore 
    			else: ###If the indices are within the bounds of the DR12 detectors
    				elem_points[i][j] = spectra[j][ind[i]] ###Get the corresponding point in the spectra
    				elem_err[i][j] = spectra_errs[j][ind[i]] #APOGEE measured errors ###Get the corresponding point in the spectral errors
    	
    	#Use only pixels with more than 5 points
    	final_points = [] ###Make an array for the final set of spectral points
    	final_err = [] ###Make an array for the final set of spectral error points
    	final_inds = [] ###Make an array for the final set of elemental indices
    	for i in range(len(elem_points)): ###Iterate through the points we just obtained
    		if np.count_nonzero(~np.isnan(elem_points[i])) >= 5: ###If the number of non-NaNs in the pixel is greater than or equal to 5
    			final_points.append(elem_points[i]) ###Append the points
    			final_err.append(elem_err[i]) ###Append the errors
    			final_inds.append(ind[i]) ###Append the indices
    	final_points = np.array(final_points) ###Make into array
    	final_err = np.array(final_err) ###Make into array
    	final_inds = np.array(final_inds) ###Make into array
    	
    	if len(final_points) == 0: ###If all pixels have less than 5 points
    		print('Warning: less than 5 points for every pixel, skipping ', elem) ###Skip the element and end here
    	else: ###If there are some pixels remaining
    	
    		#Create an appropriately-sized array of temperatures to mask as well
    		temp_array = np.full((final_points.shape), T) ###Create an array of the temperatures that is the same size as the spectral points, but each row is the same set of temperatures 
    		for i in range(0, len(final_points)): ###Iterate through the spectral points
    			for j in range(0, len_spectra): ###Iterate through the stars
    				if np.isnan(final_points[i][j]): ###If the point is a NaN
    					temp_array[i][j] = np.nan ###Mask the corresponding point in the temperature array
    					
    		#Do fits with non-nan numbers
    		nanless_inds = np.isfinite(final_points) ###Get the indices of the non-NaN points
    		fits = [] ###Create an empty list for the fit parameters
    		for i in range(len(final_points)): ###Iterate through the spectral points
    			fits.append(weight_lsq(final_points[i][nanless_inds[i]], temp_array[i][nanless_inds[i]], final_err[i][nanless_inds[i]])) ###Fit using weight_lsq function and all points that are not NaNs 
    		for i in range(len(fits)): ###Iterate through the fits
    			fits[i] = np.array(fits[i]) ###Make each sub-list into an array
    		fits = np.array(fits) ###Make the whole list into an array
    		###Check the order of these as well - I think it should be fine if you just change the order in weight_lsq bc still return a, b, c in the order 0, 1, 2
    		elem_a = fits[:,0] ###Get the a-parameter
    		elem_b = fits[:,1] ###Get the b-parameter
    		elem_c = fits[:,2] ###Get the c-parameter 
    		
    		elem_fits = np.zeros_like(final_points) ###Create an array to save the actual fits
    		for i in range(0, len(final_points)): ###Iterate through the points
    			elem_fits[i] = elem_a[i]*temp_array[i]**2 + elem_b[i]*temp_array[i] + elem_c[i] ###Fit quadratically 
    			
    		#Calculate residuals
    		elem_res = residuals(final_points, elem_fits) ###Calculate the fit residuals
    		
    		#Remove nans from fits, residuals, errors, and temperatures for plotting and cumulative distribution 
    		#calculation purposes
    		nanless_fits = [] ###Create an empty list for the nanless fits
    		nanless_res = [] ###Create an empty list for the nanless fit residuals
    		nanless_err = [] ###Create an empty list for the nanless errors
    		nanless_T = [] ###Create an empty list for the nanless temperatures
    		nanless_points = [] ###Create an empty list for the nanless spectral points
    		for i in range(len(final_points)): ###Iterate through the spectral points
    			nanless_fits.append(elem_fits[i][nanless_inds[i]]) ###Append the nanless fits
    			nanless_res.append(elem_res[i][nanless_inds[i]]) ###Append the nanless residuals
    			nanless_err.append(final_err[i][nanless_inds[i]]) ###Append the nanless errors
    			nanless_T.append(temp_array[i][nanless_inds[i]]) ###Append the nanless temperatures
    			nanless_points.append(final_points[i][nanless_inds[i]]) ###Append the nanless points
    		for i in range(len(final_points)): ###Turn all sub lists into arrays 
    			nanless_fits[i] = np.array(nanless_fits[i])
    			nanless_res[i] = np.array(nanless_res[i])
    			nanless_err[i] = np.array(nanless_err[i])
    			nanless_T[i] = np.array(nanless_T[i])
    			nanless_points[i] = np.array(nanless_points[i])
    		nanless_fits = np.array(nanless_fits) ###Turn all lists into arrays
    		nanless_res = np.array(nanless_res)
    		nanless_err = np.array(nanless_err)
    		nanless_T = np.array(nanless_T)
    		nanless_points = np.array(nanless_points)
    		
    		#Get the weights for later
    		weights = normalized_dr14_elem_windows_14[final_inds] ###Get the weights fo the DR14 lines that we're using
    		normed_weights = weights/np.sum(weights) ###Normalize the weights
    		
    		#File-saving 
    		#If we are looking at the data
    		timestr = time.strftime("%Y%m%d_%H%M%S") ###date_time string
    		name_string = str(name).replace(' ', '') ###cluster name, remove space
    		pid = str(os.getpid()) ###PID string
    		if sigma_val == None: ###IF we are looking at the data
    			if location == 'personal': ###If running on Mac
    				path_dat = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' ###Use this path
    			elif location == 'server': ###If running on server
    				path_dat = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' ###Use this path
    	
    			#If the file exists, output the desired variables
    			if glob.glob(path_dat): ###If the file already exists, don't write anything 
    				return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
    			#If the file does not exist, create file and output the desired variables
    			else: ###If the file does not exist, write all fitting information 
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
    		else: ###If we are looking at a simulation
    			if location == 'personal': ###If running from Mac
    				path_sim = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' ###Use this path
    			elif location == 'server': ###If running on server
    				path_sim = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string  + '/' + name_string  + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' ###Use this path
    	
    			#If the file exists, append to the file
    			if glob.glob(path_sim): ###If the file exists
    				file = h5py.File(path_sim, 'a') ###Append to the file
    				#If the group for the particular value of sigma exists, don't do anything
    				if glob.glob(str(sigma_val)): ###If this value of sigma has already been tested, don't write anything 
    					file.close()
    				#If not, append a new group to the file for the particular value of sigma
    				else: ###If it has not been tested, write all fitting information to a group named after the value of sigma
    					grp = file.create_group(str(sigma_val))
    					grp['points'] = final_points
    					grp['residuals'] = elem_res
    					grp['err_200'] = final_err
    					grp['a_param'] = elem_a
    					grp['b_param'] = elem_b
    					grp['c_param'] = elem_c
    					file.close()
    			#If the file does not exist, create a new file
    			else: ###If the file does not exist, write all of the fitting information to a group named after the value of sigma
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
    
if __name__ == '__main__': ###Docopt stuff
	arguments = docopt(__doc__)
	
	###Get the allStar data and spectra
	apogee_cluster_data, spectra, spectra_errs, T, bitmask = get_spectra(arguments['--cluster'], arguments['--red_clump'], arguments['--location'])
	###Make a directory for the cluster
	cluster_dir = make_directory(arguments['--cluster'])
	###Do the fits
	elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights = fit_func(arguments['--element'], arguments['--cluster'], spectra, spectra_errs, T, arguments['--type'], run_number, arguments['--location'], sigma_val)