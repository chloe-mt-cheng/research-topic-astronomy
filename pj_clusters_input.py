"""Usage: pj_clusters_input.py [-h][--cluster=<arg>][--element=<arg>][--type=<arg>]

Examples:
	Cluster name: e.g. input --cluster='PJ_26'
	Element name: e.g. input --element='AL'
	Type: e.g. input --type='simulation'

-h  Help file
--cluster=<arg>  Cluster name
--element=<arg>  Element name
--type=<arg>  Data type 

"""

from docopt import docopt
#apogee package 
import apogee.tools.read as apread
from apogee.tools.path import change_dr
change_dr('16') #use DR14
#basic math and plotting
import numpy as np
import pandas as pd
import h5py
import glob
import matplotlib.pyplot as plt
import os

fs=16
plt.rc('font', family='serif',size=fs)

def get_spectra(name):
	"""Return cluster data, spectra, spectral errors, and bitmask from APOGEE.
	
	If the data file for the specified cluster already exists locally, 
	import the data from the file (cluster data, spectra, spectral errors, bitmask).
	If the data file does not exist, obtain the APOGEE spectra from a specified cluster 
	from the published_clusters.npy catalogue.
	
	Parameters
	----------
	name : str
		Name of desired cluster (i.e. 'PJ_26') 
	
	Returns
	-------
	cluster_data : structured array
		All cluster data from APOGEE
	cluster_spectra : tuple
		Array of floats representing the cleaned-up fluxes in the APOGEE spectra
	cluster_spectra_errs : tuple
		Array of floats representing the cleaned-up spectral errors from the APOGEE spectra
	cluster_T : tuple
		Array of floats representing the effective temperatures of the stars in the cluster
		between 4000K and 5000K
	full_bitmask : tuple
		Array of ints (1 or 0), cleaned in the same way as the spectra, representing the bad pixels 
		in the APOGEE_PIXMASK bitmask
	"""
	
	#path = '/Users/chloecheng/Personal/' + str(name) + '.hdf5' #Personal path
	path = '/geir_data/scr/ccheng/AST425/Personal/' + str(name) + '.hdf5' #Server path
	
	#If the data file for this cluster exists, save the data to variables
	if glob.glob(path):
		file = h5py.File(path, 'r')
		cluster_data = file['apogee_cluster_data'].value
		cluster_spectra = file['spectra'].value
		cluster_spectra_errs = file['spectra_errs'].value
		cluster_T = file['T'].value
		full_bitmask = file['bitmask'].value
		missing_spectra = file['missing_spectra'].value
		file.close()
		
		print(name, ' complete.')
		
	#If the file does not exist
	else:
		#Read in PJ catalogue data
		#apogee_cluster_data = np.load('/Users/chloecheng/Personal/published_clusters.npy') #Personal path
		apogee_cluster_data = np.load('/geir_data/scr/ccheng/AST425/Personal/published_clusters.npy') #Server path

		#Get temperatures
		T = apogee_cluster_data["TEFF"]
		
		#Get spectra for each star
		number_of_members = 360
		spectra = np.zeros((number_of_members, 7514))
		spectra_errs = np.zeros((number_of_members, 7514))
		bitmask = np.zeros((number_of_members, 7514))
		missing_spectra = []
		for s,star in enumerate(apogee_cluster_data):
			loc = star['FIELD'].decode('utf-8')
			apo = star['APOGEE_ID'].decode('utf-8')
			try:
				spectra[s] = apread.aspcapStar(loc,apo,ext=1,header=False,dr='16',aspcapWavegrid=True,telescope=star['TELESCOPE'].decode('utf-8'))
				spectra_errs[s] = apread.aspcapStar(loc,apo,ext=2,header=False,dr='16',aspcapWavegrid=True,telescope=star['TELESCOPE'].decode('utf-8'))
				bitmask[s] = apread.apStar(loc,apo,ext=3,header=False,dr='16', aspcapWavegrid=True,telescope=star['TELESCOPE'].decode('utf-8'))[1]
			#If the spectrum is missing, set bitmask to value that will be removed
			except OSError:
				bitmask[s] = -1.0
				missing_spec.append(s)
				print('missing ',star['APOGEE_ID'].decode("utf-8"))
		
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
		for i in range(len(spectra)):
			if any(spectra[i:,0]) != 0 and any(spectra[i:,-1]) != 0:
				full_spectra.append(spectra[i])
				full_spectra_errs.append(spectra_errs[i])
				full_bitmask.append(bitmask_flip[i])
		full_spectra = np.array(full_spectra)
		full_spectra_errs = np.array(full_spectra_errs)
		full_bitmask = np.array(full_bitmask)
		
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
		good_T_inds = (T > 4000) & (T < 5000)
		final_spectra = masked_spectra[good_T_inds]
		final_spectra_errs = masked_spectra_errs[good_T_inds]
		good_T = T[good_T_inds]
		apogee_cluster_data = apogee_cluster_data[good_T_inds]
		full_bitmask = full_bitmask[good_T_inds]
		
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
				
		cluster_data = apogee_cluster_data[cluster_dict[name]]
		cluster_spectra = spectra_50[cluster_dict[name]]
		cluster_spectra_errs = spectra_err_50[cluster_dict[name]]
		cluster_T = T[cluster_dict[name]]
			
		#Write to file
		file = h5py.File(path, 'w')
		file['apogee_cluster_data'] = cluster_data
		file['spectra'] = cluster_spectra
		file['spectra_errs'] = cluster_spectra_errs
		file['T'] = cluster_T
		file['bitmask'] = full_bitmask
		file['missing_spectra'] = missing_spectra
		file.close()
		print(name, 'complete')
		
	return cluster_data, cluster_spectra, cluster_spectra_errs, cluster_T, full_bitmask

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

def fit_func(elem, name, spectra, spectra_errs, T, dat_type, sigma_val=None):
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
    
    #Get windows
    #window_file = pd.read_hdf('/Users/chloecheng/Personal/dr14_windows.hdf5', 'window_df') #Personal path
    window_file = pd.read_hdf('/geir_data/scr/ccheng/AST425/Personal/dr14_windows.hdf5', 'window_df') #Server path
    dr16_elem_windows = window_file[elem].values
    
    #change_dr('12') 
    #Find the DR16 windows from the DR12 windows
    #dr12_elem_windows = window.read(elem)
    #change_dr('16')
    #dr16_elem_windows = np.concatenate((dr12_elem_windows[246:3274], dr12_elem_windows[3585:6080], dr12_elem_windows[6344:8335]))

    #Get the indices of the lines 
    ind = np.argwhere(dr16_elem_windows > 0)
    ind = ind.flatten()

    #Get the fluxes and errors from spectra
    len_spectra = len(spectra)
    elem_points = np.zeros((len(ind), len_spectra))
    elem_err = np.zeros((len(ind), len_spectra))
    elem_err_200 = np.zeros((len(ind), len_spectra))

    for i in range(0, len(ind)):
        for j in range(0, len_spectra):
            elem_points[i][j] = spectra[j][i+ind[0]]
            elem_err[i][j] = spectra_errs[j][i+ind[0]] #APOGEE measured errors

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
        return None
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
        weights = dr16_elem_windows[final_inds]
        normed_weights = weights/np.sum(weights)

    #If we are looking at the data
    if sigma_val == None:
    		#Personal path
            #path_dat = '/Users/chloecheng/Personal/' + str(name) + '/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '.hdf5'
            #Server path
            path_dat = '/geir_data/scr/ccheng/AST425/Personal/' + str(name) + '/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '.hdf5'
            #If the file exists, return variables
            if glob.glob(path_dat):
                return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
            #If the file does not exist, create it, write to it, and return variables
            else:
                file = h5py.File(path_dat, 'w')
                file['residuals'] = elem_res
                file['err_200'] = final_err
                file['a_param'] = elem_a
                file['b_param'] = elem_b
                file['c_param'] = elem_c
                file.close()
                return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
    #If we are looking at the simulations
    else:
        #Personal path
        #path_sim = '/Users/chloecheng/Personal/' + str(name) + '/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '.hdf5'
        #Server path
        path_sim = '/geir_data/scr/ccheng/AST425/Personal/' + str(name) + '/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '.hdf5'
        #If the file exists, append to it
        if glob.glob(path_sim):
            file = h5py.File(path_sim, 'a')
            #If the simulation for this sigma has already been completed, do nothing
            if glob.glob(str(sigma_val)):
            	file.close()
            #If the simulation for this sigma has not been completed, write to file
            else:
            	grp = file.create_group(str(sigma_val))
            	grp['residuals'] = elem_res
            	grp['err_200'] = final_err
            	grp['a_param'] = elem_a
            	grp['b_param'] = elem_b
            	grp['c_param'] = elem_c
            	file.close()
        #If the file does not exist, create it
        else:
            file = h5py.File(path_sim, 'w')
            grp = file.create_group(str(sigma_val))
            grp['residuals'] = elem_res
            grp['err_200'] = final_err
            grp['a_param'] = elem_a
            grp['b_param'] = elem_b
            grp['c_param'] = elem_c
            file.close()
        return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
    
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	apogee_cluster_data, spectra, spectra_errs, T, bitmask = get_spectra(arguments['--cluster'])
	cluster_dir = make_directory(arguments['--cluster'])
	elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights = fit_func(arguments['--element'], arguments['--cluster'], spectra, spectra_errs, T, arguments['--type'])