"""Usage: occam_clusters_input.py [-h][--cluster=<arg>][--element=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
	Element name: e.g. input --element='AL'

-h  Help file
--cluster=<arg>  Cluster name
--element=<arg>  Element name

"""

from docopt import docopt
#apogee package 
import apogee.tools.read as apread
from apogee.tools import _aspcapPixelLimits
from apogee.tools.path import change_dr
from apogee.tools import pix2wv
from apogee.spec import window
from apogee.tools import apStarWavegrid
from apogee.tools import bitmask as bm
change_dr('14') #use DR14
#astropy helper functions
import astropy.io.fits as afits
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
#basic math and plotting
import numpy as np
import pandas as pd
import h5py
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
fs=16
plt.rc('font', family='serif',size=fs)

def get_spectra(name):
	"""
	If the data file for the specified cluster already exists locally, import the data from the file (cluster data, spectra, spectral errors).
	If the data file does not exist, obtain the APOGEE spectra from a specified cluster from the allStar catalogue, replacing ASPCAP abundances with astroNN abundances.
	
	name:        String cluster name (i.e. 'NGC 2682') 
	"""
	path = '/Users/chloecheng/Personal/' + str(name) + '.hdf5'
	
	#If the data file for this cluster exists, save the data to variables
	if glob.glob(path):
		file = h5py.File(path, 'r')
		apogee_cluster_data = file['apogee_cluster_data'].value
		final_spectra = file['spectra'].value
		final_spectra_errs = file['spectra_errs'].value
		file.close()
		T = apogee_cluster_data["TEFF"]
		good_T_inds = (T > 4000) & (T < 5000)
		good_T = T[good_T_inds]
		
	#If the file does not exist
	else:
		#read in APOGEE catalogue data, removing duplicated stars and replacing ASPCAP with astroNN abundances
		apogee_cat = apread.allStar(use_astroNN_abundances=True)
		unique_apoids,unique_inds = np.unique(apogee_cat['APOGEE_ID'],return_index=True)
		apogee_cat = apogee_cat[unique_inds]
		
		#read in overall cluster information
		cls = afits.open('occam_cluster-DR14.fits')
		cls = cls[1].data
		
		#read in information about cluster members
		members = afits.open('occam_member-DR14.fits')
		members = members[1].data
		
		#select all members of a given cluster
		cluster_members = (members['CLUSTER']==name) & (members['MEMBER_FLAG']=='GM') #second part of the mask indicates to only use giant stars
		member_list = members[cluster_members]
		
		#find APOGEE entries for that cluster
		#numpy.in1d finds the 1D intersection between two lists. 
		#In this case we're matching using the unique APOGEE ID assigned to each star
		#The indexes given by numpy.in1d are for the first argument, so in this case the apogee catalogue
		cluster_inds = np.in1d((apogee_cat['APOGEE_ID']).astype('U100'),member_list['APOGEE_ID'])
		apogee_cluster_data = apogee_cat[cluster_inds]
		T = apogee_cluster_data["TEFF"]
		
		#get spectra for each star - apStar
		#We can use the APOGEE package to read each star's spectrum
		#We'll read in the ASPCAP spectra, which have combined all of the visits for each star and removed the spaces between the spectra
		number_of_members = len(member_list)
		spectra = np.zeros((number_of_members,7514))
		spectra_errs = np.zeros((number_of_members,7514))
		bitmask = np.zeros((number_of_members, 7514))
		for s,star in enumerate(apogee_cluster_data):
			spectra[s] = apread.aspcapStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=1,header=False,dr='14',aspcapWavegrid=True)
			spectra_errs[s] = apread.aspcapStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=2,header=False,dr='14',aspcapWavegrid=True)
			bitmask[s] = apread.apStar(star['LOCATION_ID'],star['APOGEE_ID'],ext=3,header=False,dr='14', aspcapWavegrid=True)[1]
		
		#Set all entries in bitmask to integers	
		bitmask = bitmask.astype(int)
		
		#Function for bitmask
		def bitsNotSet(bitmask,maskbits):
			"""
			Given a bitmask, returns False where any of maskbits are set 
			and True otherwise.
			bitmask:   bitmask to check
			maskbits:  bits to check if set in the bitmask
			"""
			goodLocs_bool = np.ones(bitmask.shape).astype(bool)
			for m in maskbits:
				bitind = bm.bit_set(m,bitmask)
				goodLocs_bool[bitind] = False
			return goodLocs_bool
			
		#Run code to get mask for SIG_SKYLINE
		badcombpixmask = bm.badpixmask()
		badcombpixmask += 2**bm.apogee_pixmask_int("SIG_SKYLINE")
		maskbits = bm.bits_set(badcombpixmask)
		
		#Get mask
		spec_mask = bitsNotSet(bitmask, maskbits)
		
		#Mask the spectra
		spectra_copy = spectra.copy()
		spectra_errs_copy = spectra_errs.copy()
		masked_spectra = spectra_copy*spec_mask
		masked_spectra_errs = spectra_errs_copy*spec_mask
			
		#Exclude any spectra that are all 0 
		full_spectra = []
		for i in range(len(masked_spectra)):
			if any(masked_spectra[i:,0]) != 0 and any(masked_spectra[i:,-1]) != 0:
				full_spectra.append(masked_spectra[i])
			else:
				break
		full_spectra = np.array(full_spectra)
		
		full_spectra_errs = []
		for i in range(len(masked_spectra_errs)):
			if any(masked_spectra_errs[i:,0]) != 0 and any(masked_spectra_errs[i:,-1]) != 0:
				full_spectra_errs.append(masked_spectra_errs[i])
			else:
				break
		full_spectra_errs = np.array(full_spectra_errs)
		
		#Cut stars that are outside of the temperature limits
		good_T_inds = (T > 4000) & (T < 5000)
		final_spectra = full_spectra[good_T_inds]
		final_spectra_errs = full_spectra_errs[good_T_inds]
		good_T = T[good_T_inds]
		apogee_cluster_data = apogee_cluster_data[good_T_inds]
		
		#Set 0 errors to 0.005
		for i in range(len(final_spectra)):
			for j in range(7514):
				if final_spectra_errs[i][j] == 0.0:
					final_spectra_errs[i][j] = 0.005
		
		#Write to file
		file = h5py.File(path, 'w')
		file['apogee_cluster_data'] = apogee_cluster_data
		file['spectra'] = final_spectra
		file['spectra_errs'] = final_spectra_errs
		file.close()
		
	return apogee_cluster_data, final_spectra, final_spectra_errs, good_T	

def weight_lsq(data, temp, error):
	"""
	Quadratic fit for a data-set using the weighted least-squares method from Hogg 2015. 
	
	data:        Array of fluxes of an element for all stars in a cluster
	temp:        Array of effective temperatures, obtained from apogee_cluster_data['TEFF']
	error:       Array of uncertainties on the fluxes
	"""
	try:
		Y = data.T #Data vector
		ones_column = np.ones(len(temp)) #Constant column
		A = np.column_stack((temp**2, temp, ones_column)) #Temperature matrix
		C = np.zeros((len(data), len(data))) #Covariance matrix
		np.fill_diagonal(C, error**2) #Fill covariance matrix
		C_inv = np.linalg.inv(C) #Invert covariance matrix
		#Do the matrix multiplication
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
		plt.figure()
		plt.imshow(C)
		plt.colorbar()
		print(e)
        
#Define function for residuals
def residuals(data, fit):
	"""
	Compute the residuals from a fit
	
	data:        Array of fluxes of an element for all stars in a cluster
	fit:         Array containing line of best fit
	"""
	return data - fit

#Function for fits, residuals, and cumulative distributions
def fit_func(elem, name, spectra, spectra_errs, good_T):
	"""
	Functions:
		Obtains the wavelength scale from pix2wv.
		Creates the DR14 windows from the DR12 windows for a given element.
		Obtains the indices of pixels of the absorption lines and saves the flux value and uncertainty for each star in these pixels.
		Adjusts the uncertainty to have an SNR of 200.
		Performs the quadratic fit on each pixel using weight_lsq() and computes the residuals using residuals().
		Writes the residuals and fit parameters to .csv's.
		
	elem:              String containing element name (i.e. 'AL')
	name:              String containing name of cluster (i.e. 'NGC 2682')
	full_spectra:      Array of APOGEE spectra of the cluster obtained in get_spectra()
	full_spectra_errs: Array of APOGEE spectra uncertainties of the cluster obtained in get_spectra()
	T:                 Array of effective temperatures obtained from apogee_cluster_data['TEFF']
		
	"""
	wavelength = pix2wv(np.arange(0,7514))
	change_dr('12') 
	#Find the DR14 windows from the DR12 windows
	dr12_elem_windows = window.read(elem)
	change_dr('14')
	dr14_elem_windows = np.concatenate((dr12_elem_windows[246:3274], dr12_elem_windows[3585:6080], dr12_elem_windows[6344:8335]))
	
	#Get the indices of the lines 
	ind = np.argwhere(dr14_elem_windows > 0)
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
			#Want an SNR of 200 
			if elem_points[i][j]/elem_err[i][j] <= 200:
				elem_err_200[i][j] = elem_err[i][j]
			else:
				elem_err_200[i][j] = elem_points[i][j]/200
				
	#Perform quadratic fits on each pixel
	elem_a = np.zeros(len(elem_points))
	elem_b = np.zeros(len(elem_points))
	elem_c = np.zeros(len(elem_points))
	
	for i in range(len(elem_points)):
		elem_a[i], elem_b[i], elem_c[i] = weight_lsq(elem_points[i], good_T, elem_err_200[i])
	
	elem_fits = np.zeros((len(ind), len_spectra))
	for i in range(0, len(ind)):
		elem_fits[i] = elem_a[i]*good_T**2 + elem_b[i]*good_T + elem_c[i]
		
	#Calculate residuals
	elem_res = residuals(elem_points, elem_fits)
	
	#Write residuals to a csv
	waves = wavelength[ind]
	residual_frame = pd.DataFrame(elem_res, index=waves)
	elem_err_frame = pd.DataFrame(elem_err_200, index=waves)
	filename1 = str(name) + '_' + str(elem) +  '_'  + 'residuals.csv'
	filename2 = str(name) + '_' + str(elem) +  '_'  + 'err_200.csv'
	residual_frame.to_csv(filename1)
	elem_err_frame.to_csv(filename2)
	
	#Write fit parameters to a csv
	params_frame = pd.DataFrame({'a': elem_a, 
								'b': elem_b,
								'c': elem_c})
	param_file = str(name) + '_' + str(elem) + '_' + 'fitparams.csv'
	params_frame.to_csv(param_file)
	
	path = '/Users/chloecheng/Personal/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '.hdf5'
	file = h5py.File(path, 'w')
	file['residuals'] = elem_res
	file['err_200'] = elem_err_200
	file['a_param'] = elem_a
	file['b_param'] = elem_b
	file['c_param'] = elem_c
	file.close()
	
	return elem_res, elem_err_200, elem_points, elem_a, elem_b, elem_c, ind
    
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	apogee_cluster_data, spectra, spectra_errs, good_T = get_spectra(arguments['--cluster'])
	elem_res = fit_func(arguments['--element'], arguments['--cluster'], spectra, spectra_errs, good_T)