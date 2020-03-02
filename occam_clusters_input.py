"""Usage: occam_clusters_input.py [-h][--cluster=<arg>][--element=<arg>][--type=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
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
import os

from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from time import time
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
		spectra_50 = file['spectra'].value
		spectra_err_50 = file['spectra_errs'].value
		full_bitmask = file['bitmask'].value
		file.close()
		T = apogee_cluster_data["TEFF"]
		good_T_inds = (T > 4000) & (T < 5000)
		good_T = T[good_T_inds]
		
		print(name, ' complete.')
		
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
		
		#Want an SNR of 200
		spectra_err_200 = np.zeros_like(final_spectra_errs)
		for i in range(len(final_spectra)):
			for j in range(7514):
				if final_spectra[i][j]/final_spectra_errs[i][j] <= 200:
					spectra_err_200[i][j] = final_spectra_errs[i][j]
				else:
					spectra_err_200[i][j] = final_spectra[i][j]/200
					
		#Cut errors with SNR less than 50
		spectra_50 = np.copy(final_spectra)
		spectra_err_50 = np.copy(spectra_err_200)
		
		for i in range(len(final_spectra)):
			for j in range(7514):
				if final_spectra[i][j]/spectra_err_200[i][j] <= 50:
					spectra_50[i][j] = np.nan
					spectra_err_50[i][j] = np.nan
		
		#Write to file
		file = h5py.File(path, 'w')
		file['apogee_cluster_data'] = apogee_cluster_data
		file['spectra'] = spectra_50
		file['spectra_errs'] = spectra_err_50
		file['bitmask'] = full_bitmask
		file.close()
		print(name, 'complete')
		
	return apogee_cluster_data, spectra_50, spectra_err_50, good_T, full_bitmask

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

def make_directory(name):
    if glob.glob(name):
        return None
    else:
        os.mkdir(name)

def fit_func(elem, name, spectra, spectra_errs, T, dat_type, sigma_val=None):
    """
    Functions:
        Obtains the wavelength scale from pix2wv.
        Creates the DR14 windows from the DR12 windows for a given element.
        Obtains the indices of pixels of the absorption lines and saves the flux value and uncertainty for each star in these pixels.
        Obtains the corresponding bitmask points.
        Adjusts the uncertainty to have an SNR of 200.
        Excises bad pixels from the elemental points.
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

        #Remove nans from fits, residuals, errors, and temperatures for plotting and cumulative distribution calculation purposes
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
        weights = dr14_elem_windows[final_inds]
        normed_weights = weights/np.sum(weights)

    #If we are looking at the data
    if sigma_val == None:
            path_dat = '/Users/chloecheng/Personal/' + str(name) + '/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '.hdf5'
            if glob.glob(path_dat):
                return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
            else:
                file = h5py.File(path_dat, 'w')
                file['residuals'] = elem_res
                file['err_200'] = final_err
                file['a_param'] = elem_a
                file['b_param'] = elem_b
                file['c_param'] = elem_c
                file.close()
                return elem_res, final_err, final_points, temp_array, elem_a, elem_b, elem_c, nanless_res, nanless_err, nanless_T, nanless_points, normed_weights
    #Else for the simulations
    else:
        path_sim = '/Users/chloecheng/Personal/' + str(name) + '/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '_' + str(dat_type) + '.hdf5'
        if glob.glob(path_sim):
            file = h5py.File(path_sim, 'a')
            if glob.glob(str(sigma_val)):
            	file.close()
            else:
            	grp = file.create_group(str(sigma_val))
            	grp['residuals'] = elem_res
            	grp['err_200'] = final_err
            	grp['a_param'] = elem_a
            	grp['b_param'] = elem_b
            	grp['c_param'] = elem_c
            	file.close()
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