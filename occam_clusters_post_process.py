"""Usage: occam_clusters_post_process.py [-h][--cluster=<arg>][--element=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
	Element name: e.g. input --element='AL'

-h  Help file
--cluster=<arg>  Cluster name
--element=<arg>  Element name
"""

#Imports
#apogee package 
from apogee.tools.path import change_dr
change_dr('14') #use DR14
#basic math and plotting
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import time
from docopt import docopt

fs=16
plt.rc('font', family='serif',size=fs)

def res_cat(residuals, errors):
	"""Return the all residuals and errors from an element concatenated.
	
	Parameters
	----------
	residuals : tuple
		Array of floats representing the residuals of the quadratic fits
	errors : tuple
		Array of floats representing the spectral errors corresponding to the residuals
	
	Returns
	-------
	all_res : tuple
		One-dimensional array of all concatenated residuals
	all_err : tuple
		One-dimensional array of all concatenated errors
	"""
	
	all_res = np.concatenate(residuals, axis=0)
	all_err = np.concatenate(errors, axis=0)
	return all_res, all_err

def cum_dist(residuals, errors):
	"""Return the cumulative distribution of the normalized residuals of the fit.
	
	Parameters
	----------
	residuals : tuple
		Array of floats representing the residuals of the quadratic fits
	errors : tuple
		Array of floats representing the spectral errors corresponding to the residuals
	
	Returns
	-------
	y_ax : tuple
		One-dimensional array containing values from 0 to 1, the same size as cdist
	cdist : tuple
		One-dimensional array containing the sorted, normalized fit residuals
	"""
	
	all_res, all_err = res_cat(residuals, errors) #Concatenate residuals and errors
	num_res = len(all_res)
	y_ax = np.linspace(0, 1, num_res)
	cdist = np.sort(all_res/all_err)
	
	return y_ax, cdist

def cum_dist_plot(cluster, elem, residuals, errors, obj, axvline=False, sigma_val=None):
	"""Plot the cumulative distribution of the normalized residuals.
	
	Parameters
	----------
	cluster : str
		Name of the desired cluster (e.g. 'NGC 2682')
	elem : str
		Name of the desired element (e.g. 'AL')
	residuals : tuple
		Array of floats representing the residuals of the quadratic fits
	errors : tuple
		Array of floats representing the spectral errors corresponding to the residuals
	obj : str
		Indicates what the residuals are from (e.g. data, simulation, star)
	axvline : bool, optional
		Plots a vertical line at zero to show where centre of distribution should be
	sigma_val : float, optional
		Indicates the value of sigma being used for the simulation in question, if applicable (default is None)
	"""
	
	#Compute the cumulative distribution
	y_ax, cdist = cum_dist(cluster, elem, obj, residuals, errors, sigma_val=None) 
	title_str = 'Cumulative Distribution of ' + str(obj) + ' from ' + str(cluster) 
	save_str = 'cdist' + '_' + str(obj) + '_' + str(cluster).replace(' ','') + '.pdf'
	
	#Plot the cumulative distribution
	plt.figure(figsize=(10,8))
	plt.title(title_str, fontsize=20)
	plt.plot(cdist, y_ax, color='k')
	plt.xlabel("Normalized Residual", fontsize=15)
	plt.ylabel("Probability", fontsize=15)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlim(-4, 4)
	#If you want a vertical line at 0, plot it
	if axvline==True:
		plt.axvline(0, color='r')
	plt.savefig(save_str, bbox_inches='tight')
    
def cum_dist_rev(residuals, errors):
	"""Return a mirror image of the cumulative distribution to ensure that the cumulative distribution
	is symmetric.
	
	Parameters 
	----------
	residuals : tuple
		Array of floats representing the residuals of the quadratic fits
	errors : tuple
		Array of floats representing the spectral errors corresponding to the residuals
	"""
	
	all_res, all_err = res_cat(residuals, errors) #Concatenate errors and residuals
	num_res = len(all_res)
	y_ax = np.linspace(0, 1, num_res)
	normalized_res = all_res/all_err
	rev_norm_res = normalized_res[::-1] #Reverse normalized residuals
	cdist = np.sort(rev_norm_res)
	return y_ax, cdist

def symm_plot(cluster, elem, residuals, errors, obj, axvline=False, sigma_val=None):
	"""Plot the cumulative distribution and the mirror image of the cumulative distribution 
	to ensure that the cumulative distributionis symmetric.
	
	Parameters
	----------
	cluster : str
		Name of the desired cluster (e.g. 'NGC 2682')
	elem : str
		Name of the desired element (e.g. 'AL')
	residuals : tuple
		Array of floats representing the residuals of the quadratic fits
	errors : tuple
		Array of floats representing the spectral errors corresponding to the residuals
	obj : str
		Indicates what the residuals are from (e.g. data, simulation, star)
	axvline : bool, optional
		Plots a vertical line at zero to show where centre of distribution should be
	sigma_val : float, optional
		Indicates the value of sigma being used for the simulation in question, if applicable (default is None)
	"""
	
	y_ax, cdist = cum_dist(cluster, elem, obj, residuals, errors, sigma_val=None) #Compute cumulative distribution
	rev_y_ax, rev_cdist = cum_dist_rev(residuals, errors) #Reverse cumulative distribution
	title_str = 'Cumulative Distribution of ' + str(obj) + ' from ' + str(cluster) 
	save_str = 'symm_overplot' + '_' + str(obj) + '_' + str(cluster).replace(' ','') + '.pdf'
	
	#Plot the cumulative distributions
	plt.figure(figsize=(10,8))
	plt.title(title_str, fontsize=20)
	plt.plot(cdist, y_ax, color='k', label='Sorted Data')
	plt.plot(rev_cdist, rev_y_ax, color='b', linestyle='dotted', linewidth=5, label='Reverse-Sorted Data')
	plt.xlabel("Normalized Residual", fontsize=15)
	plt.ylabel("Probability", fontsize=15)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlim(-4, 4)
	plt.legend()
	if axvline==True:
		plt.axvline(0, color='r')
	plt.savefig(save_str, bbox_inches='tight')
    
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	y_ax, cdist = cum_dist(residuals, errors)
	cdist_dir = make_directory_cdist(arguments['--cluster'])
	elem_cdist_plot = cum_dist_plot(residuals, errors, arguments['--element'], arguments['--cluster'], axvline=False)
	elem_symm_plot = symm_plot(residuals, errors, arguments['--element'], arguments['--cluster'], axvline=False)