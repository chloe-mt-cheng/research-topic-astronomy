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
from apogee.tools.path import change_dr ###Change the data release
change_dr('14') #use DR14
#basic math and plotting
import numpy as np ###Numpy
import h5py ###For reading and writing files - I don't think I'm actually writing anything here anymore so can probably take this out
import glob ###For checking if a file already exists - I don't think I'm actually writing anything here anymore so can probably take this out
import matplotlib.pyplot as plt ###For the plotting functions
from matplotlib.colors import LogNorm ###For the plotting functions
import os ###Making directory - not sure if I need this anymore
import time ###Labelling files - not sure if I need this anymore
from docopt import docopt ###Docopt to run in terminal

fs=16
plt.rc('font', family='serif',size=fs)

def res_cat(residuals, errors): ###Function to concatenate residuals and errors to feed into cumulative distribution later
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
	
	all_res = np.concatenate(residuals, axis=0) ###Concatenate the residuals
	all_err = np.concatenate(errors, axis=0) ###Concatenate the errors 
	return all_res, all_err

def cum_dist(residuals, errors): ###Function to compute the cumulative distribution 
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
	
	all_res, all_err = res_cat(residuals, errors) #Concatenate residuals and errors ###Calls on the res_cat function above
	num_res = len(all_res) ###Get the length of the concatenated residuals to make the y-axis length right 
	y_ax = np.linspace(0, 1, num_res) ###Make the y-axis from 0 to 1, length of the residuals
	cdist = np.sort(all_res/all_err) ###Make the cumulative distribution by normalizing the residuals and sorting them
	
	return y_ax, cdist

def cum_dist_plot(cluster, elem, residuals, errors, obj, axvline=False, sigma_val=None): ###Plot the cumulative distribution - I don't really use this function
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
	y_ax, cdist = cum_dist(residuals, errors) ###Compute the cumulative distirbution
	title_str = 'Cumulative Distribution of ' + str(obj) + ' from ' + str(cluster) ###Make a string for the title of the plot
	save_str = 'cdist' + '_' + str(obj) + '_' + str(cluster).replace(' ','') + '.pdf' ###Make a string under which to save the plot
	
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
    
def cum_dist_rev(residuals, errors): ###Function to plot a mirror image of the cumulative distribution to ensure symmetry - I don't really use this function
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

def symm_plot(cluster, elem, residuals, errors, obj, axvline=False, sigma_val=None): ###Plot the mirror image
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
    
if __name__ == '__main__': ###Docopt stuff - I don't really need this
	arguments = docopt(__doc__)
	
	y_ax, cdist = cum_dist(residuals, errors)
	cdist_dir = make_directory_cdist(arguments['--cluster'])
	elem_cdist_plot = cum_dist_plot(residuals, errors, arguments['--element'], arguments['--cluster'], axvline=False)
	elem_symm_plot = symm_plot(residuals, errors, arguments['--element'], arguments['--cluster'], axvline=False)