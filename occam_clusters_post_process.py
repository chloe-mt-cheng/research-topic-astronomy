"""Usage: occam_clusters_post_process.py [-h][--cluster=<arg>][--element=<arg>]

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
fs=16
plt.rc('font', family='serif',size=fs)

def imports(name, elem):
	path = '/Users/chloecheng/Personal/' + str(name) + '_' + str(elem) + '_' + 'fit_res' + '.hdf5'
	file = h5py.File(path, 'r')
	residuals = file['residuals'].value
	errors = file['err_200'].value
	elem_a = file['a_param'].value
	elem_b = file['b_param'].value
	elem_c = file['c_param'].value
	file.close()
	return residuals, errors

def res_cat(residuals, errors):
	all_res = np.concatenate(residuals, axis=0)
	all_err = np.concatenate(errors, axis=0)
	return all_res, all_err

def cum_dist(residuals, errors):
	all_res, all_err = res_cat(residuals, errors)
	num_res = len(all_res)
	y_ax = np.linspace(0, 1, num_res)
	cdist = np.sort(all_res/all_err)
	return y_ax, cdist

def cum_dist_plot(residuals, errors, obj, cluster, axvline=False):
    y_ax, cdist = cum_dist(residuals, errors)
    title_str = 'Cumulative Distribution of ' + str(obj) + ' from ' + str(cluster) 
    save_str = 'cdist' + '_' + str(obj) + '_' + str(cluster) + '.pdf'
    
    plt.figure(figsize=(10,8))
    plt.title(title_str, fontsize=20)
    plt.plot(cdist, y_ax, color='k')
    plt.xlabel("Normalized Residual", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-4, 4)
    if axvline==True:
        plt.axvline(0, color='r')
    plt.savefig(save_str, bbox_inches='tight')
    
def cum_dist_rev(residuals, errors):
    all_res, all_err = res_cat(residuals, errors)
    num_res = len(all_res)
    y_ax = np.linspace(0, 1, num_res)
    div_array = all_res/all_err
    rev_div_array = div_array[::-1]
    cdist = np.sort(rev_div_array)
    return y_ax, cdist

def symm_plot(residuals, errors, obj, cluster, axvline=False):
	y_ax, cdist = cum_dist(residuals, errors)
	rev_y_ax, rev_cdist = cum_dist_rev(residuals, errors)
	title_str = 'Cumulative Distribution of ' + str(obj) + ' from ' + str(cluster) 
	save_str = 'symm_overplot' + '_' + str(obj) + '_' + str(cluster) + '.pdf'
	
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
	
	residuals, errors = imports(arguments['--cluster'], arguments['--element'])
	elem_cdist_plot = cum_dist_plot(residuals, errors, arguments['--element'], arguments['--cluster'], axvline=False)
	elem_symm_plot = symm_plot(residuals, errors, arguments['--element'], arguments['--cluster'], axvline=False)