"""Usage: run_code.py [-h][--cluster=<arg>][--num_sigma=<arg>][--red_clump=<arg>][--run_number=<arg>][--location=<arg>][--elem=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
	Number of sigma: e.g. input --num_sigma='1000'
	Red clump: e.g. input --red_clump='True'
	Run number: e.g. input --run_number='$i'
	Location: e.g. input --location='personal'
	Element: e.g. input --elem='AL'

-h  Help file
--cluster=<arg>  Cluster name
--num_sigma=<arg> Number of sigma
--red_clump=<arg> Whether to exclude red clump stars in rcsample or not
--run_number=<arg> Number of iteration
--location=<arg> Machine where the code is being run
--elem=<arg> Element name

"""

#Imports
#project scripts
import ABC
import occam_clusters_input as oc
import occam_clusters_post_process as pp
#basic math and plotting
from docopt import docopt
import numpy as np
import h5py
import time
import os

def run_everything(cluster, num_sigma, red_clump, run_number, location, elem):
    """Return the covariance matrix statistics and KS distances for every element in APOGEE in the desired cluster,
    for every simulation run.  Function also saves all final summary statistics and values of sigma to file.

    Parameters
    ----------
    cluster : str
        Name of the desired cluster (e.g. 'NGC 2682')
    num_sigma : int
        Number of simulations to run 
    red_clump : str
        If the red clump stars in rcsample are to be removed, set to True.  If all stars are to be used,
        set to False.
    run_number : int
        Number of the run by which to label files.
    location : str
        If running locally, set to 'personal'.  If running on the server, set to 'server'.
    elem : str
        Element being analyzed.

    Returns
    -------
    D_cov_all : tuple
        All covariance matrix summary statistics for all simulations
    ks_all : tuple
        All KS distances for all simulations
    """
    
    #Create cluster directory, if doesn't exist already
    cluster_dir = oc.make_directory(cluster) 
    #Get APOGEE and spectral data
    apogee_cluster_data, spectra, spectra_errs, T, bitmask = oc.get_spectra(cluster, red_clump, location)
    num_elem = 15
    num_stars = len(spectra)

    #Create synthetic spectra for each value of sigma and fit
    sigma_vals = np.random.uniform(0, 0.1, int(num_sigma))
    fake_res = []
    fake_err = []
    y_ax_psm = []
    psm_cdists = []
    fake_nanless_res = []
    final_real_spectra = []
    final_real_spectra_err = []
    for i in range(len(sigma_vals)):
        fake_dat = ABC.psm_data(num_elem, num_stars, apogee_cluster_data, sigma_vals[i], T, cluster, spectra, spectra_errs, run_number, location, elem)
        fake_res.append(fake_dat[0])
        fake_err.append(fake_dat[1])
        y_ax_psm.append(fake_dat[2])
        psm_cdists.append(fake_dat[3])
        fake_nanless_res.append(fake_dat[4])
        final_real_spectra.append(fake_dat[5])
        final_real_spectra_err.append(fake_dat[6])

    fake_res = np.array(fake_res)
    fake_err = np.array(fake_err)
    y_ax_psm = np.array(y_ax_psm)
    psm_cdists = np.array(psm_cdists)
    fake_nanless_res = np.array(fake_nanless_res)
    final_real_spectra = np.array(final_real_spectra)
    final_real_spectra_err = np.array(final_real_spectra_err)
    
    #Fit the data
    real_res = []
    real_err = []
    real_nanless_res = []
    real_nanless_err = [] 
    real_weights = [] 
    for i in range(len(sigma_vals)):
        real_dat = oc.fit_func(elem, cluster, final_real_spectra[i], final_real_spectra_err[i], T, dat_type = 'data', run_number = run_number, location = location, sigma_val=None)
        real_res.append(real_dat[0])
        real_err.append(real_dat[1])
        real_nanless_res.append(real_dat[7])
        real_nanless_err.append(real_dat[8])
        real_weights.append(real_dat[11])

    all_real_res = np.array(real_res)
    all_real_err = np.array(real_err)
    all_real_nanless_res = np.array(real_nanless_res)
    all_real_nanless_err = np.array(real_nanless_err)
    all_real_weights = np.array(real_weights)

    #Get the cumulative distributions for the data
    y_ax_real = [] 
    real_cdists = [] 
    for i in range(len(sigma_vals)):
        real_cdist_dat = pp.cum_dist(all_real_nanless_res[i], all_real_nanless_err[i])
        y_ax_real.append(real_cdist_dat[0])
        real_cdists.append(real_cdist_dat[1])
    all_y_ax_real = np.array(y_ax_real)
    all_real_cdists = np.array(real_cdists)
    
    #Calculate summary statistics
    D_cov_all = []
    ks_all = []
    for i in range(len(sigma_vals)):
        D_cov_all.append(ABC.d_cov(cluster, all_real_weights[i], all_real_res[i], all_real_err[i], fake_res[i], fake_err[i], num_stars, sigma_vals[i], elem, location, run_number))
        ks_all.append(ABC.KS(cluster, all_y_ax_real[i], all_real_cdists[i], y_ax_psm[i], psm_cdists[i], sigma_vals[i], elem, location, run_number))

    #Write to file
    timestr = time.strftime("%Y%m%d_%H%M%S") #Date and time by which to identify file
    name_string = str(cluster).replace(' ','') #Remove spaces from name of cluster
    pid = str(os.getpid())
    if location == 'personal':
        path = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + elem + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    elif location == 'server':
        path = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string + '/' + name_string + '_' + elem + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' #Server path
    file = h5py.File(path, 'w')
    file['D_cov'] = D_cov_all
    file['KS'] = ks_all
    file['sigma'] = sigma_vals
    file.close()
    
    return D_cov_all, ks_all, sigma_vals
	
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	D_cov_all, ks_all, sigma_vals = run_everything(arguments['--cluster'], arguments['--num_sigma'], arguments['--red_clump'], arguments['--run_number'], arguments['--location'], arguments['--elem'])