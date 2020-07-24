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
import ABC ###Script for generating fake spectra, fitting it, calculating summary statistics
import occam_clusters_input as oc ###Script for reading in data and fit function
import occam_clusters_post_process as pp ###Script for calculating cumulative distribution
#basic math and plotting
from docopt import docopt ###Docopt for running in terminal
import numpy as np ###Numpy
import h5py ###File saving
import time ###For labelling files
import os ###Making directories

def run_everything(cluster, num_sigma, red_clump, run_number, location, elem): ###Function to run the entire algorithm
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
    cluster_dir = oc.make_directory(cluster) ###Make a directory named after the cluster 
    #Get APOGEE and spectral data
    apogee_cluster_data, spectra, spectra_errs, T, bitmask = oc.get_spectra(cluster, red_clump, location) ###Get the allStar data and spectra
    num_elem = 15 ###Number of elements in APOGEE
    num_stars = len(spectra) ###Number of stars in the cluster

    #Create synthetic spectra for each value of sigma and fit
    sigma_vals = np.random.uniform(0, 0.1, int(num_sigma)) ###Create an array of sigma values between 0 and 0.1 dex that are randomly drawn from a uniform 
    ###distribution, the size of the number of simulations that you want to run at once
    fake_res = [] ###Empty list for the fake residuals
    fake_err = [] ###Empty list for the fake errors
    y_ax_psm = [] ###Empty list for the y-axis for the fake cumulative distributions
    psm_cdists = [] ###Empty list for the fake cumulative distributions
    fake_nanless_res = [] ###Empty list for the fake residuals with NaNs removed
    final_real_spectra = [] ###Empty list for the observed spectra that are masked in the same way as the fake spectra
    final_real_spectra_err = [] ###Empty list for the observed spectral errors that are masked in the same way as the fake spectra
    for i in range(len(sigma_vals)): ###Iterate through the number of simulations you want to run
    	###Run the psm_data function from ABC.py to get the fake fits, etc.
        fake_dat = ABC.psm_data(num_elem, num_stars, apogee_cluster_data, sigma_vals[i], T, cluster, spectra, spectra_errs, run_number, location, elem)
        fake_res.append(fake_dat[0]) ###Get the fake residuals
        fake_err.append(fake_dat[1]) ###Get the fake errors
        y_ax_psm.append(fake_dat[2]) ###Get the y-axis for the fake cumulative distributions
        psm_cdists.append(fake_dat[3]) ###Get the fake cumulative distributions
        fake_nanless_res.append(fake_dat[4]) ###Get the fake residuals with no NaNs
        final_real_spectra.append(fake_dat[5]) ###Get the observed spectra that are masked in the same way as the fake spectra
        final_real_spectra_err.append(fake_dat[6]) ###Get the observed spectral errors that are masked in the same way as the fake spectra

    fake_res = np.array(fake_res) ###Make into array
    fake_err = np.array(fake_err) ###Make into array
    y_ax_psm = np.array(y_ax_psm) ###Make into array
    psm_cdists = np.array(psm_cdists) ###Make into array
    fake_nanless_res = np.array(fake_nanless_res) ###Make into array
    final_real_spectra = np.array(final_real_spectra) ###Make into array
    final_real_spectra_err = np.array(final_real_spectra_err) ###Make into array
    
    #Fit the data
    real_res = [] ###Empty list for the real residuals
    real_err = [] ###Empty list for the real errors
    real_nanless_res = [] ###Empty list for the real residuals with no NaNs
    real_nanless_err = [] ###Empty list for the real errors with no NaNs
    real_weights = []  ###Empty list for the weights of the windows for the element
    for i in range(len(sigma_vals)): ###Iterate through the number of simulations
    	###Run the fit_func function from occam_clusters_input.py to get fits for real data, using the observed spectra and errors masked in the same way as the simulations
        real_dat = oc.fit_func(elem, cluster, final_real_spectra[i], final_real_spectra_err[i], T, dat_type = 'data', run_number = run_number, location = location, sigma_val=None)
        real_res.append(real_dat[0]) ###Get the real residuals
        real_err.append(real_dat[1]) ###Get the real errors
        real_nanless_res.append(real_dat[7]) ###Get the real residuals with no NaNs
        real_nanless_err.append(real_dat[8]) ###Get the real errors with no NaNs
        real_weights.append(real_dat[11]) ###Get the weights of the windows for the element

    all_real_res = np.array(real_res) ###Make into array
    all_real_err = np.array(real_err) ###Make into array
    all_real_nanless_res = np.array(real_nanless_res) ###Make into array
    all_real_nanless_err = np.array(real_nanless_err) ###Make into array
    all_real_weights = np.array(real_weights) ###Make into array

    #Get the cumulative distributions for the data
    y_ax_real = [] ###Empty list for y-axis for real cumulative distributions
    real_cdists = [] ###Empty list for real cumulative distributions
    for i in range(len(sigma_vals)): ###Iterate through the number of simulations
        real_cdist_dat = pp.cum_dist(all_real_nanless_res[i], all_real_nanless_err[i]) ###Compute the cumulative distributions using the cum_dist function from occam_clusters_post_process.py
        y_ax_real.append(real_cdist_dat[0]) ###Get the y-axes for the real cumulative distributions
        real_cdists.append(real_cdist_dat[1]) ###Get the real cumulative distributions
    all_y_ax_real = np.array(y_ax_real) ###Make into array
    all_real_cdists = np.array(real_cdists) ###Make into array
    
    #Calculate summary statistics
    D_cov_all = [] ###Empty list for the delta covariance statistics
    ks_all = [] ###Empty list for the KS distance statistics
    for i in range(len(sigma_vals)): ###Iterate through the simulations
    	###Compute the delta covariance statistics
        D_cov_all.append(ABC.d_cov(cluster, all_real_weights[i], all_real_res[i], all_real_err[i], fake_res[i], fake_err[i], num_stars, sigma_vals[i], elem, location, run_number))
        ###Compute the KS distance statistics
        ks_all.append(ABC.KS(cluster, all_y_ax_real[i], all_real_cdists[i], y_ax_psm[i], psm_cdists[i], sigma_vals[i], elem, location, run_number))

    #Write to file
    timestr = time.strftime("%Y%m%d_%H%M%S") #Date and time by which to identify file
    name_string = str(cluster).replace(' ','') #Remove spaces from name of cluster
    pid = str(os.getpid()) ###PID for file labelling
    if location == 'personal': ###If running on Mac
        path = '/Users/chloecheng/Personal/run_files/' + name_string + '/' + name_string + '_' + elem + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5'
    elif location == 'server': ###If running on server
        path = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + name_string + '/' + name_string + '_' + elem + '_' + timestr + '_' + pid + '_' + str(run_number) + '.hdf5' #Server path
    file = h5py.File(path, 'w') ###Write file
    file['D_cov'] = D_cov_all
    file['KS'] = ks_all
    file['sigma'] = sigma_vals
    file.close()
    
    return D_cov_all, ks_all, sigma_vals
	
if __name__ == '__main__': ###Docopt stuff
	arguments = docopt(__doc__)
	
	D_cov_all, ks_all, sigma_vals = run_everything(arguments['--cluster'], arguments['--num_sigma'], arguments['--red_clump'], arguments['--run_number'], arguments['--location'], arguments['--elem'])