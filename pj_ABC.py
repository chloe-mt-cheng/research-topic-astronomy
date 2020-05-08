"""Usage: pj_ABC.py [-h][--cluster=<arg>][--sigma=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
	Sigma choice: e.g. input --sigma=0.1

-h  Help file
--cluster=<arg>  Cluster name
--sigma=<arg> Sigma value choice

"""

#Import project scripts
import pj_clusters_input as pj
import occam_clusters_post_process as pp
#Import python modules
import numpy as np
from docopt import docopt
from scipy.interpolate import interp1d
import h5py
import glob
#Import apogee
from apogee.tools import toApStarGrid
from apogee.tools import toAspcapGrid
#Import psm
import psm

def get_cluster_data(cluster):
    """Return the APOGEE data for the desired cluster.
    
    Parameters
    ----------
    cluster : str
    	Name of the desired cluster (e.g. 'PJ_26)
    
    Returns
    -------
    apogee_cluster_data : structured array
		All cluster data from APOGEE
	spectra : tuple
		Array of floats representing the cleaned-up fluxes in the APOGEE spectra
	spectra_errs : tuple
		Array of floats representing the cleaned-up spectral errors from the APOGEE spectra
	T : tuple
		Array of floats representing the effective temperatures of the stars in the cluster
		between 4000K and 5000K
	num_elem : int
		The number of elements in APOGEE (15)
	num_stars : int
		The number of stars in the desired cluster
	bitmask : tuple
		Array of ints (1 or 0), cleaned in the same way as the spectra, representing the bad pixels 
		in the APOGEE_PIXMASK bitmask
    """
    apogee_cluster_data, spectra, spectra_errs, T, bitmask = pj.get_spectra(cluster)
    num_elem = 15
    num_stars = len(spectra)
    return apogee_cluster_data, spectra, spectra_errs, T, num_elem, num_stars, bitmask

def real_data(num_elem, cluster, spectra, spectra_errs, T):
    """Return the fit residuals (with and without NaNs), errors (with and without NaNs), element weights, 
    cumulative distributions, used and skipped elements, and number of elements for every element in the
    desired cluster for the data.
    
    Parameters
    ----------
    num_elem : int
    	The number of elements in APOGEE
    cluster : str
    	The name of the desired cluster (e.g. 'PJ_26')
    spectra : tuple
    	Array of floats representing the spectra of the desired cluster
    spectra_errs : tuple
    	Array of floats representing the spectral uncertainties of the desired cluster
    T : tuple
    	Array of floats representing the effective temperature of each star in the cluster
    	
    Returns
    -------
    real_res : tuple
    	Array of floats representing the residuals of the quadratic fit 
    real_err : tuple
    	Array of floats representing the spectral errors corresponding to the residuals
    real_nanless_res : tuple
    	Array of floats representing the residuals of the quadratic fit, with NaNs removed
    real_nanless_err : tuple
    	Array of floats representing the spectral errors corresponding to the residuals, with NaNs removed
    real_weights : tuple
    	Array of floats representing the weight of each elemental window
    y_ax_real : tuple
		One-dimensional array containing values from 0 to 1, the same size as cdist
	real_cdists : tuple
		One-dimensional array containing the sorted, normalized fit residuals
	real_used_elems : tuple (may remove, don't think I use this)
		Array of str representing the elements used in the cluster (some elements may be omitted due to 
		lack of data)
	real_skipped_elems : tuple (may remove, don't think I use this)
    	Array of str representing the elements skipped due to lack of data 
    new_num_elem : int
    	The number of elements used, taking into account elements omitted due to lack of data
    """
    
    #Fit function for every element (real)
    elem_dict = {'element': ['C', 'N', 'O', 'NA', 'MG', 'AL', 'SI', 'S', 'K', 'CA', 'TI', 'V', 'MN', 'FE', 'NI']}
    real_res = []
    real_err = []
    real_points = []
    real_temp = []
    real_a = []
    real_b = []
    real_c = []
    real_nanless_res = []
    real_nanless_err = []
    real_nanless_T = []
    real_nanless_points = []
    real_weights = []
    real_skipped_elems = []
    real_used_elems = []
    for i in range(len(elem_dict['element'])):
        real_dat = pj.fit_func(elem_dict['element'][i], cluster, spectra, spectra_errs, T, dat_type = 'data', sigma_val=None)
        if real_dat is None:
            real_skipped_elems.append(elem_dict['element'][i])
            continue
        else:
            real_used_elems.append(elem_dict['element'][i])
            real_res.append(real_dat[0])
            real_err.append(real_dat[1])
            real_points.append(real_dat[2])
            real_temp.append(real_dat[3])
            real_a.append(real_dat[4])
            real_b.append(real_dat[5])
            real_c.append(real_dat[6])
            real_nanless_res.append(real_dat[7])
            real_nanless_err.append(real_dat[8])
            real_nanless_T.append(real_dat[9])
            real_nanless_points.append(real_dat[10])
            real_weights.append(real_dat[11])

    real_res = np.array(real_res)
    real_err = np.array(real_err)
    real_points = np.array(real_points)
    real_temp = np.array(real_temp)
    real_a = np.array(real_a)
    real_b = np.array(real_b)
    real_c = np.array(real_c)
    real_nanless_res = np.array(real_nanless_res)
    real_nanless_err = np.array(real_nanless_err)
    real_nanless_T = np.array(real_nanless_T)
    real_nanless_points = np.array(real_nanless_points)
    real_weights = np.array(real_weights)

    #Cumulative distributions
    y_ax_real = []
    real_cdists = []
    cdist_dir = pp.make_directory_cdist(cluster)
    new_num_elem = len(real_used_elems)
    for i in range(new_num_elem):
        cum_dists = pp.cum_dist(cluster, real_used_elems[i], 'data', real_nanless_res[i], real_nanless_err[i])
        y_ax_real.append(cum_dists[0])
        real_cdists.append(cum_dists[1])
    for i in range(new_num_elem):
        y_ax_real[i] = np.array(y_ax_real[i])
        real_cdists[i] = np.array(real_cdists[i])
    y_ax_real = np.array(y_ax_real)
    real_cdists = np.array(real_cdists)
    return real_res, real_err, real_nanless_res, real_nanless_err, real_weights, y_ax_real, real_cdists, real_used_elems, real_skipped_elems, new_num_elem

def psm_data(num_elem, num_stars, apogee_cluster_data, sigma, T, cluster, spectra, spectra_errs):
    """Return the residuals (with and without NaNs), errors (with and without NaNs), cumulative distributions,
    and skipped elements for the simulated spectra.
     
    This function generates synthetic spectra using PSM and a specified sigma value, then fits the simulated
    spectra in the same way as the data.
    
    Parameters
    ----------
    num_elem : int
    	The number of elements in APOGEE
    num_stars : int
    	The number of stars in the desired cluster
    apogee_cluster_data : structured array
    	All cluster data from APOGEE
    sigma : float
    	The value of sigma to create the simulated spectra
    T : tuple
    	Array of floats representing the effective temperature of each star in the cluster
    cluster : str
    	Name of the desired cluster (e.g. 'PJ_26')
    pectra : tuple
    	Array of floats representing the spectra of the desired cluster
    spectra_errs : tuple
    	Array of floats representing the spectral uncertainties of the desired cluster
    	
    Returns
    -------
    fake_res : tuple
    	Array of floats representing the residuals of the quadratic fit 
	fake_err : tuple
    	Array of floats representing the spectral errors corresponding to the residuals
    y_ax_psm : tuple
		One-dimensional array containing values from 0 to 1, the same size as cdist
	psm_cdists : tuple
		One-dimensional array containing the sorted, normalized fit residuals
    fake_nanless_res : tuple
    	Array of floats representing the residuals of the quadratic fit, with NaNs removed 
    	(doesn't return fake_nanless_err because they are the same as real_nanless_err)
    fake_used_elems : tuple (may remove, don't think I use this)
		Array of str representing the elements used in the cluster (some elements may be omitted due to 
		lack of data)
	fake_skipped_elems : tuple (may remove, don't think I use this)
    	Array of str representing the elements skipped due to lack of data 
    """
    
    #Abundances WRT H
    elem_dict = {'element': ['C', 'N', 'O', 'NA', 'MG', 'AL', 'SI', 'S', 'K', 'CA', 'TI', 'V', 'MN', 'FE', 'NI']}
    fe_abundance_dict = {'element': ['C_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE', 'SI_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'V_FE', 'MN_FE', 'NI_FE', 'FE_H']}
    cluster_xh = np.zeros((num_elem, num_stars))
    for i in range(num_elem):
        for j in range(num_stars):
            cluster_xh[i] = apogee_cluster_data[fe_abundance_dict['element'][i]]*apogee_cluster_data['FE_H']
    cluster_avg_abundance = np.mean(cluster_xh, axis=1)

    #Generate a synthetic spectrum
    elem_number_dict = {'C': 0,
                       'N': 1,
                       'O': 2,
                       'NA': 3,
                       'MG': 4,
                       'AL': 5,
                       'SI': 6,
                       'S': 7,
                       'K': 8,
                       'CA': 9,
                       'TI': 10,
                       'V': 11,
                       'MN': 12,
                       'FE': 13,
                       'NI': 14}
    num_stars = len(spectra)
    cluster_logg = apogee_cluster_data['LOGG']

    #Draw simulated abundances
    cluster_fake_abundance = np.zeros((num_elem, num_stars))
    for i in range(num_elem):
        cluster_fake_abundance[i] = np.random.normal(loc = cluster_avg_abundance[i], scale = float(sigma), size = num_stars)

    #Create synthetic spectra
    cluster_gen_spec = np.zeros((num_stars, 7214))
    for i in range(len(spectra)):
        cluster_gen_spec[i] = psm.generate_spectrum(Teff = T[i]/1000, logg = cluster_logg[i], vturb = psm.vturb, 
                                                   ch = cluster_fake_abundance[elem_number_dict['C']][i], 
                                                   nh = cluster_fake_abundance[elem_number_dict['N']][i], 
                                                   oh = cluster_fake_abundance[elem_number_dict['O']][i],
                                                   nah = cluster_fake_abundance[elem_number_dict['NA']][i], 
                                                   mgh = cluster_fake_abundance[elem_number_dict['MG']][i], 
                                                   alh = cluster_fake_abundance[elem_number_dict['AL']][i], 
                                                   sih = cluster_fake_abundance[elem_number_dict['SI']][i], 
                                                   sh = cluster_fake_abundance[elem_number_dict['S']][i], 
                                                   kh = cluster_fake_abundance[elem_number_dict['K']][i],
                                                   cah = cluster_fake_abundance[elem_number_dict['CA']][i], 
                                                   tih = cluster_fake_abundance[elem_number_dict['TI']][i], 
                                                   vh = cluster_fake_abundance[elem_number_dict['V']][i], 
                                                   mnh = cluster_fake_abundance[elem_number_dict['MN']][i], 
                                                   nih = cluster_fake_abundance[elem_number_dict['NI']][i], 
                                                   feh = cluster_fake_abundance[elem_number_dict['FE']][i], 
                                                   c12c13 = psm.c12c13)

    #Pad psm spectra with zeros to make appropriate size for DR16
    apStar_cluster_gen_spec = toApStarGrid(cluster_gen_spec, dr='12')
    cluster_padded_spec = toAspcapGrid(apStar_cluster_gen_spec, dr='16')

    #Create array of NaNs to mask the psm in the same way as the spectra
    masked_psm = np.empty_like(spectra)
    masked_psm[:] = np.nan

    #Mask the spectra
    for i in range(len(spectra)):
        for j in range(7514):
            if ~np.isnan(spectra[i][j]):
                masked_psm[i][j] = cluster_padded_spec[i][j]

    #Create fake noise to add to the psm 
    cluster_fake_errs = np.zeros_like(spectra_errs)
    for i in range(num_stars):
        for j in range(7514):
            #Maintain zero-padding in errors
            if masked_psm[i][j] == 0.0:
                cluster_fake_errs[i][j] = 0.0
            else:
                cluster_fake_errs[i][j] = np.random.normal(loc = 0.0, scale = spectra_errs[i][j])

    #Add the noise to the psm 
    final_fake_spec = masked_psm + cluster_fake_errs

    #Run fitting function on synthetic spectra
    fake_res = []
    fake_err = []
    fake_points = []
    fake_temp = []
    fake_a = []
    fake_b = []
    fake_c = []
    fake_nanless_res = []
    fake_nanless_err = []
    fake_nanless_T = []
    fake_nanless_points = []
    fake_skipped_elems = []
    fake_used_elems = []
    for i in range(len(elem_dict['element'])):
        fake_dat = pj.fit_func(elem_dict['element'][i], cluster, final_fake_spec, spectra_errs, T, dat_type = 'sim', sigma_val=sigma)
        if fake_dat is None:
            fake_skipped_elems.append(elem_dict['element'][i])
            continue
        else:
            fake_used_elems.append(elem_dict['element'][i])
            fake_res.append(fake_dat[0])
            fake_err.append(fake_dat[1])
            fake_points.append(fake_dat[2])
            fake_temp.append(fake_dat[3])
            fake_a.append(fake_dat[4])
            fake_b.append(fake_dat[5])
            fake_c.append(fake_dat[6])
            fake_nanless_res.append(fake_dat[7])
            fake_nanless_err.append(fake_dat[8])
            fake_nanless_T.append(fake_dat[9])
            fake_nanless_points.append(fake_dat[10])

    fake_res = np.array(fake_res)
    fake_err = np.array(fake_err)
    fake_points = np.array(fake_points)
    fake_temp = np.array(fake_temp)
    fake_a = np.array(fake_a)
    fake_b = np.array(fake_b)
    fake_c = np.array(fake_c)
    fake_nanless_res = np.array(fake_nanless_res)
    fake_nanless_err = np.array(fake_nanless_err)
    fake_nanless_T = np.array(fake_nanless_T)
    fake_nanless_points = np.array(fake_nanless_points)

    #Cumulative distributions
    y_ax_psm = []
    psm_cdists = []
    cdist_dir = pp.make_directory_cdist(cluster)
    new_num_elem = len(fake_used_elems)
    for i in range(new_num_elem):
        cum_dists = pp.cum_dist(cluster, fake_used_elems[i], 'sim', fake_nanless_res[i], fake_nanless_err[i])
        y_ax_psm.append(cum_dists[0])
        psm_cdists.append(cum_dists[1])
    for i in range(new_num_elem):
        y_ax_psm[i] = np.array(y_ax_psm[i])
        psm_cdists[i] = np.array(psm_cdists[i])
    y_ax_psm = np.array(y_ax_psm)
    psm_cdists = np.array(psm_cdists)
    return fake_res, fake_err, y_ax_psm, psm_cdists, fake_nanless_res, fake_used_elems, fake_skipped_elems

def cov_matrix(res, err, num_stars):
    """Return the covariance matrix of the normalized fit residuals.
	
	Parameters
	----------
	res : tuple
		Array of floats representing the residuals of the quadratic fits
	err : tuple
		Array of floats representing the spectral errors corresponding to the residuals
	num_stars : int
		The number of stars in the cluster
	
	Returns 
	-------
	covariance_matrix : tuple
		The covariance matrix of the normalized residuals
	"""
    normalized_res = res/err
    covariance_matrix = np.zeros((len(normalized_res), len(normalized_res)))
    
    #Calculate means and sums
    pixel_means = np.nanmean(normalized_res, axis=1)
    tiled_means = np.tile(pixel_means, (num_stars, 1)).T
    diffs = normalized_res - tiled_means
    row_sums = np.sum(~np.isnan(normalized_res), axis=1)
    
    #Create covariance matrix
    for pixel in range(len(normalized_res)):
        rowdiff = diffs[pixel]
        tiled_row = np.tile(rowdiff, (len(normalized_res), 1))
        covariance_matrix[pixel] = np.nansum(diffs*tiled_row, axis=1)/(row_sums[pixel]-1)
    return covariance_matrix

def d_cov(weights, data_res, data_err, simulated_res, simulated_err, num_stars):
    """Return the covariance matrix summary statistic, as computed in Bovy 2016.
	
	Parameters
	----------
	weights : tuple
		Array of floats representing the weights of each pixel in the element
	data_res : tuple
		Array of floats representing the residuals of the quadratic fits of the data
	data_err : tuple
		Array of floats representing the spectral errors corresponding to the residuals of the data
	simulated_res : tuple
		Array of floats representing the residuals of the quadratic fits of the simulation
	simulated_err : tuple
		Array of floats representing the spectral errors corresponding to the residuals of the simulation
	num_stars : int
		The number of stars in the cluster
	
	Returns
	-------
	D_cov : float
		The covariance matrix summary statistic
	"""
	
    #Compute the covariance matrices
    data_cov = cov_matrix(data_res, data_err, num_stars)
    sim_cov = cov_matrix(simulated_res, simulated_err, num_stars)
    
    #Compute the summary statistic
    stat = np.zeros_like(data_cov)
    for i in range(len(data_cov)):
        for j in range(len(data_cov)):
        	stat[i][j] = np.sqrt(weights[i]*weights[j])*((data_cov[i][j] - sim_cov[i][j])**2)
    D_cov = np.sqrt(np.sum(stat))
    return D_cov
    
def d_cov_all(cluster, new_num_elem, weights, real_res, real_err, fake_res, fake_err, num_stars, sigma):
	"""Return the covariance matrix summary statistic for all APOGEE elements in the desired cluster.
	
	Parameters
	----------
	cluster : str
		Name of the desired cluster (e.g. 'PJ_26)
	new_num_elem : int
		The number of elements after all bitmasking and large error cutting (some elements may be 
		removed due to lack of data)
	weights : tuple
		Array of floats representing the weights of each pixel in the element
	real_res : tuple
		Array of floats representing the residuals of the quadratic fits of the data for every APOGEE element
	real_err : tuple
		Array of floats representing the spectral errors corresponding to the residuals of the data for every
		APOGEE element
	fake_res : tuple
		Array of floats representing the residuals of the quadratic fits of the simulation for every APOGEE 
		element
	fake_err : tuple
		Array of floats representing the spectral errors corresponding to the residuals of the simulation for 
		every APOGEE element
	num_stars : int
		Number of stars in desired cluster
	sigma : float
		Value of sigma used for the simulation
	
	Returns
	-------
	D_cov_all : tuple
		Array of floats representing the covariance matrix summary statistic for every element in the cluster
	"""
	
	D_cov_all = np.zeros(new_num_elem)
	for i in range(new_num_elem):
		D_cov_all[i] = d_cov(weights[i], real_res[i], real_err[i], fake_res[i], fake_err[i], num_stars)
	
	#Save data to file
	#path = '/Users/chloecheng/Personal/' + str(cluster) + '/' + str(cluster) + '_' + 'D_cov' + '.hdf5' #Personal path
	path = '/geir_data/scr/ccheng/AST425/Personal/' + str(cluster) + '/' + str(cluster) + '_' + 'D_cov' + '.hdf5' #Server path
	#If file exists, append to file
	if glob.glob(path):
		file = h5py.File(path, 'a')
		grp = file.create_group(str(sigma))
		grp['D_cov'] = D_cov_all
		file.close()
	#If the file does not exist, create a new file
	else:
		file = h5py.File(path, 'w')
		grp = file.create_group(str(sigma))
		grp['D_cov'] = D_cov_all
		file.close()
	return D_cov_all
    
def KS(data_yax, data_cdist, sim_yax, sim_cdist):
    """Return the KS distance summary statistic.
    
    Parameters
    ----------
    data_yax : tuple
		One-dimensional array containing values from 0 to 1, the same size as cdist, for the data
	data_cdist : tuple
		One-dimensional array containing the sorted, normalized fit residuals for the data
	sim_yax : tuple
		One-dimensional array containing values from 0 to 1, the same size as cdist, for the simulation
	sim_cdist : tuple
		One-dimensional array containing the sorted, normalized fit residuals for the simulation
    """
    
    #Interpolate the cumulative distributions for subtraction
    real_interp = interp1d(data_cdist, data_yax)
    fake_interp = interp1d(sim_cdist, sim_yax)
    lower_bound = np.max((np.min(data_cdist), np.min(sim_cdist)))
    upper_bound = np.min((np.max(data_cdist), np.max(sim_cdist)))
    xnew = np.linspace(lower_bound, upper_bound, 1000)
    
    #Compute the KS distance
    dist = np.max(np.abs(real_interp(xnew) - fake_interp(xnew)))
    return dist

def KS_all(cluster, new_num_elem, y_ax_real, real_cdists, y_ax_psm, psm_cdists, sigma):
    """Return the KS distance for all APOGEE elements in the desired cluster.
    
    Parameters
    ----------
    cluster : str
    	Name of the desired cluster (e.g. 'PJ_26')
    new_num_elem : int
    	The number of elements in APOGEE, taking into account omitted elements 
    y_ax_real : tuple
    	One-dimensional array containing values from 0 to 1, the same size as cdist, for each element in the data
    real_cdists : tuple
    	One-dimensional array containing the sorted, normalized fit residuals for each element in the data
     y_ax_psm : tuple
    	One-dimensional array containing values from 0 to 1, the same size as cdist, for each element in the 
    	simulation
    psm_cdists : tuple
    	One-dimensional array containing the sorted, normalized fit residuals for each element in the simulation
    sigma : float
    	The value of sigma used to create the simulation
    """
    
    ks_all = np.zeros(new_num_elem)
    for i in range(new_num_elem):
    	ks_all[i] = KS(y_ax_real[i], real_cdists[i], y_ax_psm[i], psm_cdists[i])
    	
    #Save data to file
    #path = '/Users/chloecheng/Personal/' + str(cluster) + '/' + str(cluster) + '_' + 'KS' + '.hdf5' #Personal path
    path = '/geir_data/scr/ccheng/AST425/Personal/' + str(cluster) + '/' + str(cluster) + '_' + 'KS' + '.hdf5' #Server path
    #If file exists, append to file
    if glob.glob(path):
    	file = h5py.File(path, 'a')
    	grp = file.create_group(str(sigma))
    	grp['KS'] = ks_all
    	file.close()
    #If the file does not exist, create a new file
    else:
    	file = h5py.File(path, 'w')
    	grp = file.create_group(str(sigma))
    	grp['KS'] = ks_all
    	file.close()
    return ks_all

if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	apogee_cluster_data, spectra, spectra_errs, T, num_elem, num_stars, bitmask = get_cluster_data(arguments['--cluster'])
	real_res, real_err, real_nanless_res, real_nanless_err, real_weights, y_ax_real, real_cdists, real_used_elems, real_skipped_elems, new_num_elem = real_data(num_elem, arguments['--cluster'], spectra, spectra_errs, T)
	fake_res, fake_err, y_ax_psm, psm_cdists, fake_nanless_res, fake_used_elems, fake_skipped_elems = psm_data(num_elem, num_stars, apogee_cluster_data, arguments['--sigma'], T, arguments['--cluster'], spectra, spectra_errs)
	D_cov_all = d_cov_all(arguments['--cluster'], new_num_elem, real_weights, real_res, real_err, fake_res, fake_err, len(spectra), arguments['--sigma'])
	ks_all = KS_all(arguments['--cluster'], new_num_elem, y_ax_real, real_cdists, y_ax_psm, psm_cdists, arguments['--sigma'])