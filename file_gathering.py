"""Usage: file_gathering.py [-h][--cluster=<arg>]

Examples:
	Cluster name: e.g. input --cluster='NGC 2682'
	
-h  Help file
--cluster=<arg>  Cluster name

"""

#Imports
from docopt import docopt
import os
import h5py
import numpy as np
import time
import glob
import shutil

def gather_files(name):
    """Concatenate all generated files during parallel run into their respective files.  
    Append to existing files or write files if they don't exist.
    
    Parameters
    ----------
    name : str
        Name of desired cluster (e.g. 'NGC2682')
    
    Returns
    -------
    None
    """
    
    cluster = str(name)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    elem_dict = {'element': ['C', 'N', 'O', 'NA', 'MG', 'AL', 'SI', 'S', 'K', 'CA', 'TI', 'V', 'MN', 'FE', 'NI']}
    #Path to directory created during run
    #path = '/Users/chloecheng/Personal/run_files/' + cluster + '/' - REMOVE FOR FINAL VERSION
    path = '/geir_data/scr/ccheng/AST425/Personal/run_files/' + cluster + '/'
    #Path to main cluster directory
    #outerpath = '/Users/chloecheng/Personal/' + cluster + '/' - REMOVE FOR FINAL VERSION
    outerpath = '/geir_data/scr/ccheng/AST425/Personal/' + cluster + '/'
    
    #Get all file names and sort
    files = os.listdir(path)
    res_data_files = []
    res_sim_files = []
    dcov_files = []
    ks_files = []
    algorithm_files = []
    for file in files:
        if 'res_data' in file:
            res_data_files.append(file)
        elif 'res_sim' in file:
            res_sim_files.append(file)
        elif 'D_cov' in file:
            dcov_files.append(file)
        elif 'KS' in file:
            ks_files.append(file)
        elif 'cdist' not in file and 'simclusterinfo' not in file and '.DS_Store' not in file: 
            algorithm_files.append(file)

    j = 0    
    res_data_0_files = []
    while j <= len(res_data_files)-10:
        res_data_0_files.append(np.sort(res_data_files)[j])
        j += 10
        
    #******************************************
    #*  GATHER ALGORITHM FILES FROM WHOLE RUN *
    #******************************************
    #Get all summary statistic values from created files
    D_cov = []
    KS = []
    sigma = []
    for i in range(len(algorithm_files)):
        open_file = h5py.File(path + algorithm_files[i], 'r')
        D_cov.append(open_file['D_cov'][()])
        KS.append(open_file['KS'][()])
        sigma.append(open_file['sigma'][()])
        open_file.close()
    D_cov = np.array(D_cov)
    D_cov = D_cov.squeeze()
    KS = np.array(KS)
    KS = KS.squeeze()
    sigma = np.array(sigma)
    sigma = sigma.squeeze()

    #Concatenate Dcov and ks values together
    all_D_cov = np.concatenate((D_cov))
    all_KS = np.concatenate((KS))
    all_sigma = np.concatenate((sigma))

    #Write a file for the run
    full_algorithm_file = h5py.File(outerpath + cluster + '_' + timestr + '.hdf5', 'w')
    full_algorithm_file['D_cov'] = all_D_cov
    full_algorithm_file['KS'] = all_KS
    full_algorithm_file['sigma'] = all_sigma
    full_algorithm_file.close()

    #Remove all created subfiles
    for file in algorithm_files:
        os.remove(path + file)
        
    #*****************************************
    #*  GATHER DATA FITS AND RESIDUALS FILES *
    #*****************************************
    a_param = []
    b_param = []
    c_param = []
    err_200 = []
    residuals = []
    points = []
    res_data_path = [] 
    for i in range(len(elem_dict['element'])):
        res_data_path.append(outerpath + cluster + '_' + elem_dict['element'][i] + '_fit_res_data.hdf5')
    for i in range(len(res_data_path)):
        #If the data file doesn't exist in the main cluster directory, create it
        if not glob.glob(res_data_path[i]):
            open_res_data_file = h5py.File(path + res_data_0_files[i] , 'r')
            a_param.append(open_res_data_file['a_param'][()])
            b_param.append(open_res_data_file['b_param'][()])
            c_param.append(open_res_data_file['c_param'][()])
            err_200.append(open_res_data_file['err_200'][()])
            residuals.append(open_res_data_file['residuals'][()])
            points.append(open_res_data_file['points'][()])
            open_res_data_file.close()
            new_res_data_file = h5py.File(res_data_path[i], 'w')
            new_res_data_file['a_param'] = a_param[i]
            new_res_data_file['b_param'] = b_param[i]
            new_res_data_file['c_param'] = c_param[i]
            new_res_data_file['err_200'] = err_200[i]
            new_res_data_file['residuals'] = residuals[i]
            new_res_data_file['points'] = points[i]
            new_res_data_file.close()
        #If it does exist, pass because this information remains the same for every run
        else: 
            pass     

    #Remove all created subfiles
    for file in res_data_files:
        os.remove(path + file)
        
    #**********************************
    #*  GATHER DELTA COVARIANCE FILES *
    #**********************************
    dcov_keys = []
    for i in range(len(dcov_files)):
        open_dcov = h5py.File(path + dcov_files[i], 'r')
        dcov_keys.append(list(open_dcov.keys()))
        for j in dcov_keys[i]:
            #If the data file exists in the main directory, append to it
            if glob.glob(outerpath + cluster + '_' + 'D_cov.hdf5'):
                open_exist_dcov = h5py.File(outerpath + cluster + '_' + 'D_cov.hdf5', 'a')
                #If the group for the particular generated value of sigma doesn't exist, copy the data to the file
                if not j in list(open_exist_dcov.keys()):
                    open_dcov.copy(j, open_exist_dcov)
                #If it does, pass
                else:
                    pass
                open_exist_dcov.close()
            #If the file doesn't exist, write one
            else:
                open_exist_dcov = h5py.File(outerpath + cluster + '_' + 'D_cov.hdf5', 'w')
                open_dcov.copy(j, open_exist_dcov)
                open_exist_dcov.close()
        open_dcov.close()

    #Remove all created subfiles
    for file in dcov_files:
        os.remove(path + file)
        
    #*****************************
    #*  GATHER KS DISTANCE FILES *
    #*****************************
    ks_keys = []
    for i in range(len(ks_files)):
        open_ks = h5py.File(path + ks_files[i], 'r')
        ks_keys.append(list(open_ks.keys()))
        for j in ks_keys[i]:
            #If the data file exists in the main directory, append to it
            if glob.glob(outerpath + cluster + '_' + 'KS.hdf5'):
                open_exist_ks = h5py.File(outerpath + cluster + '_' + 'KS.hdf5', 'a')
                #If the group for the particular generated value of sigma doesn't exist, copy the data to the file
                if not j in list(open_exist_ks.keys()):
                    open_ks.copy(j, open_exist_ks)
                #If it does, pass
                else:
                    pass
                open_exist_ks.close()
            #If the file doesn't exist, write one
            else:
                open_exist_ks = h5py.File(outerpath + cluster + '_' + 'KS.hdf5', 'w')
                open_ks.copy(j, open_exist_ks)
                open_exist_ks.close()
        open_ks.close()

    #Remove all created subfiles
    for file in ks_files:
        os.remove(path + file)
        
    #***********************************************
    #*  GATHER SIMULATION FITS AND RESIDUALS FILES *
    #***********************************************
    #Gather filenames (complicated because there are three copies of each with different PIDs)
    split_res_sim_files = []
    for i in res_sim_files:
        split_res_sim_files.append(i.split('_'))

    res_sim_elem_files = [[] for i in range(len(split_res_sim_files))]
    for i in range(len(split_res_sim_files)):
        for j in split_res_sim_files[i]:
            if j != name and j != 'fit' and j != 'res' and j != 'sim':
                res_sim_elem_files[i].append(j)

    res_sim_inds = []
    for i in range(len(elem_dict['element'])):
        res_sim_inds.append({elem_dict['element'][i]: []})

    for i in range(len(elem_dict['element'])):
        for j in range(len(res_sim_elem_files)):
            if elem_dict['element'][i] in res_sim_elem_files[j]:
                res_sim_inds[i][elem_dict['element'][i]].append(j)

    #Get all of the keys for the simulations in each file
    res_sim_keys = [[] for i in range(len(elem_dict['element']))]
    for i in range(len(res_sim_inds)):
        for j in res_sim_inds[i][elem_dict['element'][i]]:
            open_res_sim = h5py.File(path + res_sim_files[j], 'r')
            res_sim_keys[i].append(list(open_res_sim.keys()))
            open_res_sim.close()

    res_sim_keys_final = [[] for i in range(len(elem_dict['element']))]
    for i in range(len(res_sim_keys)):
        for j in range(len(res_sim_keys[i])):
            for k in res_sim_keys[i][j]:
                res_sim_keys_final[i].append(k)

    for i in range(len(res_sim_keys_final)):
        #If the data file exists in the main directory, append to it
        if glob.glob(outerpath + cluster + '_' + elem_dict['element'][i] + '_fit_res_sim.hdf5'):
            open_exist_res_sim = h5py.File(outerpath + cluster + '_' + elem_dict['element'][i] + '_fit_res_sim.hdf5', 'a')
            for j in range(len(res_sim_inds[i][elem_dict['element'][i]])):
                open_res_sim = h5py.File(path + res_sim_files[res_sim_inds[i][elem_dict['element'][i]][j]], 'r')
                #If the group for the particular generated value of sigma doesn't exist, copy the data to the file
                if not res_sim_keys_final[i][j] in list(open_exist_res_sim.keys()):
                    open_res_sim.copy(res_sim_keys_final[i][j], open_exist_res_sim)
                #If it does, pass
                else:
                    pass
                open_res_sim.close()
            open_exist_res_sim.close()
        else:
            #If the data file does not exist, write one
            open_exist_res_sim = h5py.File(outerpath + cluster + '_' + elem_dict['element'][i] + '_fit_res_sim.hdf5', 'w')
            for j in range(len(res_sim_inds[i][elem_dict['element'][i]])):
                open_res_sim = h5py.File(path + res_sim_files[res_sim_inds[i][elem_dict['element'][i]][j]], 'r')
                open_res_sim.copy(res_sim_keys_final[i][j], open_exist_res_sim)
                open_res_sim.close()
            open_exist_res_sim.close()

    #Remove all created subfiles
    for file in res_sim_files:
        os.remove(path + file)
        
    #Delete the directory
    #shutil.rmtree('/Users/chloecheng/Personal/run_files/') - REMOVE IN FINAL VERSION 
    shutil.rmtree('/geir_data/scr/ccheng/AST425/Personal/run_files/')

if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	files = gather_files(arguments['--cluster'])