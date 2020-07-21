# research-topic-astronomy

Code from AST425: Research Topic in Astronomy course at University of Toronto, 2019-2020

# Project: Testing the chemical homogeneity of open clusters

This project consists of testing the chemical homogeneity of open clusters, helping to paint a clearer picture of the 
early Galactic disc.  The goal of this work is to constrain the initial abundance scatter in open clusters, by 
determining tight upper limits on their intrinsic abundance scatters using stellar spectra from APOGEE.  
To do this, we model spectra as a one-dimensional function of initial stellar mass, perform forward modelling on the 
observed stellar spectra, and compare the simulations and the data using Approximate Bayesian Computation to derive 
these upper limits.

There are two categories of clusters that we study: clusters from OCCAM (Donor et al. 2018) and blindly
chemically-tagged open clusters (Price-Jones et al. 2020).  There are a set of scripts for each category, which largely
serve the same purpose for each set of clusters.  Common between the two sets are the scripts 
occam_clusters_post_process.py, file_gathering.py, and limits.py.  Here, the procedure of running the code and brief 
description of each script will be given in terms of the OCCAM scripts, but the functions are virtually the same for 
the PJ scripts.

## Quick Start
To run several simulations at once, run the homogeneity_algorithm.sh shell script.  To do this, run the following
commands in terminal:

```
chmod 755 homogeneity_algorithm.sh
./homogeneity_algorithm.sh
```

Three prompts will appear and you can fill them with the name of the cluster that is being examined, the number of 
simulations to run, and whether or not red clump stars should be removed.  For example:
> **Enter a cluster**:
> NGC2682
> **Enter the number of simulations**:
> 200
> **Enter True if you want to remove red clump stars, False if not:**
> True

In this example, the script will proceed to run 200 simulations 10 times, for a total of 2000 simulations.  This script
runs the simulations in the background so that many simulations can be run in a shorter amount of time.  

When the simulations begin to run, the directories `run_files/<cluster_name>` will be created and each of the required 
scripts will be copied over to the `run_files` directory.  To check the process of the simulations, I usually check 
how many files there are in `run_files/<cluster_name>`, using the command `ls -1 | wc -l` in terminal.  This is 
because several different files are created for each of the simulations.  Once this number stops changing, the process
is complete.

Once the process is complete, run file_gathering.py in the directory where it is located as such:
`python3 file_gathering.py --cluster='<cluster_name>'`
This will write/append all of the individual files to a master file of each type, i.e. a file for all of the fitting
information for the data, a file for all of the fitting information for the simulations, a file for all of the Dcov
calculations, a file for all of the KS calculations, and a total file for the run, which contains the Dcov, KS, and 
sigma values for the run.  

This last file is the crucial one for running the ABC algorithm later and has the naming convention 
`cluster_name_date_time.hdf5`.  You can feed the quantities in this file into the `ABC()` function in the limits.py
script to obtain the upper limits on the intrinsic abundance scatter for that particular cluster.

##  




OCCAM Scripts:
1. occam_clusters_input.py: Gets APOGEE data and defines fit function
2. occam_clusters_post_process.py: Computes cumulative distribution of normalized residuals
3. ABC.py: Fits real data, generates synthetic spectra and fits them, computes summary statistics
4. run_code.py: Runs the entire algorithm for desired number of sigma

PJ Scripts: 
1. pj_clusters_input.py: Similar to occam_clusters_input.py, only difference is how we read in the clusters
2. pj_ABC.py: Similar to ABC.py, no major differences
3. pj_run_code.py: Similar to run_code.py, no major differences

Other Scripts:
1. scale_err.py: Similar to run_code.py but draws an error scaling factor for every simulation
2. psm.py: The PSM code from Yuan-Sen Ting (Rix+ 2017)

Other Files:
1. dr14_windows.hdf5: The DR14 windows read in from SDSS-IV
2. kurucz_quadratic_psm.npz: Required for psm.py
3. occam_cluster-DR14.fits: Required for occam_clusters_input.py
4. occam_member-DR14.fits: Required for occam_clusters_input.py
5. published_clusters.npy: Catalogue of chemically tagged clusters from Price-Jones et al. 2020