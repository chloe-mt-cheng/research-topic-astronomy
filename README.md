# Testing the chemical homogeneity of open clusters: 
## Code from AST425: Research Topic in Astronomy course at University of Toronto, 2019-2020

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
If you're running a cluster for the first time, create the directories `run_files/<cluster_name>` and run the 
run_code.py script first to create the HDF5 file containing the allStar information and the spectra.  To do this, run 
the following command from terminal:

`python3 run_code.py --cluster='<cluster_name>' --num_sigma='1' --red_clump=True --run_number='1'`

where you can enter True or False depending on whether you would like to remove the red clump stars or not.  Then, 
proceed with the following instructions.  If you have run the cluster previously, you can skip this step and proceed
to the following instructions directly.

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
`clustername_date_time.hdf5`.  You can feed the quantities in this file into the `ABC()` function in the limits.py
script to obtain the upper limits on the intrinsic abundance scatter for that particular cluster.

## Description of Scripts
OCCAM Scripts:
1. run_code.py: This script contains a function that runs the entire algorithm for the desired number of sigma.  It 
calls on the functions in all of the other project scripts, directly or indirectly.  It reads in the allStar information
and the spectra, creates synthetic spectra for each value of sigma chosen, and fits these spectra for each element in 
APOGEE.  It then fits the observed spectra for each element in APOGEE.  There will be fitting information for the 
observed spectra for each value of sigma as well, due to the different masking that occurs for each value of sigma as a
result of the repeats.  Then, it computes the summary statistics for each simulation and each element, masks the 
summary statistics wherever an element may be missing due to lack of pixels, and returns these final summary 
statistics.
2. occam_clusters_input.py: This script contains 6 functions:
	* `photometric_Teff`: Computes the photometric effective temperatures for the cluster.
	* get_spectra: Reads in the allStar data and spectra from APOGEE.  Masks the spectra according to the 
	`APOGEE_PIXMASK` bitmask (Holtzman et al. 2015).  Computes the photometric effective temperatures using 
	`photometric_Teff`.  Corrects the spectra for small and large uncertainties.  Removes empty spectra.  Removes red 
	clump stars (if applicable). 
	* `weight_lsq`: Returns the quadratic fit parameters for a data set using the weighted least-squares method from 
	Hogg 2015. 
	* `residuals`: Returns the fit residuals.
	* `make_directory`: Creates a new directory for a cluster, if it does not already exist.
	* `fit_func`: The fitting algorithm.  Gets the indices of the pixels for the element being fit.  The DR14 windows
	are used as they are for every element except for C, N, and Fe, where the DR14 windows are thresholded at 70% of 
	DR12 windows to reduce the number of pixels being used and therefore the run time.  Uses `weight_lsq` to fit and 
	`residuals` to find the fit residuals.  Also returns a version with NaNs from masking removed, for the 
	cumulative distributions later.  Note that if an element has less than 5 points for every pixel, it is skipped.
3. occam_clusters_post_process.py: The only relevant functions in this script for the most part is are `res_cat` and
`cum_dist`.  `res_cat` concatenates the fit residuals and errors.  `cum_dist` computes the cumulative distribution of
the normalized fit residuals.
4. ABC.py: This script contains 8 functions:
	* `get_cluster_data`: Calls on `get_spectra` from occam_clusters_input.py to read in the allStar data and corrected
	spectra.
	* `psm_data`: Creates synthetic spectra using psm.py (Rix+ 2017), pads it with zeroes to match DR14, masks it using
	the `APOGEE_PIXMASK` bitmask (Holtzman et al. 2015), and adds fake noise to the fake spectra by multiplying the 
	spectral errors by a randomly drawn repeat from Bovy 2016.  Fits the synthetic spectra and creates cumulative
	distributions of the normalized residuals.
	* `real_data`: Fits the observed spectra.
	* `cov_matrix`: Returns the covariance matrix of the normalized fit residuals.
	* `d_cov`: Returns the covariance matrix summary statistic, as computed in Bovy 2016.
	* `d_cov_all`: Returns the covariance matrix summary statistic for all APOGEE elements in the desired cluster.
	* `KS`: Returns the KS distance summary statistic.
	* `KS_all`: Returns the KS distance for all APOGEE elements in the desired cluster.

PJ Scripts:
1. pj_run_code.py: Does the same thing as run_code.py.
2. pj_clusters_input.py: Similar to occam_clusters_input.py, the only difference is how the allStar data and spectra
are read in.
3. pj_ABC.py: Essentially the same as ABC.py

Other Scripts: 
1. psm.py: The PSM code from Yuan-Sen Ting (Rix+ 2017).
2. file_gathering.py: Gathers up all files into major files after running multiple simulations.
3. homogeneity_algorithm.sh: Shell script to run multiple simulations in the background.
4. pj_homogeneity_algorithm.sh: Similar to homogeneity_algorithm.sh, but for the blindly chemically-tagged clusters.
5. limits.py: Automated calculation of the upper limits on the intrinsic abundance scatter for each element in a 
cluster.

Other Files:
1. dr14_windows.hdf5: The DR14 windows read in from SDSS-IV.
2. kurucz_quadratic_psm.npz: Required for psm.py.
3. occam_cluster-DR14.fits: Required for occam_clusters_input.py.
4. occam_member-DR14.fits: Required for occam_clusters_input.py.
5. published_clusters.npy: Catalogue of chemically tagged clusters from Price-Jones et al. 2020.