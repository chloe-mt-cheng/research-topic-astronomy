# research-topic-astronomy

Code from AST425: Research Topic in Astronomy course at University of Toronto, 2019-2020

Project: Testing the chemical homogeneity of open clusters

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