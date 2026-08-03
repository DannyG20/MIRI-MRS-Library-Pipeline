# MIRI-MRS-Library-Pipeline
Pipeline and reference files resulting from PID3779. Please take a look at the following papers regarding the methods and limitations: https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.226G/abstract (Paper I, BFE correction), https://ui.adsabs.harvard.edu/abs/2025A%26A...697A..58G/abstract (Paper II, fringe correction), and *Paper III coming soon* (Paper III, optimal extraction and detector-based HCI). Please also cite these works when making use of the relevant parts of the code.

This code requires STScI's JWST pipeline to run:  
https://github.com/spacetelescope/jwst

  
The following alternative reference files must be downloaded:  
https://owncloud.ster.kuleuven.be/index.php/s/bswdjrMZzZaKdHX  
Make sure to place the '/coeff_fits' folder inside '/core', and '/references' in the same folder as the code.

______

# Running stage 1
In order to run stage 1 of the pipeline, both an _uncal_ and associated _rate_ file are required. The _rate_ file can simply be the product of a standard pipeline stage 1 reduction. The user must fill in the lists of directories and files to run, after which the associated reference files are selected automatically.

# Running stage 2
Running stage 2 simply requires an input _rate_ file, which is the product of stage 1. Similarly to the stage 1 setup, the user provides a list of directories and files to run. The code automatically selects the correct reference files to use.

# Running optimal extraction
The optimal extraction code requires directory containing _rate_ files and _cal_ files to run. The user then provides the list of bands that these files are associated to. Note that the _cal_ files must have been reduced using the continuous reference files, which can be done in the second half of the stage 2 notebook.

In order to extract a companion spectrum, a PSF-subtracted _cal_ file along with the aforementioned files must be provided. The directory must be set separately from the other files. If running an extraction on a companion, set `PSFsub` to `True`. An offset in $\alpha$ and $\beta$ (in arcsec) compared to the central star must be provided. `peakfitlist` then indicates how many peaks to fit, provided as a list with the indices corresponding to the different channels (i.e., \[channel1, channel2, ...\]). It is advised to set the aperture correction to `True`, and the aperture size to 1.2.

______
#Diagnostics
<ins>To do: </ins>
- Provide a map of the science data compared to the grid. This will give the user a sense of how well they can expect the corrections to perform, where science data located within the grid will benefit more from the special reference files.
- Allow the option to save the fitted PSFs in the optimal extraction code. These PSFs can be compared to the science data to check whether the fits make sense.


