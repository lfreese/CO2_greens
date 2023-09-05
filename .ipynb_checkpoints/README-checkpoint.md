# CO2 Green's Function work

## Data

All CMIP6 data is obtained from the CMIP6 archive at ESGF, and can be freely downloaded. It is located locally at /fs11/d0/emfreese/CO2_GF/cmip6_data

## Notebooks

### Green's Functions:

CO2_GF_creation_cdr_all_models creates the Green's functions for cdr runs
CO2_GF_creation_pulse_all_models creates the Green's functions for pulse runs
CO2_GF_plots plots the Green's Functions
CO2_*_climatology is to create the climatology uncertainty testing Green's functions 
CO2_*_polyfit is to create the polyfit uncertainty testing Green's functions

### 1pct and 1pct-1000PgC evaluation

1pct_emis_profile diagnoses the emissions for the 1pct run based on CO2mass, nbp, fgco2
1pct_1000gtc_emis_profile diagnoses the emissions for the 1pct run based on CO2mass, nbp, fgco2

1pct_and_1000gtc_convolution is for convolving the GF with the 1pct and 1pct-1000PgC runs emissions

1pct_evaluation is for evaluating how well the GF convolution does at recreating the 1pct case
1pct_and_1000gtc_evaluation is for plotting the temperature change and how well the convolution does for both 1pct and 1000gtc case
1pct_and_1000gtc_uncertainty is for significance testing and SNR testing for the 1pct and 1000gtc case vs. convolution

### Assessment

scenarios is for evaluating two trajectories that end with the same cumulative emissions and location dependence of 2 degrees C across pathways

### Additional figures/eval

Edwards_IAM_convolutions is for testing Morgan Edwards CO2 scenarios
example_GF is for a simple video for slides on how a convolution works
example_trajectories is for a GF explainer slide on how they work with an example
Fourier_transform is for the fourier transform analysis in the supplement

