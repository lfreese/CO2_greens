# CO2 Green's Function work

## Data

All CMIP6 data is obtained from the CMIP6 archive at ESGF, and can be freely downloaded. It is located locally at /fs11/d0/emfreese/CO2_GF/cmip6_data

## Notebooks
Suggested to run in the order shown below:

### 1. Creating the Green's Functions:

CO2_GF_creation creates the Green's functions 
CO2_GF_sensitivity_climatology_creation creates the climatology based sensitivity Green's Functions
CO2_GF_sensitivity_internal_var_creation creates the internal variance based sensitivity for the Green's Functions
CO2_GF_sensitivity_polyfit_creation creates the polyfit of the Green's Function in two forms for comparison
CO2_GF_plots is plotting of the Green's functions

### 2. RTCRE
RTCRE_calc calculates the RTCRE for the relevant model simulations

### 3. Emission Profiles

Emission_profile_1pct_1000gtc diagnoses the emissions for the 1pct-1000pgc case
Emission_profile_1pct diagnoses the emissions for the 1pct co2 case
Emission_profile_hist_co2_only diagnoses the emissions for the hist co2 case

### 4. Convolutions
Convolution_1pct_and_1000gtc does the convolution for the 1pct and 1pct-1000pgc cases
Convolution_hist_co2_only does the convolution for the hist co2 case

### 6. Paper Figures
Paper_Figure_1_global_mean is the figure 1 for the GRL paper
Paper_Figure_2_spatial_pattern is the figure 2 for the GRL paper
Paper_Figure_3_normalized_dif is the figure 3 for the GRL paper
Paper_Figure_4_scenarios is the figure 4 for the GRL paper

### 7. Supplementary Figures
Supplementary_figs_CMIP6_RMSE_comparison has supplementary Figs S8-S19
Supplementary_figs_Fourier_transform has supplementary Fig 23
Supplementary_figs_GF_smoothing has supplementary Figs S20 and S21
Supplementary_figs_Global_mean_pulse_cdr_comparison has supplementary Figs S3
Supplementary_figs_Model_comparison_std_deviation has supplementary Figs S2 and S7
Supplementary_figs_GF_by_model has supplementary Figs S1 and S4
Supplementary_figs_Variance has supplementary Figs S22

Supplementary Figs S5-S6

### Folders

## toy_examples 
for small examples on how convolutions and Green's functions work, useful for presentations
toy_example_GF is the way the GF works
toy_example_trajectories is an example of convolutions

## Outputs
for all outputs
## figures
for all figures
## exploratory 
for notebooks exploring related questions or new ideas, not used in the paper
1pct_A1_B1_global_mean_concentration_profile is for the comparison of two ZECMIP, but these are in EMICs
4x_concentration_profile for potential comparison of Green's function to the 4x CO2 derived Green's Function
4x_validation to validate that
scenarios_internal_var to do a massive spanning of the internal vs. scenario vs. model variability from trajectories
CO2_GF_internal_var_analysis is to plot the scenarios_internal_var
Edwards_IAM_convolution is an attempt to use some GCAM data to convolve with 
uncertainty_analysis is to do different literature types of uncertainty analysis including the scenarios
uncertainty_data_prep gets the data ready to go into the uncertainty_analysis

## ENROADS_base_T
establishing the base temperature that this could be emulated off of to project from 2020-2100 (otherwise we start in 1850)
## cmip6_data
contains all cmip6 data needed for this work (see paper for table to download yourself)
## additional_data
unecessary for analysis, this is just to test other comparisons, and includes the zecmip, Joos, and Edwards data
