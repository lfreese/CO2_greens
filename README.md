# CO2 Green's Function work

## Data

All CMIP6 data is obtained from the CMIP6 archive at ESGF, and can be freely downloaded. It is located locally at /fs11/d0/emfreese/CO2_GF/cmip6_data

## Notebooks
Notebooks are ordered 1-7, with the main notebook $(\#_xxx)$ being for the main analysis, and if there are subcomponents $(\#.\#_xxx)$ those would be sensitivities for the work.

### 1. Creating the Green's Functions:

1_CO2_GF_creation creates the Green's functions 
1.1_CO2_GF_sensitivity_climatology_creation creates the climatology based sensitivity Green's Functions
1.2_CO2_GF_sensitivity_internal_var_creation creates the internal variance based sensitivity for the Green's Functions
1.3_CO2_GF_sensitivity_polyfit_creation creates the polyfit of the Green's Function in two forms for comparison
1.4_CO2_GF_plots is plotting of the Green's functions

### 2. RTCRE
2_RTCRE_calc calculates the RTCRE for the relevant model simulations

### 3. Emission Profiles

3_emission_profile_1pct_1000gtc diagnoses the emissions for the 1pct-1000pgc case
3_emission_profile_1pct diagnoses the emissions for the 1pct co2 case
3_emission_profile_hist_co2_only diagnoses the emissions for the hist co2 case

### 4. Convolutions
4_convolution_1pct_and_1000gtc does the convolution for the 1pct and 1pct-1000pgc cases
4_convolution_hist_co2_only does the convolution for the hist co2 case

### 5. Evaluation
5_evaluation is the main evaluation of the green's function compared to the CMIP6 model runs
5.1_RTCRE_comparison_hist compares how a RTCRE does vs. the Green's function at recreating the hist CO2 run

### 6. Scenarios
6_scenarios_trajectories is in the main text, and is the two trajectories for the same cumulative emissions
6.1_scenarios_2degrees is an extra analysis looking at the temp in local places when 2 degrees global is hit

### 7. Uncertainty
7.1_uncertainty_CO2_GF_internal_var tests the role of internal variability vs model variability (in a Hawkins and Sutton 2009 approach)
7.2_uncertainty_CO2_GF_climatology_polyfit tests the impact different polyfits have on the Green's Function, as well as the use of climatological mean for the pictrl versus not
7.3_uncertainty_fourier_transorm is the fourier transform of emissions, Green's function and their product to look at the role of internal variability

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
