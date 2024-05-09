import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cftime
import dask
import xarrayutils
import cartopy.crs as ccrs
from xmip.preprocessing import combined_preprocessing
from xmip.preprocessing import replace_x_y_nominal_lat_lon
from xmip.drift_removal import replace_time
from xmip.postprocessing import concat_experiments
import xmip.drift_removal as xm_dr
import xmip as xm
import xesmf as xe
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import cf_xarray as cfxr


########################### CMIP 6 DATA AND REGRIDDING ###################################

#### initial import and data merging ####
#subroutines to import

def _import_combine_pulse_control(control_path, pulse_path, replace_xy, m):
    ds_control = xr.open_mfdataset(control_path, use_cftime=True)
    ds_pulse = xr.open_mfdataset(pulse_path, use_cftime=True)
    lat_corners = cfxr.bounds_to_vertices(ds_control.isel(time = 0)['lat_bnds'], "bnds", order=None)
    lon_corners = cfxr.bounds_to_vertices(ds_control.isel(time = 0)['lon_bnds'], "bnds", order=None)
    ds_control = ds_control.assign(lon_b=lon_corners, lat_b=lat_corners)

    lat_corners = cfxr.bounds_to_vertices(ds_pulse.isel(time = 0)['lat_bnds'], "bnds", order=None)
    lon_corners = cfxr.bounds_to_vertices(ds_pulse.isel(time = 0)['lon_bnds'], "bnds", order=None)
    ds_pulse = ds_pulse.assign(lon_b=lon_corners, lat_b=lat_corners)

    if ds_control.attrs['parent_source_id'] != ds_pulse.attrs['parent_source_id']:
        print('WARNING: Control and Pulse runs are not from the same parent source!')
    
    #fix the time for two of the models
    if m == 'NORESM2':
        ds_pulse['time'] = ds_pulse['time']+timedelta(365*1)
    if m =='CANESM5_r1p2' or m == 'CANESM5_r2p2' or m == 'CANESM5_r3p2':
        ds_control['time'] = ds_control['time']-timedelta(365*300)
    #select only the times that match up with the pulse
    ds_control = ds_control.sel(time = slice(ds_control['time'].min(), ds_pulse['time'].max()))

    return(ds_control, ds_pulse)


def _regrid_cont_pulse(ds_control, ds_pulse, ds_out):
    regridder = xe.Regridder(ds_control, ds_out, "conservative")
    attrs = ds_control.attrs
    ds_control = regridder(ds_control) 
    ds_control.attrs = attrs
    
    regridder = xe.Regridder(ds_pulse, ds_out, "conservative")
    attrs = ds_pulse.attrs
    ds_pulse = regridder(ds_pulse) 
    ds_pulse.attrs = attrs
    
    return(ds_control, ds_pulse)


def _calc_greens(ds_control, ds_pulse, variable, m, pulse_type, climatology, internal_variability_test, pulse_size = 100):
    if climatology == False and internal_variability_test == False:
        print('normal run')
        G = (ds_pulse[variable] - ds_control[variable])/(pulse_size)
        times = G.time.get_index('time')
        weights = times.shift(-1, 'MS') - times.shift(1, 'MS')
        weights = xr.DataArray(weights, [('time', G['time'].values)]).astype('float')
        G =  (G * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
        #select ten years in for two of the models
        if pulse_type == 'pulse':
            ten_years_in = 10 #in years
            if m == 'ACCESS':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
            if m == 'UKESM1_r1':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        elif pulse_type == 'cdr':
            ten_years_in = 10 #in years
            if m == 'ACCESS':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        G.attrs = ds_pulse.attrs
        return(G)
    
    elif climatology == True:
        print('climatology run')
        G = (ds_pulse[variable].groupby("time.month") - ds_control[variable].groupby('time.month').mean('time'))/(pulse_size)
        times = G.time.get_index('time')
        weights = times.shift(-1, 'MS') - times.shift(1, 'MS')
        weights = xr.DataArray(weights, [('time', G['time'].values)]).astype('float')
        G =  (G * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
        #select ten years in for two of the models
        if pulse_type == 'pulse':
            ten_years_in = 10 #in years
            if m == 'ACCESS':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
            if m == 'UKESM1_r1':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        elif pulse_type == 'cdr':
            ten_years_in = 10 #in years
            if m == 'ACCESS':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        G.attrs = ds_pulse.attrs
        return(G)
    
    elif internal_variability_test == True:
        print('internal variability run')
        G = {}
        for n in np.arange(0,100)[::5]:
            G[n] = (ds_pulse[variable] - ds_control[variable].shift(time = -n))/(pulse_size)
            times = G[n].time.get_index('time')
            weights = times.shift(-1, 'MS') - times.shift(1, 'MS')
            weights = xr.DataArray(weights, [('time', G[n]['time'].values)]).astype('float')
            G[n] =  (G[n] * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
            #select ten years in for two of the models
            if pulse_type == 'pulse':
                ten_years_in = 10 #in years
                if m == 'ACCESS':
                    G[n] = G[n].isel(year = slice(ten_years_in,len(G[n].year)))
                if m == 'UKESM1_r1':
                    G[n] = G[n].isel(year = slice(ten_years_in,len(G[n].year)))
            elif pulse_type == 'cdr':
                ten_years_in = 10 #in years
                if m == 'ACCESS':
                    G[n] = G[n].isel(year = slice(ten_years_in,len(G[n].year)))
            G[n].attrs = ds_pulse.attrs
        G = xr.concat([G[m] for m in G.keys()], pd.Index([m for m in G.keys()], name='pulse_year'))
        return(G)    

    

#full function
def import_regrid_calc(control_path, pulse_path, ds_out, variable, m, pulse_type, pulse_size = 100,  replace_xy = True, regrid = True, anomaly = False, climatology = False, internal_variability_test = False):
    '''Imports the control run and pulse run for a CMIP6 model run, combines them on the date the pulse starts
    Regrids it to the chosen grid size
    Calculates the Green's Function'''
    
    ds_control, ds_pulse = _import_combine_pulse_control(control_path, pulse_path, replace_xy, m)
    if regrid == True:
        ds_control, ds_pulse = _regrid_cont_pulse(ds_control, ds_pulse, ds_out)
    G = _calc_greens(ds_control, ds_pulse, variable, m, pulse_type, climatology, internal_variability_test, pulse_size)
    if anomaly == True:
        return(anom_control, anom_pulse, anom_G)
    else:
        return(ds_control, ds_pulse, G)


#### single regridder ####
def _regrid_ds(ds_in, ds_out):
    regridder = xe.Regridder(ds_in, ds_out,  'conservative', ignore_degenerate = True)
    ds_new = regridder(ds_in) 
    ds_new.attrs = ds_in.attrs
    return(ds_new)


### define our output grid size

ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
        "lon": (["lon"], np.arange(0, 360, 1)),
        "lat_b": (["lat_b"], np.arange(-90.,91.,1.0)),
        "lon_b":(["lon_b"], np.arange(.5, 361.5, 1.0))
    }
)


#### function to find area of a grid cell from lat/lon ####
def find_area(ds, R = 6378.1):
    """ ds is the dataset, i is the number of longitudes to assess, j is the number of latitudes, and R is the radius of the earth in km. 
    Must have the ds['lat'] in descending order (90...-90)
    Returns Area of Grid cell in km"""
    circumference = (2*np.pi)*R
    deg_to_m = (circumference/360) 
    dy = (ds['lat_b'].roll({'lat_b':-1}, roll_coords = False) - ds['lat_b'])[:-1]*deg_to_m

    dx1 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b']))
    
    dx2 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b'].roll({'lat_b':-1}, roll_coords = False)[:-1]))
    
    A = .5*(dx1+dx2)*dy
    
    #### assign new lat and lon coords based on the center of the grid box instead of edges ####
    A = A.assign_coords(lon_b = ds.lon.values,
                    lat_b = ds.lat.values)
    A = A.rename({'lon_b':'lon','lat_b':'lat'})

    A = A.transpose()
    
    return(A)

A = find_area(ds_out)


### polyfit for G ###

def import_polyfit_G(G_ds_path, G_cdr_ds_path, chunks = None):
    G_ds = xr.open_dataset(G_ds_path, chunks = chunks)['__xarray_dataarray_variable__']

    G_CDR_ds = xr.open_dataset(G_cdr_ds_path, chunks = chunks)['__xarray_dataarray_variable__']

    #4th order polyfit
    Gpoly = G_ds.polyfit('year', 4)
    G_ds= xr.polyval(G_ds.year, Gpoly)['polyfit_coefficients']

    Gpoly_cdr = G_CDR_ds.polyfit('year', 4)
    G_CDR_ds= xr.polyval(G_CDR_ds.year, Gpoly_cdr)['polyfit_coefficients']

    G_ds = xr.concat([G_ds, -G_CDR_ds], pd.Index(['pulse','cdr'], name = 'pulse_type'))


    G_ds.name = 'G[tas]'
    G_ds = G_ds.rename({'year':'s'})
    return(G_ds)


########################### 1pct increase ###################################
def compound_mult(start,years, percentage):
    num = start
    arr = np.array(num)
    for year in range(years):
        num += num*percentage
        #print(num)
        arr = np.append(arr, num)
    return(arr)

def np_to_xr(C, G, E):
    E_len = len(E)
    G_len = len(G.s)
    C = xr.DataArray(
    data = C,
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(0, C.shape[0])), #np.arange(0,(E_len+G_len))),
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )
    return(C)

def np_to_xr_mean(C, G, E):
    E_len = len(E)
    G_len = len(G.s)
    C = xr.DataArray(
    data = C,
    dims = ['s'],
    coords = dict(
        s = (['s'], np.arange(0, C.shape[0])), #np.arange(0,(E_len+G_len))),
            )
        )
    return(C)


########################## model weights ###########################
#define our weights for convolution
model_weights = {'UKESM1_r1': 0.25, 'UKESM1_r2': 0.25, 'UKESM1_r3': 0.25, 'UKESM1_r4': 0.25, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'ACCESS': 1,  'CANESM5_r2p2':1/3, 'CANESM5_r1p2':1/3, 'CANESM5_r3p2':1/3}
model_weights = xr.DataArray(
    data=list(model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(model_weights.keys()))
    ),
    attrs=dict(
        description="weights for models"
    ),
)

#define our weights 1pct models
onepct_model_weights = {'UKESM1_r1': 0.25, 'UKESM1_r2': 0.25, 'UKESM1_r3': 0.25, 'UKESM1_r4': 0.25, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'CANESM5_r3p1':1/6, 'ACCESS':1, 'CANESM5_r2p2':1/6, 'CANESM5_r2p1':1/6,
       'CANESM5_r1p2':1/6, 'CANESM5_r1p1':1/6, 'CANESM5_r3p2':1/6}
onepct_model_weights = xr.DataArray(
    data=list(onepct_model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(onepct_model_weights.keys()))
    ),
    attrs=dict(
        description="weights for 1pct models"
    ),
)
#define our weights for G
G_model_weights = {'UKESM1_r1': 1, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'ACCESS': 1,  'CANESM5_r1p2':1/3, 'CANESM5_r2p2':1/3, 'CANESM5_r3p2':1/3}
G_model_weights = xr.DataArray(
    data=list(G_model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(G_model_weights.keys()))
    ),
    attrs=dict(
        description="weights for Green's function"
    ),
)

#define our weights for the pictrl
pictrl_model_weights = {'UKESM1_r1': 1, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'ACCESS': 1,  'CANESM5_r1p1':1/2, 'CANESM5_r1p2':1/2}
pictrl_model_weights = xr.DataArray(
    data=list(pictrl_model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(pictrl_model_weights.keys()))
    ),
    attrs=dict(
        description="weights for pictrl models"
    ),
)

#define our weights for the 4x
fourx_model_weights = {'UKESM1_r1': 1, 'NORESM2': 1,
       'MIROC': 1, 'ACCESS': 1,  'CANESM5_r1p1':1/2, 'CANESM5_r1p2':1/2}
fourx_model_weights = xr.DataArray(
    data=list(fourx_model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(fourx_model_weights.keys()))
    ),
    attrs=dict(
        description="weights for 4x models"
    ),
)

#define our weights for the 1000gtc
onepct_1000gtc_weights = {'UKESM1_r1': 1/4, 'UKESM1_r2': 1/4,
       'UKESM1_r3': 1/4, 'UKESM1_r4': 1/4,  'MIROC':1, 'NORESM2':1, 'ACCESS':1,
                         'CANESM5_r1p2':1/3, 'CANESM5_r2p2':1/3, 'CANESM5_r3p2':1/3}
onepct_1000gtc_weights = xr.DataArray(
    data=list(onepct_1000gtc_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(onepct_1000gtc_weights.keys()))
    ),
    attrs=dict(
        description="weights for 1pct 1000PgC branch models"
    ),
)


##### Shift colormap (from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib)
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


################################# dataset dictionaries ###########################


model_run_pulse_dict = {'UKESM1_r1':'UKESM1-0-LL_esm-pi-CO2pulse_r1i1p1f2*', 
                        'MIROC':'MIROC-ES2L_esm-pi-CO2pulse_r1i1p1f2*', 
                        'NORESM2':'NorESM2-LM_esm-pi-CO2pulse_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_esm-pi-CO2pulse_r1i1p1f1*',  
                        'GFDL': 'GFDL-ESM4_esm-pi-CO2pulse_r1i1p1f1**',
                       'CANESM5_r1p2':'CanESM5_esm-pi-CO2pulse_r1i1p2f1*',
                       'CANESM5_r2p2':'CanESM5_esm-pi-CO2pulse_r2i1p2f1*',
                       'CANESM5_r3p2':'CanESM5_esm-pi-CO2pulse_r3i1p2f1*'}

model_run_cdr_pulse_dict = {'UKESM1_r1':'UKESM1-0-LL_esm-pi-cdr-pulse_r1i1p1f2*', 
                        'MIROC':'MIROC-ES2L_esm-pi-cdr-pulse_r1i1p1f2*', 
                        'NORESM2':'NorESM2-LM_esm-pi-cdr-pulse_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_esm-pi-cdr-pulse_r1i1p1f1*',  
                        'GFDL': 'GFDL-ESM4_esm-pi-cdr-pulse_r1i1p1f1**',
                       'CANESM5_r1p2':'CanESM5_esm-pi-cdr-pulse_r1i1p2f1*',
                       'CANESM5_r2p2':'CanESM5_esm-pi-cdr-pulse_r2i1p2f1*',
                       'CANESM5_r3p2':'CanESM5_esm-pi-cdr-pulse_r3i1p2f1*'}


model_run_esm_picontrol_dict = {'UKESM1_r1':'UKESM1-0-LL_esm-piControl_r1i1p1f2*', 
                          'MIROC':'MIROC-ES2L_esm-piControl_r1i1p1f2*', 
                          'NORESM2':'NorESM2-LM_esm-piControl_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_esm-piControl_r1i1p1f1*', 
                          'GFDL': 'GFDL-ESM4_esm-piControl_r1i1p1f1**',
                         'CANESM5_r1p2':'CanESM5_esm-piControl_r1i1p2f1*',
                          'CANESM5_r1p1':'CanESM5_esm-piControl_r1i1p1f1*',
                         } ## for use with pulse run

model_run_picontrol_dict = {'UKESM1_r1':'UKESM1-0-LL_piControl_r1i1p1f2*', 
                          'MIROC':'MIROC-ES2L_piControl_r1i1p1f2*', 
                          'NORESM2':'NorESM2-LM_piControl_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_piControl_r1i1p1f1*', 
                          'GFDL': 'GFDL-ESM4_piControl_r1i1p1f1**',
                         'CANESM5_r1p2':'CanESM5_piControl_r1i1p2f1*',
                          'CANESM5_r1p1':'CanESM5_piControl_r1i1p1f1*',
                         } ## for use with 1pct run


model_run_1pct_dict = {'UKESM1_r1':'UKESM1-0-LL_1pctCO2_r1i1p1f2*',
                       'UKESM1_r2':'UKESM1-0-LL_1pctCO2_r2i1p1f2*',
                       'UKESM1_r3':'UKESM1-0-LL_1pctCO2_r3i1p1f2*',
                       'UKESM1_r4':'UKESM1-0-LL_1pctCO2_r4i1p1f2*',
                        'MIROC':'MIROC-ES2L_1pctCO2_r1i1p1f2*', 
                        'NORESM2':'NorESM2-LM_1pctCO2_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_1pctCO2_r1i1p1f1*',  
                        'GFDL': 'GFDL-ESM4_1pctCO2_r1i1p1f1**',
                      'CANESM5_r1p2':'CanESM5_1pctCO2_r1i1p2f1*',
                      'CANESM5_r2p2':'CanESM5_1pctCO2_r2i1p2f1*',
                      'CANESM5_r3p2':'CanESM5_1pctCO2_r3i1p2f1*',
                      'CANESM5_r1p1':'CanESM5_1pctCO2_r1i1p1f1*',
                      'CANESM5_r2p1':'CanESM5_1pctCO2_r2i1p1f1*',
                      'CANESM5_r3p1':'CanESM5_1pctCO2_r3i1p1f1*'}

model_run_1pct_1000gtc_dict = {'UKESM1_r1':'UKESM1-0-LL_esm-1pct-brch-1000PgC_r1i1p1f2*',
                       'UKESM1_r2':'UKESM1-0-LL_esm-1pct-brch-1000PgC_r2i1p1f2*',
                       'UKESM1_r3':'UKESM1-0-LL_esm-1pct-brch-1000PgC_r3i1p1f2*',
                       'UKESM1_r4':'UKESM1-0-LL_esm-1pct-brch-1000PgC_r4i1p1f2*',
                        'MIROC':'MIROC-ES2L_esm-1pct-brch-1000PgC_r1i1p1f2_gn*', 
                        'NORESM2':'NorESM2-LM_esm-1pct-brch-1000PgC_r1i1p1f1_*', 
                  'ACCESS':'ACCESS-ESM1-5_esm-1pct-brch-1000PgC_r1i1p1f1*',  
                       # 'GFDL': 'GFDL-ESM4_esm-1pct-brch-1000PgC_r1i1p1f1**', no gfdl data beyond tas
                      'CANESM5_r1p2':'CanESM5_esm-1pct-brch-1000PgC_r1i1p2f1*',
                      'CANESM5_r2p2':'CanESM5_esm-1pct-brch-1000PgC_r2i1p2f1*',
                      'CANESM5_r3p2':'CanESM5_esm-1pct-brch-1000PgC_r3i1p2f1*'}

model_run_A1_B1_dict = {'GFDL_B1': 'GFDL-ESM2M_esm-bell-1000PgC_1861_2360.csv',
                       'NORESM2_B1':'NorESM2-LM_esm-bell-1000PgC_1850-2049.csv',
                       'GFDL_A1': 'GFDL-ESM2M_esm-1pct-brch-1000PgC_1861_2360.csv',
                       'NORESM2_A1':'NorESM2-LM_esm-1pct-brch-1000PgC_0066-0167.csv'}


model_run_4x_dict = {'UKESM1_r1':'UKESM1-0-LL_abrupt-4xCO2_r1i1p1f2*',
                        'MIROC':'MIROC-ES2L_abrupt-4xCO2_r1i1p1f2_gn*', 
                        'NORESM2':'NorESM2-LM_abrupt-4xCO2_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_abrupt-4xCO2_r1i1p1f1*',  
                        #'GFDL': 'GFDL-ESM4_abrupt-4xCO2_r1i1p1f1**',
                      'CANESM5_r1p2':'CanESM5_abrupt-4xCO2_r1i1p2f1_gn*',
                      'CANESM5_r1p1':'CanESM5_abrupt-4xCO2_r1i1p1f1_gn*'
                    }
model_run_hist_co2_dict = {'CANESM5_r1p1':'CanESM5_hist-CO2_r1i1p1f1*',
                      'CANESM5_r2p1':'CanESM5_hist-CO2_r2i1p1f1*',
                      'CANESM5_r3p1':'CanESM5_hist-CO2_r3i1p1f1*',
                      'CANESM5_r4p1':'CanESM5_hist-CO2_r4i1p1f1*',
                      'CANESM5_r5p1':'CanESM5_hist-CO2_r5i1p1f1*',
                      'CANESM5_r6p1':'CanESM5_hist-CO2_r6i1p1f1*',
                      'CANESM5_r7p1':'CanESM5_hist-CO2_r7i1p1f1*',
                      'CANESM5_r8p1':'CanESM5_hist-CO2_r8i1p1f1*',
                      'CANESM5_r9p1':'CanESM5_hist-CO2_r9i1p1f1*'}

################## colors ######################

type_color = {'model_1pct': 'darkcyan',
              'model_1000gtc': 'darkcyan',
             'emulator_1pct': 'maroon',
              'emulator_1000gtc':'maroon'}

model_color = {'UKESM1_r1':'darkgreen', 'UKESM1_r2':'mediumaquamarine', 'UKESM1_r3':'seagreen', 'UKESM1_r4':'lightgreen', 'NORESM2':'blue', 'GFDL':'red', 'MIROC':'purple', 'ACCESS':'pink', 'CANESM5_r1p2':'orange', 'CANESM5_r2p2':'sienna', 'CANESM5_r3p2':'goldenrod', 'CANESM5_r1p1':'sienna','mean':'black'}

#type_color = {'model':'maroon', 'all':'darksalmon',  'pulse':'darkcyan', 'cdr':'darkgreen'} 

proper_names = {'UKESM1_r1':'UKESM1', 'MIROC':'MIROC', 'NORESM2':'NORESM2', 'ACCESS':'ACCESS', 'GFDL':'GFDL', 'CANESM5_r1p2':'CANESM5'}