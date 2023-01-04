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
    #import and check files
    #if replace_xy == True:
       # ds_control = xr.open_mfdataset(control_path, use_cftime=True)
       # ds_pulse = xr.open_mfdataset(pulse_path, use_cftime=True)
    #else:
    ds_control = xr.open_mfdataset(control_path, use_cftime=True)
    ds_pulse = xr.open_mfdataset(pulse_path, use_cftime=True)
    
#     dlon = (ds_control['lon'][1] - ds_control['lon'][0])
#     ds_control['lon_b'] = np.arange(0,360+dlon,dlon)
#     dlat = (ds_control['lat'][1] - ds_control['lat'][0])
#     ds_control['lat_b'] = np.arange(-90,90+dlat,dlat)
#     ds_control = ds_control.drop(['lat_bnds','lon_bnds'])
    #regridder = xe.Regridder(ds, ds_out, "conservative")
#     print(ds_control)
#     dlon = (ds_pulse['lon'][1] - ds_pulse['lon'][0])
#     lon_b = np.arange(ds_pulse['lon'][0] - .5*dlon, ds_pulse['lon'][-1] + 1.5*dlon, dlon)
#     #lon_b = lon_b.loc[lon_b > 0]
#     ds_pulse['lon_b'] = lon_b
#     dlat = (ds_pulse['lat'][1] - ds_pulse['lat'][0])
#     lat_b = np.arange(ds_pulse['lat'][0] - .5*dlat, ds_pulse['lat'][-1] + 1.5*dlat, dlat)
#     ds_pulse['lat_b'] = lat_b
#     ds_pulse = ds_pulse.drop(['lat_bnds','lon_bnds'])
    lat_corners = cfxr.bounds_to_vertices(ds_control.isel(time = 0)['lat_bnds'], "bnds", order=None)
    lon_corners = cfxr.bounds_to_vertices(ds_control.isel(time = 0)['lon_bnds'], "bnds", order=None)
    ds_control = ds_control.assign(lon_b=lon_corners, lat_b=lat_corners)

    lat_corners = cfxr.bounds_to_vertices(ds_pulse.isel(time = 0)['lat_bnds'], "bnds", order=None)
    lon_corners = cfxr.bounds_to_vertices(ds_pulse.isel(time = 0)['lon_bnds'], "bnds", order=None)
    ds_pulse = ds_pulse.assign(lon_b=lon_corners, lat_b=lat_corners)

    #print(ds_pulse)
        #fix the lat lon/xy gridding (xmip)
    #if replace_xy == True:
       # ds_control = replace_x_y_nominal_lat_lon(ds_control)
       # ds_pulse = replace_x_y_nominal_lat_lon(ds_pulse)
    #ds_control = ds_control.rename({'x':'latitude', 'y':'longitude'})
    #ds_pulse = ds_pulse.rename({'x':'latitude', 'y':'longitude'})
    if ds_control.attrs['parent_source_id'] != ds_pulse.attrs['parent_source_id']:
        print('WARNING: Control and Pulse runs are not from the same parent source!')
    #fix the time for two of the models
    if m == 'NORESM2':
        ds_pulse['time'] = ds_control['time'][:len(ds_pulse['time'])]
    if m =='CANESM5_r1p2':
        ds_pulse['time'] = ds_control['time'][:len(ds_pulse['time'])]
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


def _calc_greens(ds_control, ds_pulse, variable, m, pulse_size = 100):
    
    #A = find_area(ds_control.isel(time = 0), lat_bound_nm = 'lat_bounds', lon_bound_nm = 'lon_bounds')
    G = (ds_pulse[variable] - ds_control[variable])/(pulse_size)
    times = pd.date_range('2000', periods=len(G['time']), freq='MS')
    weights = times.shift(1, 'MS') - times
    weights = xr.DataArray(weights, [('time', G['time'].values)]).astype('float')
    G =  (G * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
    #select ten years in for two of the models
    ten_years_in = 10 #in months
    if m == 'ACCESS':
        G = G.isel(year = slice(ten_years_in,len(G.year)))
    if m == 'UKESM1':
        G = G.isel(year = slice(ten_years_in,len(G.year)))
    #G = G.groupby('time.year').mean()
    G.attrs = ds_pulse.attrs
    
    return(G)

def _calc_anomaly(ds_control, ds_pulse):
    anom_control = (ds_control.groupby("time.month") - ds_control.groupby("time.month").mean(dim = 'time'))
    anom_pulse = (ds_pulse.groupby("time.month") - ds_pulse.groupby("time.month").mean(dim = 'time'))
    return(anom_control, anom_pulse)

def _calc_greens_anomaly(anom_control, anom_pulse, variable, pulse_size = 100):
    
    #A = find_area(ds_control.isel(time = 0), lat_bound_nm = 'lat_bounds', lon_bound_nm = 'lon_bounds')
    G = (anom_pulse[variable] - anom_control[variable])/(pulse_size)
    G = G.groupby('time.year').mean()
    G.attrs = anom_pulse.attrs
    
    return(G)


#full function
def import_regrid_calc(control_path, pulse_path, ds_out, variable, m, pulse_size = 100,  replace_xy = True, regrid = True, anomaly = False):
    '''Imports the control run and pulse run for a CMIP6 model run, combines them on the date the pulse starts
    Regrids it to the chosen grid size
    Calculates the Green's Function'''
    
    ds_control, ds_pulse = _import_combine_pulse_control(control_path, pulse_path, replace_xy, m)
    if regrid == True:
        ds_control, ds_pulse = _regrid_cont_pulse(ds_control, ds_pulse, ds_out)
    if anomaly == True:
        anom_control, anom_pulse = _calc_anomaly(ds_control, ds_pulse)
        anom_G = _calc_greens_anomaly(anom_control, anom_pulse, variable, pulse_size)
    else:
        G = _calc_greens(ds_control, ds_pulse, variable, m, pulse_size)
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
########################### convolutions ###################################

def convolve_single_lev(G, E, dt):
    '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
    E_len = len(E)
    G_len = len(G.year)
    C = dask.array.empty(((E_len+G_len), len(G.lat), len(G.lon))) 
    for i, tp in enumerate(np.arange(0,E_len)):
        C[i:i+G_len] = C[i:i+G_len]+ G*E[i]*dt #C.loc slice or where
        #print((G*E[i]*dt).values)
        #print(C.compute())
    C = xr.DataArray(
    data = C,
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(0,(E_len+G_len))),
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )
    return C

def convolve_single_lev_mean(G, E, dt):
    '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
    E_len = len(E)
    G_len = len(G.year)
    C = dask.array.empty((E_len+G_len)) 
    for i, tp in enumerate(np.arange(0,E_len)):
        C[i:i+G_len] = C[i:i+G_len]+ G*E[i]*dt #C.loc slice or where
        #print((G*E[i]*dt).values)
        #print(C.compute())
    C = xr.DataArray(
    data = C,
    dims = ['s'],
    coords = dict(
        s = (['s'], np.arange(0,(E_len+G_len))),
            )
        )
    return C

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

model_color = {'UKESM1_r1':'olive', 'UKESM1_r2':'brown', 'UKESM1_r3':'green', 'UKESM1_r4':'lightgreen', 'NORESM2':'blue', 'GFDL':'red', 'MIROC':'purple', 'ACCESS':'pink', 'CANESM5_r1p2':'orange', 'CANESM5_r2p2':'sienna', 'CANESM5_r3p2':'goldenrod', 'mean':'black'}
