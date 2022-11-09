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
from dateutil.relativedelta import relativedelta


########################### CMIP 6 DATA AND REGRIDDING ###################################

#### initial import and data merging ####
#subroutines to import

def _import_combine_pulse_control(control_path, pulse_path, replace_xy):
    #import and check files
    if replace_xy == True:
        ds_control = xr.open_mfdataset(control_path, preprocess = combined_preprocessing, use_cftime=True)
        ds_pulse = xr.open_mfdataset(pulse_path, preprocess = combined_preprocessing, use_cftime=True)
    else:
        ds_control = xr.open_mfdataset(control_path, use_cftime=True)
        ds_pulse = xr.open_mfdataset(pulse_path, use_cftime=True)
    if ds_control.attrs['parent_source_id'] != ds_pulse.attrs['parent_source_id']:
        print('WARNING: Control and Pulse runs are not from the same parent source!')
    #select only the times that match up with the pulse
    ds_control = ds_control.sel(time = slice(ds_control['time'].min(), ds_pulse['time'].max()))
    #fix the lat lon/xy gridding (xmip)
    if replace_xy == True:
        ds_control = replace_x_y_nominal_lat_lon(ds_control)
        ds_pulse = replace_x_y_nominal_lat_lon(ds_pulse)

    return(ds_control, ds_pulse)


def _regrid_cont_pulse(ds_control, ds_pulse, ds_out):
    regridder = xe.Regridder(ds_control, ds_out, "bilinear")
    ds_control = regridder(ds_control) 

    regridder = xe.Regridder(ds_pulse, ds_out, "bilinear")
    ds_pulse = regridder(ds_pulse) 
    
    return(ds_control, ds_pulse)


def _calc_greens(ds_control, ds_pulse, variable, pulse_size = 100):
    
    #A = find_area(ds_control.isel(time = 0), lat_bound_nm = 'lat_bounds', lon_bound_nm = 'lon_bounds')
    G = (ds_pulse[variable] - ds_control[variable])/(pulse_size)
    G = G.groupby('time.year').mean()
    G.attrs = ds_pulse.attrs
    
    return(G)


#full function
def import_regrid_calc(control_path, pulse_path, ds_out, variable, pulse_size = 100,  replace_xy = True, regrid = True):
    '''Imports the control run and pulse run for a CMIP6 model run, combines them on the date the pulse starts
    Regrids it to the chosen grid size
    Calculates the Green's Function'''
    
    ds_control, ds_pulse = _import_combine_pulse_control(control_path, pulse_path, replace_xy)
    if regrid == True:
        ds_control, ds_pulse = _regrid_cont_pulse(ds_control, ds_pulse, ds_out)
    G = _calc_greens(ds_control, ds_pulse, variable)
    
    return(ds_control, ds_pulse, G)


#### single regridder ####
def _regrid_ds(ds_in, ds_out):
    regridder = xe.Regridder(ds_in, ds_out, "bilinear")
    ds_new = regridder(ds_in) 
    return(ds_new)


#### function to find area of a grid cell from lat/lon ####
def find_area(ds, R = 6378.1):
    """ ds is the dataset, i is the number of longitudes to assess, j is the number of latitudes, and R is the radius of the earth in km. 
    Must have the ds['lat'] in descending order (90...-90)
    Returns Area of Grid cell in km"""
    
    dy = (ds['lat_b'].roll({'lat_b':-1}, roll_coords = False) - ds['lat_b'])[:-1]*2*np.pi*R/360 

    dx1 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*2*np.pi*R*np.cos(np.deg2rad(ds['lat_b']))
    
    dx2 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*2*np.pi*R*np.cos(np.deg2rad(ds['lat_b'].roll({'lat_b':-1}, roll_coords = False)[:-1]))
    
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