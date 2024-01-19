#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00

#SBATCH --cpus-per-task=3
#SBATCH --mem=18G


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
import utils
import cf_xarray as cfxr

from sklearn.linear_model import LinearRegression
import scipy.signal as signal
from scipy import stats
from datetime import timedelta

import seaborn as sns
import matplotlib as mpl
import cmocean
import cmocean.cm as cmo
from matplotlib.gridspec import GridSpec

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import argparse

import string
alphabet = list(string.ascii_lowercase)       


# In[2]:


dask.config.set(**{'array.slicing.split_large_chunks': True})


# # Import data

# ## Green's Function

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--pulse_year',type=int, required=True) 

args = parser.parse_args()

p = args.pulse_year


G_ds_path = 'Outputs/G_internal_var_ds.nc4'
G_cdr_ds_path = 'Outputs/G_cdr_internal_var_ds.nc4'
chunks = {'pulse_type':2,'model':8, 'pulse_year':20}
G_ds = utils.import_polyfit_G(G_ds_path, G_cdr_ds_path, chunks = chunks).sel(pulse_year = p)


# ## Convolution

# In[4]:


conv_mean_ds = xr.open_dataset('Outputs/conv_mean_ds.nc4')['__xarray_dataarray_variable__']

conv_ds = xr.open_dataset('Outputs/conv_ds.nc4')['__xarray_dataarray_variable__']


# ## CMIP6 1pct

# In[5]:


ds_dif = xr.open_dataset('Outputs/ds_dif.nc4')


# ## Emissions profile

# In[6]:


emis_profile = xr.open_dataset('Outputs/emis_profile.nc4')


# ## RTCRE

# In[7]:


RTCRE = xr.open_dataset('Outputs/RTCRE.nc')


# # Settings and Define our Model Weights

# In[8]:


#define our weights for models (grouping UKESM and CANESM realizations)
model_weights = utils.model_weights

onepct_model_weights = utils.onepct_model_weights

G_model_weights = utils.G_model_weights


# In[9]:


type_color = utils.type_color


# In[10]:


A = utils.A
ds_out = utils.ds_out


# # Multiple options for path to 2 degrees C

# In[11]:


T = 69 #years to 2 degrees for the global mean
cum_emis = emis_profile.sel(year = slice(0,T)).mean(dim = ['model']).sel(experiment = '1pct').sum()['emis'].values


# In[12]:


cum_emis


# In[13]:


#function for getting emissions at time, t, that matches the cumulative goal

def poly_fit_cumulative_emis(T, n, t, c):
    '''T is the years at which we reach a given cumulative emissions,
    n is the polynomial fit we want,
    t is the time range,
    c is the cumulative emissions goal'''
    T = T+0.5
    e = (c*(n+1)*t**n)/(T**(n+1))
    return(e)


# In[14]:


#create our emissions based on polynomial values
n_range = [1/8,1/4,1/2,2,4,8]
e_range = {}
for n in n_range:
    e_range[n] = poly_fit_cumulative_emis(T, n, np.arange(0,90), cum_emis)


# In[16]:


for n in n_range:
    plt.plot(e_range[n].cumsum())
print(emis_profile.sel(year = slice(0,T)).mean(dim = ['model']).sel(experiment = '1pct').sum()['emis'].values)
plt.xlim(60,71)
plt.ylim(1100,1280)
plt.axvline(69, color = 'k')


# In[41]:

conv_2degC = {}
for m in G_ds.model.values:
    conv_2degC[m] = {}
    GF = G_ds.sel(model = m).mean(dim = ['pulse_type'])
    for n in n_range:
        conv_2degC[m][n] = signal.convolve(np.array(GF.dropna(dim = 's')), e_range[n][..., None, None],'full')
        conv_2degC[m][n] = utils.np_to_xr(conv_2degC[m][n], GF, e_range[n])
        conv_2degC[m][n].to_netcdf(f'Outputs/2degC_convolution_var/{m}_{p}_{n}.nc')

# In[43]:


# conv_2degC_dict = {}
# for m in conv_2degC.keys():
#     conv_2degC_dict[m] = {}
#     for p in conv_2degC[m].keys():
#         conv_2degC_dict[m][p] = xr.concat([conv_2degC[m][p][n] for n in conv_2degC[m][p].keys()], 
#                                     pd.Index([n for n in conv_2degC[m][p].keys()], name='polyfit'), coords='minimal')



# # In[51]:


# for m in conv_2degC.keys():
#     conv_2degC_dict[m] = xr.concat([conv_2degC_dict[m][p] for p in conv_2degC[m].keys()], 
#                                     pd.Index([p for p in conv_2degC[m].keys()], name='pulse_year'), coords='minimal')

    
# conv_2degC_ds = xr.concat([conv_2degC_dict[m] for m in conv_2degC_dict.keys()], 
#                           pd.Index([p for p in conv_2degC_dict.keys()], name='model'), coords='minimal')


# conv_2degC_ds.to_netcdf('Outputs/2degC_convolution_var.nc')


# model_var = conv_2degC_ds.var(dim = ['model']).mean(dim = ['pulse_year','polyfit'])


# # In[ ]:


# scenario_var = conv_2degC_ds.var(dim = ['polyfit']).mean(dim = ['pulse_year','model'])


# # In[ ]:


# internal_var = conv_2degC_ds.var(dim = ['pulse_year']).mean(dim = ['polyfit','model'])


# # In[ ]:


# total_var = internal_var + model_var + scenario_var


# # In[40]:


# plt.stackplot(model_var['year'], 
#               scenario_var/total_var,
#               model_var/total_var,
#               internal_var/total_var,
#               labels = ['Scenario Variance', 'Model Variance', 'Internal Variance'],
#              alpha = 0.7);
# plt.legend(loc = 'lower right')
# plt.xlim(0,90)
# plt.ylim(0,1)
# plt.ylabel('Percent of total Variance')
# plt.xlabel('Year')
# plt.savefig('figures/paper/supp_variance_model_2degrees.png', bbox_inches = 'tight')

