#!/usr/bin/env python
# coding: utf-8


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


from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


import string
alphabet = list(string.ascii_lowercase)
          


# ## Data
# 
# Data for this is from https://gmd.copernicus.org/articles/11/1133/2018/ CDRMIP data, where pi-CO2pulse is the 100GtC pulse and piControl is the control


G_ds = xr.open_dataset('Outputs/G_internal_var_ds.nc4')['__xarray_dataarray_variable__']
G_cdr_ds = xr.open_dataset('Outputs/G_cdr_internal_var_ds.nc4')['__xarray_dataarray_variable__']


# In[5]:


G_ds = xr.concat([G_ds, -G_cdr_ds], pd.Index(['pulse','cdr'], name = 'pulse_type'))
G_ds.name = 'G[tas]'


# In[6]:


A = utils.A


# In[7]:


model_color = utils.model_color


# ## Global Mean plots

# In[ ]:


G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').mean(dim = 'pulse_year')

print('G ready')



fig, axes = plt.subplots(2,4,figsize = [30,10], sharey = True, sharex = True)
alpha_labels = iter(alphabet)

plt.suptitle('Annual Mean Green\'s Function', fontsize = 16)
for ax, m in zip(axes.ravel(), G_ds.model.values):
    print(m)
    G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').sel(pulse_year = 0).plot(ax = ax, color = model_color[m])
    
    ax.set_xlim(G_ds.year.min(), G_ds.year[150].values)
    ax.set_title(m, fontsize = 14)
    
    ax.set_ylabel('')
    ax.set_xlabel('')
    label = next(alpha_labels)
    ax.text(x = .06, y =.94, s = label, transform=ax.transAxes, fontweight="bold", fontsize = 14)

#     #inter model spread
#     ax.fill_between(np.arange(0,len(G_ds.sel(model = m)['year'])),
#             G_ds.weighted(A).mean(dim = ['lat','lon']).sel(pulse_year = 0).mean(dim = 'pulse_type').min(dim = 'model'),
#             G_ds.weighted(A).mean(dim = ['lat','lon']).sel(pulse_year = 0).mean(dim = 'pulse_type').max(dim = 'model'), color = 'khaki', alpha = 0.2)
    ax.fill_between(np.arange(0,len(G_ds['year'])),
                 G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').mean(dim = 'pulse_year') + G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').std(dim = 'pulse_year'),
                 G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').mean(dim = 'pulse_year') + G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').std(dim = 'pulse_year'), color = 'grey', alpha = 0.2)
    ax.fill_between(np.arange(0,len(G_ds.sel(model = m)['year'])),
        G_ds.weighted(A).mean(dim = ['lat','lon']).sel(pulse_year = 0).mean(dim = 'pulse_type').mean(dim = 'model') + G_ds.weighted(A).mean(dim = ['lat','lon']).sel(pulse_year = 0).mean(dim = 'pulse_type').std(dim = 'model'),
        G_ds.weighted(A).mean(dim = ['lat','lon']).sel(pulse_year = 0).mean(dim = 'pulse_type').mean(dim = 'model') + G_ds.weighted(A).mean(dim = ['lat','lon']).sel(pulse_year = 0).mean(dim = 'pulse_type').std(dim = 'model'), color = 'khaki', alpha = 0.2)

    #internal var spread
#     ax.fill_between(np.arange(0,len(G_ds.sel(model = m)['year'])),
#                  G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').min(dim = 'pulse_year'),
#                  G_ds.weighted(A).mean(dim = ['lat','lon']).sel(model = m).mean(dim = 'pulse_type').max(dim = 'pulse_year'), color = 'grey', alpha = 0.2)
#     break
    
axes[0,0].set_ylabel('G [$\degree$K/GtC]', fontsize = 14)
axes[1,0].set_ylabel('G [$\degree$K/GtC]', fontsize = 14)
for idx in [0,1,2,3]:
    axes[1,idx].set_xlabel('Years', fontsize = 14)


####### legend ##########
plt.tight_layout()
plt.savefig('figures/paper/supp_GF_internal_var_model_spread.png', bbox_inches = 'tight', dpi = 300)




