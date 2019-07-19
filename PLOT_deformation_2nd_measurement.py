# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
## imports
# %matplotlib nbagg
# # %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %reload_ext autoreload
# %autoreload 2
# %aimport bscan_class
import os 

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.plotly as py
from plotly import tools
init_notebook_mode(connected=True) # to use in Jupyter

import bz2 # to zip the pickle file: 
import pickle

# + {"code_folding": []}
## unpickling
pckl = r"C:\Users\lucab\Desktop\20190718_PDMS_Sample20180219_2\20190718_PDMS_Sample20180219_2_processed.bz2"
init_folder = os.path.dirname(pckl)
sfile = bz2.BZ2File(pckl, 'r')
data = pickle.loads(sfile.read())
regions = list(data.keys())
sfile.close()

# +
px_to_d_vert = 5.75 # micron per pixel
regions = ['thin', 'thick', 'boundary']
reps = ['001', 
        '002',
        '003',
        '004', 
        '005'
       ]

# blueprint -> profile[region][rep]
deformation = {}
for i,meas in enumerate(data.keys()):
    for region in regions:
        deformation[region]=[]
        for rep in reps:
            name_rel = region+'_rel_'+rep
            name_suc = f"{region}_suc_{rep}"
            deformation[region].append(
                np.squeeze(np.array(
                    ((data[name_suc]["profile"])-(data[name_suc]["profile"][0])) -
                    ((data[name_rel]["profile"])-(data[name_rel]["profile"][0]))                    
                )*-px_to_d_vert)
            )

# aperture size is 4mm, divided by the number of columns (A-scans) that it takes to image it
px_to_d_hor = 4/len(data[name_rel]["profile"]) 
hor = np.squeeze(np.arange(0,4,px_to_d_hor))

# +
N = len(regions) # locations

fig, ax = plt.subplots(ncols=N, sharey = True, figsize = (8, 4))

for i in range(N):
    ax[i].set(xlabel='Profile position (mm)')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].set_title(f"{regions[i]}")
#     ax[i].get_xaxis().set_ticks([])
#     ax[i].get_yaxis().set_ticks([])

for i in range(N):
    for j in range(len(deformation[regions[0]])):
        ax[i].plot(hor, deformation[regions[i]][j], alpha=0.6, c = 'forestgreen')
    
ax[0].set(ylabel='$\Delta$z ($\mu$m)')

sup_title='Vertical deformation $\Delta$z, in three regions of the phantom, at 500mbar suction'
st = fig.suptitle(sup_title, fontsize=14)

# ax[0].set_title('Top-layer thinner')
# ax[1].set_title('Top-layer thicker')
# ax[2].set_title('Transition region')

fig.tight_layout()

st.set_y(0.95)
fig.subplots_adjust(top=0.8)

plt.show()
fname = init_folder+"\\Deformations_comparison.png"
# fig.savefig(fname, dpi = 600)
# -


