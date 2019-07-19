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
pckl = r"C:\Users\user\Google Drive\OCT\Measurements\20190410 - step PDMS 2\processed\20190410 - step PDMS 2_processed.bz2"
sfile = bz2.BZ2File(pckl, 'r')
data = pickle.loads(sfile.read())
regions = list(data.keys())
sfile.close()

# +
## data cleanup and conversions
locs = list(data.keys())
profile = {}

px_to_d_vert = 5.75 # micron per pixel
for loc in locs:
    profile[loc] = np.squeeze((max(data[loc]["profile"])-data[loc]["profile"]))[35:-15]*px_to_d_vert
    
# aperture size is 4mm, divided by the number of columns (A-scans) that it takes to image it
px_to_d_hor = 4/len(profile[locs[0]]) 
hor = np.squeeze(np.arange(0,4,px_to_d_hor))


# + {"code_folding": [0]}
## Parabolic Fit - NOT LOOKING TOO GOOD
def parabola_eq(x, a, center, offset):
    return np.array(a*(x-center)**2+offset)
i = 7
init_guess = [-100, 2, 350]
lower = [-1000, 1.5, 0]
upper = [0 , 2.5, 1000]
b = (lower, upper)
best_vals, covar = curve_fit(
    parabola_eq, hor, profile[locs[i]],
    p0 = init_guess, bounds = b,
    method = 'trf',
)   
parabola1 = parabola_eq(hor, best_vals[0], best_vals[1], best_vals[2])
print(f"for loc: {locs[i]}, fit: {best_vals}")

fit_parabola = np.polyfit(x=hor, y=profile[locs[i]], deg=2)
parabola2 = np.poly1d(np.squeeze(fit_parabola))

fig, ax1 = plt.subplots()
# ax1.plot(hor,profile[locs[0]],'ro',label = 'data')
ax1.plot(hor,profile[locs[i]],'go',label = 'data')
ax1.plot(hor, parabola1, label='parabolic fit', c='darkorange')
# ax1.plot(hor, parabola2(hor), label='parabolic fit 2', c='darkgreen')

ax1.set_label('Profile px')
plt.show()

# +
N = 3 # locations

fig, ax = plt.subplots(ncols=N, sharey = True, figsize = (8, 4))

for i in range(N):
    ax[i].set(xlabel='Profile position (mm)')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
#     ax[i].get_xaxis().set_ticks([])
#     ax[i].get_yaxis().set_ticks([])
#     ax[i].set_xlim(-2, 10)
#     ax[i].set_ylim(-3, 2)

ax[0].plot(hor,profile["thin_suc"])
ax[1].plot(hor,profile["thick_suc"])
ax[2].plot(hor,profile["boundary_suc"])
ax[0].plot(hor,profile["2thin_suc"])
ax[1].plot(hor,profile["2thick_suc"])
ax[2].plot(hor,profile["2boundary_suc"])

ax[0].set(ylabel='$\Delta$z ($\mu$m)')

sup_title='Vertical deformation $\Delta$z, in three regions of the phantom, at 500mbar suction'
st = fig.suptitle(sup_title, fontsize=14)


ax[0].set_title('Top-layer thinner')
ax[1].set_title('Top-layer thicker')
ax[2].set_title('Transition region')

fig.tight_layout()
# fig.savefig("test.png")

st.set_y(0.95)
fig.subplots_adjust(top=0.8)

plt.show()
# -


