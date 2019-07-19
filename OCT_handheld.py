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

# + {"code_folding": []}
## imports
# %matplotlib nbagg
# # %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

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

import cv2 as cv
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese, checkerboard_level_set)
# -

# initial_path = r"C:\Users\user\Google Drive\OCT\Measurements\20190410 - step PDMS 2"
initial_path = r"C:\Users\lucab\Desktop\20190718_PDMS_Sample20180219_2"
files = os.listdir(initial_path)
files = [f for f in files if os.path.isfile(os.path.join(initial_path, f))] # f not in ['processed'] ]
# issue: the first file created by LabView is always empty. I am deleting it from the list
print(f'Found {len(files)} files')
debug = True


# + {"code_folding": [25, 30]}
def get_level_set(img, **kwargs):
    '''
    Returns an initial level set, also called "mask", i.e. a matrix of booleans with
    the same shape as the `img` parameter.
    By default, it returns a `checkerboard level set` as described in scikit image module. 
    Alternatively, allows:
    `mode='threshold', lvl=0..1` and return a mask for intensity > lvl*255
    '''
    mode = kwargs.get('mode', 'checkerboard')
    lvl = kwargs.get('lvl', 6)
    
    if mode in ['threshold', 'thr']:
        init_ls = img > lvl * 255
    elif mode in ['checkerboard', 'checkers']:
        init_ls = checkerboard_level_set(img.shape, lvl)
    elif mode in ['otsu']:
        from skimage.filters import threshold_otsu
        init_ls = img > threshold_otsu(img)
    return init_ls

def filtering(img, **kwargs):
    '''
    Returns a filtered image, a necessary pre-processing step
    '''
    mode = kwargs.get('mode','bilateral')
    if mode in ['bilateral']:
        d = kwargs.get('d', 5)
        sigma_color = kwargs.get('sigma_color', 30)
        sigma_space = kwargs.get('sigma_space', d*3)
        return cv.bilateralFilter(raw[top:bottom,left:right], d, sigma_color, sigma_space)
    elif mode in ['NLM']:
        from skimage.restoration import denoise_nl_means
        sigma_est = kwargs.get('sigma', None)
        if sigma_est is None:
            from skimage.restoration import estimate_sigma
            print('Warning, no parameter sigma passed for this type of filter. Automatic estimation takes computational time!')
            sigma_est = np.mean(estimate_sigma(img, multichannel=False))
            print(sigma_est)
        patch_kw = dict(
                patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False)
        return denoise_nl_means(img, h=12*sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    else:
        raise ValueError('Filtering Mode not understood')
        
def plot_raw (rawimgs: list):
    fig, ax = plt.subplots()
    assert type(rawimgs)==list    
    for rawimg in rawimgs:
        ax.imshow(rawimg)
    plt.show()

def plot_profile (profiles: list):
    fig, ax = plt.subplots()
    assert type(profiles)==list
    for profile in profiles:
        ax.plot(profile)
    plt.show()


# + {"code_folding": []}
#parameters
p=dict(
left = 310, #78,
right = 2238, #580,
top = 300,
bottom = 800,
N_iter = 20,
segmentation_flag = True,
cropping_flag = True,
filtering_flag = True,
savefile_flag = True,
initial_path = initial_path,
filtermode = 'NLM',
sigmaNLM = 3,
bypassmode = 7,
)

# + {"code_folding": []}
d = {}
onoff = lambda flag: "ON" if flag else "OFF"
print(f'Analysis settings: '\
f'\nSegmentation is {onoff(p["segmentation_flag"])}'\
f'\n    Cropping is {onoff(p["cropping_flag"])}'\
f'\n   Filtering is {onoff(p["filtering_flag"])}'\
f'\n    Savefile is {onoff(p["savefile_flag"])} \n')

for i,file in enumerate(files):
    print(f'File "{file}" - {i+1}/{len(files)}')
    d[file]={}
    # relaxed or suction?
    if 'suc' in file:
        d[file]['pressure'] = 'suction'
    elif 'rel' in file:
        d[file]['pressure'] = 'relaxed'
        
    # get raw image
    cur_path = os.path.join(initial_path,file)
    handheld_bscan = bscan_class.bscan(path = cur_path, debug = debug, bypass=8)
    raw = handheld_bscan.raw.T[p['top']:p['bottom'],p['left']:p['right']] \
                        if p['cropping_flag'] else handheld_bscan.raw.T
    # estimating sigmaNLM
    #     sigma_est = np.mean(estimate_sigma(raw, multichannel=False))
    #     print(sigma_est)

    # filtering the image
    im = filtering(raw, mode=p['filtermode'],sigma=p['sigmaNLM'] ) if p['filtering_flag'] else raw
#     im = filtering(raw, mode='bilateral') if p['filtering_flag'] else raw

    d[file]['image'] = im
    
    # segmentation
    if p['segmentation_flag']:
        # level set
        init_ls = get_level_set(im, mode='otsu')  # need to define a `get_level_set` function
        # morphological snakes ACWE
        ls = morphological_chan_vese(im, p['N_iter'], init_level_set=init_ls, smoothing=4)
        d[file]['ls']=ls
        # top contour: searches for first non-zero in the column 
        # (column iteration is obtained by the usual iteration (which is on rows),
        # but performed on the transposed matrix, hernce iterating columns)
        profile = np.zeros((len(ls.T),1))
        for i,a in enumerate(ls.T):
            try:
                # sometimes, in the upper rows of the image, there is the recognition of spurious areas from aritifical reflections, 
                # that messes up the profile detection
                excluded_px = 100 # rows from the top that are excluded from the profile search.
                profile[i] = np.nonzero(a[excluded_px:]>0.5)[0][0]+excluded_px
            except:
                profile[i] = None

        d[file]['profile'] = profile
        
# save data: raw_png, png_with_contours, profile height, delta_z
if p['savefile_flag']:
#     d['params'] = p
    import bz2 # to zip the pickle file: 
    import pickle
    pickle_name = (os.path.normpath(initial_path))+'_processed.bz2'
    print(f"Saving at location: {pickle_name}")
    sfile = bz2.BZ2File(pickle_name, 'w')
    pickle.dump(d, sfile)
    sfile.close()
    
    
# position = 'Left Index2'
# rel = bscan_class.bscan(path = initial_path+r'\\'+position+'_relaxed', debug = True)
# suc = bscan_class.bscan(path = initial_path+r'\\'+position+'_suction', debug = True)
# -

#plot_raw(d['Left Anular_suction']['ls'])#,d['Left Anular_suction']['ls']])
plot_profile([d['Left Anular_suction']['profile'],d['Left Anular_relaxed']['profile']])

title = 'Profile extensions for different locations'
fig, ax = plt.subplots()
ax.set_title(title)
colors = ['#1b9e77','#d95f02','#7570b3']
for i in range(0,len(d),2):
    diff = d[files[i]]['profile']-d[files[i+1]]['profile']
    c = colors[i%3]
    ax.plot(diff, label = files[i][:], c=c, alpha = 0.7)
plt.legend()
plt.show()
fig.savefig(title+'.png',dpi=600)

# + {"code_folding": [22, 25]}
## plotting of segmentation results
plt.ioff()
savefig_flag = True
contour = False
cmap = 'gray' if contour else None
dpi = 300

for f in files[:2]:
    fig, ax = plt.subplots(figsize=(10,4.5));
    ## this crops twice :/
    if p['cropping_flag']:
        cur_path = os.path.join(initial_path,f)
        handheld_bscan = bscan_class.bscan(path = cur_path, debug = debug)
        img = handheld_bscan.raw.T[p['top']:p['bottom'],p['left']:p['right']]
        if p['filtering_flag']: img = filtering(img, mode='NLM')
        ax.imshow(img, cmap=cmap);
    else: # it's the same dumb code as the if above, but img is not cropped 
        cur_path = os.path.join(initial_path,f)
        handheld_bscan = bscan_class.bscan(path = cur_path, debug = debug)
        img = handheld_bscan.raw.T
        if p['filtering_flag']: img = filtering(img, mode='NLM')
        ax.imshow(img, cmap=cmap);

    if contour:
        ax.contour(d[f]['ls'], [0.5], colors='r', alpha = 0.5);    
        title = f'Segmentation Result - {f}'
    else:
        title = f'B-scan for {f}'    
    
    ax.set_title(title);
    ax.set_axis_off();
    fig.tight_layout();
    if savefig_flag: fig.savefig(initial_path+'\\'+title+'.png',dpi=dpi);
# -

# # Second derivative of the profile

# +
from scipy.interpolate import UnivariateSpline
i=7
y = -d[files[i]]['profile'][:,0]
x = np.linspace(0,len(y),len(y))
y_spl = UnivariateSpline(x,y, k=5)
# y_foll = UnivariateSpline(x,y, k=5, s=300)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x,y,'ro',label = 'data')
x_range = x
x_range[:60] = None
x_range[440:] =None # np.linspace(x[0],x[-1],1000)
ax1.plot(x_range,y_spl(x_range) ,label='spline', c='darkorange')
ax1.set_label('Profile px')
y_spl_2d = y_spl.derivative(n=1)
start, stop = 15, 470

ax2.plot(x_range,abs(y_spl_2d(x_range)), 
         label='derivative', 
         c='forestgreen',
         alpha=0.7,
         lw = 3,         
        )   

ax2.set_label('Derivative')
ax2.legend()
ax1.set_title(files[i])
plt.show()
# -

# # Investigation of different thresholding methods, to see what's better.
# ## RESULT: Not much difference btwn the best ones: OTSU, ISODATA and LI. Otsu was arbitrarily chosen
#

# +
from skimage.filters import (threshold_local, threshold_otsu, threshold_niblack, 
                             threshold_sauvola, threshold_yen, threshold_isodata, threshold_li)

def save_raw (rawimgs: list, whatfor: str, titles=None):
    assert type(rawimgs)==list, "Need to pass images as a list; if single image plotting is requested, pass `rawimgs=[image]`" 
    N = len(rawimgs)
    figsize = (8,1+3*N)
    fig, ax = plt.subplots(nrows=N, figsize = figsize,)
    for i,rawimg in enumerate(rawimgs):
        if N==1:
            ax.imshow(rawimg, cmap = 'gray')
            if titles is not None: ax.set_title(titles)
        elif N>1: 
            ax[i].imshow(rawimg, cmap = 'gray')
            if titles is not None: ax[i].set_title(titles[i])
    fname = whatfor+'.png'
    plt.tight_layout()
    fig.savefig(fname, dpi=300)

for f in files:
    image = d[f]['image']
    
    binary_global = image > threshold_otsu(image)
    yen_global = image > threshold_yen(image)
    isodata_global = image > threshold_isodata(image)
    li_global = image > threshold_li(image)
    single_value = image > 0.35*255
    titles= ['otsu','yen','isodata','li','local']
    fname = f'thresholds_{f}'
    save_raw([binary_global,yen_global,isodata_global,li_global,single_value], whatfor=fname, titles=titles)
# -

# # Investigating different bilateral filtering methods:

# +
import cv2 as cv

left = 310
right = 2238
top = 200
bottom = 850

def save_raw_2 (rawimgs: list, whatfor: str, titles=None):
    assert type(rawimgs)==list, "Need to pass images as a list; if single image plotting is requested, pass `rawimgs=[image]`" 
    N = len(rawimgs)
    figsize = (15,1+3*(N/2))
    fig, ax = plt.subplots(nrows=int(N/2), ncols=2, figsize = figsize, sharex=True, sharey=True)
    ax = ax.flatten()
    for i,rawimg in enumerate(rawimgs):
        if N==1:
            ax.imshow(rawimg, cmap = 'gray')
            if titles is not None: ax.set_title(titles)
        elif N>1: 
            ax[i].imshow(rawimg, cmap = 'gray')
            if titles is not None: ax[i].set_title(titles[i])
    fname = whatfor+'.png'
    plt.tight_layout()
    fig.savefig(fname, dpi=600)


# -

for i,file in enumerate(files[:1]):
    print(f'File "{file}" - {i+1}/{len(files)}')
    cur_path = os.path.join(initial_path,file)
    handheld_bscan = bscan_class.bscan(path = cur_path, debug = False)
    ## bilateral filtering
    raw = handheld_bscan.raw.T[top:bottom,left:right]
    d = [5,9,15]
    s = [30,90,200]
    xx = [(x0,x1) for x0 in d for x1 in s] #
    titles = ['(d, sigma_col)='+str(x) for x in xx]
    titles.insert(0,'raw')
    images = [filtering(raw,mode='bilateral',d=x[0],sigma_color=x[1]) for x in xx]
    images.insert(0,raw)
    save_raw_2(
        images,
        whatfor = 'bilateral_filtering',
        titles = titles)

# # Alternative filtering:  Non-local means of denoising
#

# +
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr
from skimage.util import random_noise
# -

for i,file in enumerate(files[:1]):
    print(f'File "{file}" - {i+1}/{len(files)}')
    cur_path = os.path.join(initial_path,file)
    handheld_bscan = bscan_class.bscan(path = cur_path, debug = False)
    ## bilateral filtering
    raw = handheld_bscan.raw.T[top:bottom,left:right]
    sigmas = [30,90,200]
    titles = ['(d, sigma_col)='+str(x) for x in xx]
    titles.insert(0,'raw')
    images = [filtering(raw,mode='NLM',) for x in xx]
    images.insert(0,raw)
    save_raw_2(
        images,
        whatfor = 'bilateral_filtering',
        titles = titles)

# # Felzenszwalb segmentation

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# +
# img = img_as_float(astronaut()[::2, ::2])
immagine = d['Left Index_suction']['image']
img = img_as_float(a)

sigma = 5.5
segments_fz = felzenszwalb(img, 
                           scale=50, 
                           sigma=sigma, 
                           min_size=1000)
# -

segments_slic = slic(immagine, 
                     n_segments=250, 
                     compactness=0.01, 
                     sigma=sigma,
                     enforce_connectivity=True,
                     slic_zero=True)

# +
# segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
# -

# gradient = sobel(img)
input_watershed = img
segments_watershed = watershed(immagine,
                               markers=500, 
                               compactness=0.001,
                              )

# +
print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
# print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
ax[0, 0].imshow(segments_fz, cmap='magma')
# ax[0, 0].imshow(mark_boundaries(img, segments_fz, color=(1,0,0)))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(segments_slic, cmap='magma')
# ax[0, 1].imshow(mark_boundaries(img, segments_slic, color=(1,0,0)))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(img)
ax[1, 0].set_title('Raw Image')
# ax[1, 1].imshow(mark_boundaries(img, segments_watershed, color=(1,0,0)))
ax[1, 1].imshow(segments_watershed, cmap='magma')
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
# -


