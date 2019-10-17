# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:44:39 2018

@author: lucab

Axsun Manual:
http://downloads.axsun.com/public/manuals/Axsun_EthernetPCIe_DAQ_User_Manual.pdf

"""

import bz2  # to zip the pickle file
import ctypes  # to show popup windows in Microsoft Window
import os
import pickle
import tkinter
from tkinter import filedialog

import cv2
import matplotlib.cbook as cbook
import matplotlib.cm as cm  # colormaps
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from PIL import Image
from scipy.signal import savgol_filter
from scipy.stats import skew
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import morphological_chan_vese

# TODO:
# check that when bypass3 is used, the matrix stored in the binary file has to be split in two (go back to Nate Kemp's emails)

class bscan:
    '''
    A class to read and manipulate B-scans from Axsun OCT.
    '''

    def __init__(self, **kwargs):
        self.path = kwargs.get('path', None)

        if self.path is None:
            print('\nSelect OCT file')
            self.path = get_path()
        elif os.path.isdir(self.path):
            self.path = get_path(self.path)

        # OCT specifications: the height of a px (vertical resolution) is 5.75um
        self.px_height = kwargs.get('px_height', default_settings['OCT']['axial_res'])
        self.dirname = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)

        self.filesize = os.path.getsize(self.path)  # in bytes
        self.debug = kwargs.get('debug', False)
        self.ascanlength = kwargs.get('ascanlength', 1024)

        if 'bypass' not in kwargs:
            print('Bypass not specified, will read the binary in chunks of 1 byte')
            kwargs['bypass'] = 8
        self.bypass = kwargs.get('bypass')

        self.datatype = datatype_bypass[self.bypass]

        # check that remainder is zero! (in other words, make sure that width results in an int
        self.width = int(
            self.filesize/(self.ascanlength*self.datatype.itemsize))

        # reads from the binary file and returns a the RAW data as numpy array
        with cbook.get_sample_data(self.path) as dfile:
            self.raw = np.frombuffer(dfile.read(), dtype=self.datatype).reshape(
                (self.width, self.ascanlength)).T

        print(
            f'"{self.filename}" loaded.\n - {self.ascanlength}x{self.width}px, bypass: {self.bypass}\n')

        # to get to bypass 7 and for imaging purposes
        self.gain_ref = 7  # actual gain as given by Axsun
        # gain multiplied by 2, so that in bypassupdate(7), the factor "2" is already accounted for
        self.gain2 = self.gain_ref*2
        self.offset = -32
        
        

    def __len__(self):
        '''
        python requires __len__() to return an integer. In this case, without knowing if it's the best solution,
        len() has been defined to be the total number of pizels in the image. Could also be the filesize(bytes)
        '''
        return(self.width)

    def __call__(self):
        '''
        when an instance (name e.g. "thumb") of this class is created, calling thumb() will execute the lines below
        '''
        return self.raw

    def __getitem__(self, idx):
        '''
        To iterate over the Bscan, we need to access the A-scans.
        TODO self.raw might be stored in Transposed form, make sure that the correct one (between row or column) is returned by this function
        '''
        if ((idx < 0) or (idx > self.width)):
            raise IndexError('The specified A-scan in not in the image. /LB')
        return(self.raw[idx, :])

    # def __iter__(self):
    #     return self

    # def __add__
    # should be an "addition" which is actually an averaging!. Two images can be added, and their average returned
    # should depend on the bypass

    # def __sub__
    # for background subtraction

    def properties(self):
        '''
        returns the dictionary of the "bscan" class attributes (props is short for properties)
        NOTE: theres a built-in function for this: it's: ```vars(self)```
        '''

        return {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

    def __repr__(self):
        '''
        Returns a string that will be printed out when the built-in "print" is called
        Blueprint: "[filename] - [width] x [height] px. Bypass=[bypassmode]"
        '''
        return f'File "{self.filename}": {self.width}x{self.ascanlength}px'

    def resize_raw(self, w=None, h=None):
        """
        Resizes self.raw

        Parameters
        ----------
        w : [int], optional
            [New width] - by default, same as self.raw
        h : [int], optional
            [New heigth] - by default, same as self.raw
        """
        if w is None:
            w = self.width
        if h is None:
            h = self.ascanlength
        else:  # if h changes, we need to update the vertical resolution
            h_ratio = h/self.ascanlength
            self.px_height = self.px_height/h_ratio
        
        w = int(w)
        h = int(h)
        
        self.raw = cv2.resize(self.raw, (w, h),
                              interpolation=cv2.INTER_CUBIC)
        # print(f"self.raw has a new shape: {self.raw.shape}")
        
        # to resize with an integer factor
        # self.raw = cv2.resize(self.raw , None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

        self._update_size()
        return True

    def crop(self, top=None, bottom=None, left=None, right=None):
        """
        Crops self.raw

        Parameters
        ----------
        top : [int], optional
            [Index of the top row (included) to crop at], by default None
        bottom : [int], optional
            [Index of the bottom row (included) to crop at], by default None
        left : [int], optional
            [Index of the left column (included) to crop at], by default None
        right : [int], optional
            [Index of the right column (included) to crop at], by default None
        """
        # assert that params are integer
        self.raw = self.raw[top:bottom, left:right]
        self._update_size()
        
        return self.raw

    def apply_filter(self, **kwargs):
        """
        Applies a custom filter to the bscan, as specified by `mode`
        Depending on the mode, additional parameters are available
        
        Parameters
        ----------
        mode : [str], optional, by default None
            `None`: returns the bscan as it is
            `bilateral`: applies cv2.bilateral_filter
            `NLM`: applies NON LINEAR MEANS denoising            
        """
        mode = kwargs.get('mode')
        if mode in ['None', None]:
            return self.raw
        elif mode not in ['bilateral', 'NLM']:
            raise ValueError(
                'Filtering Mode not understood. Pass mode="bilateral" or mode="NLM"')

        elif mode in ['bilateral']:
            d = kwargs.get('d', 5)
            sigma_color = kwargs.get('sigma_color', 30)
            sigma_space = kwargs.get('sigma_space', d*3)
            self.raw = cv2.bilateralFilter(
                self.raw, d, sigma_color, sigma_space)
        elif mode in ['NLM']:
            sigma_est = kwargs.get('sigma', None)
            if sigma_est is None:
                print('Warning, no parameter "sigma_est" passed for this type of filter. Automatic estimation takes computational time!')
                sigma_est = np.mean(estimate_sigma(
                    self.raw, multichannel=False))
                print(sigma_est)
            patch_kw = {
                'patch_size': 5,      # 5x5 patches
                'patch_distance': 6,  # 13x13 search area
                'multichannel': False}
            self.raw = denoise_nl_means(
                self.raw, h=12*sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
        

        
        return self.raw

    def get_level_set(self, mode, **kwargs):
        """
        Calculates a Level-set, also known as "mask", i.e. a matrix of booleans with the same shape 
        as the given `img` 
        Three modes are allowed: "threshold", "otsu", "checkerboard".
        `mode="threshold"` requires an additional parameter, the level above which a pixel becomes `True`

        Parameters
        ----------
        'mode': [str]
            'threshold' - requires also 'lvl'=int, the returned image has 1 for all pixels above 'lvl' and 0 in the others 
            'otsu' - just like threshold, but the 'lvl' is calculated automatically based on the skimage.filters.threshold_otsu(img)
            'checkerboard' - uses skimage.segmentation.checkerboard_level_set(...) 
            'chan_vese' - 

        Returns
        -------
        [ndarray]
            The level set
        """
        if mode in ['threshold']:
            lvl = kwargs.get('lvl')
            return self.raw > lvl * 255
        elif mode in ['checkerboard', 'checkers']:
            return checkerboard_level_set(self.raw.shape, lvl)
        elif mode in ['otsu']:
            return self.raw > threshold_otsu(self.raw)
        elif mode in ['chan_vese', 'chanvese']:
            N_iter = kwargs.get('N_iter', 30)
            sm = kwargs.get('smoothing', 4)
            # guess the initial level set with otsu
            thr = threshold_otsu(self.raw)
            init_ls = self.raw > thr
            return morphological_chan_vese(self.raw, iterations=N_iter, init_level_set=init_ls, smoothing=sm)

    def get_profile(self, levelset=None, exclude_top_px=150, **kwargs):
        """
        Returns the height of the surface along the Bscan

        Parameters
        ----------
        levelset : [numpy array], optional
            the segmented image (i.e. levelset or mask) from which the profile height is calculated,
            by default - if not give, segments self.raw using the Chan Vese ACME
        mode: ['chan_vese', 'otsu']
            if `levelset` is not provided, than a `mode` to calculate the level_set has to be specified  
        exclude_topN [int]:

        Returns
        -------
        profile : 1D [ndarray]
            The height (in px) of the profile.
        """
        assert (type(exclude_top_px) == int), \
            "`exclude_top_px` has to be integer"

        if levelset is None:
            mode = kwargs.get('mode')
            assert (
                mode is not None), "If no levelset is provided, a mode to obtain the level_set (`otsu` or `chan_vese`) has to be specified!"
            if mode == 'chan_vese':
                levelset = self.get_level_set(mode='chan_vese')
            elif mode == 'otsu':
                levelset = self.get_level_set(mode='otsu')

        profile = np.zeros((len(levelset.T), 1))
        for i, Ascan in enumerate(levelset.T):
            try:
                # Artifacts (or spurious) reflections in the upper rows of the image, there is the recognition of spurious surfaces,
                # that messes up the profile detection
                # rows from the top that are excluded from the profile search.
                # index of the first non-zero element (considering also the excluded_px)
                profile[i] = exclude_top_px + \
                    np.nonzero(Ascan[exclude_top_px:] > 0.5)[0][0]
            except Exception as e:
                print(f"Some error happened: {e}. \nMoving on...")
                profile[i] = None
                
        return profile

    def save_png(self, img=None, profile=None, levelset=None, name=None):
        """This function creates a .png file and stores it on the hard-disk
        
        Parameters
        ----------
        img : [type], optional
            [description], by default self.raw
        profile : [type], optional
            [description], by default None
        levelset : [type], optional
            [description], by default None
        name : [type], optional
            [description], by default None
        """
    
        if name is None:
            fname = os.path.join(self.dirname, 'processed', self.filename)
        else:
            fname = os.path.join(self.dirname, 'processed', name)
        fname += '.png'

        assert ((profile is None) or (levelset is None)), \
            "Please provide one only among `contour` or `levelset`"
        plt.ioff()

        if img is None:
            img = self.raw

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray', 
                  extent=[0, self.width, 0, self.ascanlength])
        ax.axis('off')
        
        # print(f"width = {self.width}, heigth = {self.ascanlength}")

        if levelset is not None:
            ax.contour(levelset, [0.5], colors='r', alpha=0.5)
        elif profile is not None:
            ax.plot((self.ascanlength-profile), color='red', alpha=0.5)
        
        if self.debug:
            print(f"Saving file at: \n{fname}")

        fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

    def _update_size(self):
        """
        Called internally to update the parameters `self.width` and `self.ascanlength` when a function modifies them
        """
        self.ascanlength, self.width = self.raw.shape
        assert (self.ascanlength > 0 and self.width > 0), \
            "self.raw is being updated to an array with at least a dimension == 0"

# Class definition ends here


def smooth_profile(profile, rel_window_size = 0.6, poly_degree=3):
    """[Smoothes a profile by applying a Savitkzy Golay filter]
    
    Wrapper for
    scipy.signal.savgol_filter()

    Parameters
    ----------
    profile : [nd.array]
        [The profile that will be smoothed]
    rel_window_size : float, optional
        [This fraction of len(profile) will be the window_size parameter in the Savitky Golay filter], by default 0.5
    poly_degree : int, optional
        [Same as in Savitzky Golay filter], by default 3
    
    Returns
    -------
    [Smoothed profile]
    """
    
    window_size = int(len(profile)*rel_window_size)  # big window size
    if window_size%2==0: # make sure window size is odd (required by `sdvgol_filter`)
        window_size+=1        

    return savgol_filter(profile, window_size, 3) # smoothed profile

def center_profile(x_range, profile):
    """[Centers `x_range` so that the peak of `profile` is at x=0]
    
    Parameters
    ----------
    x_range : [type]
        [description]
    profile : [type]
        [description]
    
    Returns
    -------
    x_range [type]
        [Centered x_range. The peak of `profile` is now found at x=0]
    """
    assert len(profile)==len(x_range), "The profile and the x_range must have the same length"
    
    profile_peak = max(profile)
    peak_idx  = int(np.argwhere(profile == profile_peak)[0])

    x_peak = x_range[np.argwhere(profile==profile_peak)[0]]
    # shift the x axis so that the profile peak is at x = 0
    x_range = x_range - x_peak
    return x_range

def asymmetry_factor(profile, peak_cutoff = 0.2):
    
    """Evaluates the asymmetry of a given profile, returning a value in the range [0,1] 
    
    The profile is a f(x). We can decompose f(x) in a symmetric and an antisymmetric component:
    f(x) = ( f_plus(x) + f_minus(x) )
    where the symmetric component is:
    f_plus(x) = f(x) + f(-x)
    and the asymmetric component is:
    f_minus(x)= f(x) - f(-x)
    
    take the norm of those two vectors:
    asymm_factor = norm(f_minus) / ( norm(f_plus)+norm(f_minus) )

    Parameters
    ----------
    profile : np.array or list
        The profile of which asymmetry is calculated

    Returns
    -------
    asymmetry_factor: float
    """

    profile_peak = max(profile)
    peak_idx  = int(np.argwhere(profile == profile_peak)[0])
    # first index above cutoff
    left_idx  = int(np.argwhere(profile >  profile_peak*peak_cutoff)[0]) 
    # last index above cutoff
    right_idx = int(np.argwhere(profile >  profile_peak*peak_cutoff)[-1]) 
    
    if ((right_idx == len(profile)) or (left_idx==0)):
        print("The cutoff might be too low to get both sides of the profile")
        try:
            print(f"Height diff {(profile[left_idx-1]-profile[right_idx+1])*100/profile_peak:.3}")
        except IndexError:
            print(f"Cutoff too low! Increase the value of `peak_cutoff` (currently {peak_cutoff})")

    f_left_norm  = np.sqrt(np.sum((profile[left_idx:peak_idx])**2))
    f_right_norm = np.sqrt(np.sum((profile[peak_idx+1:right_idx+1])**2)) #+1 to exclude the peak
    
    # asymmetry factor will be calculated the portion of the profile above the peak_cutoff
    f = profile[left_idx:right_idx]

    f_left_norm_2  = np.sqrt( np.sum(f[:int(len(f)//2)] )**2)
    f_right_norm_2 = np.sqrt( np.sum(f[int(len(f)//2):] )**2) #+1 to exclude the peak

    f_plus  = 0.5 * (f + np.flip(f))
    f_minus = 0.5 * (f - np.flip(f))
    f_plus_norm = (sum(f_plus**2)**0.5)
    f_minus_norm = (sum(f_minus**2)**0.5)
    
    # asymmetry factor
    af1 = f_minus_norm/(f_plus_norm+f_minus_norm)
    af2 = (f_right_norm - f_left_norm)/(f_right_norm+f_left_norm)
    af3 = (f_right_norm_2 - f_left_norm_2)/(f_right_norm_2+f_left_norm_2)

    return [af1, af2, af3]


def MessageBox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


def get_files(initdir='.', directory=None, ext=''):
    """
    Prompts the selection of a folder, and returns a list of the files in that location.

    Parameters
    ----------
    initdir : str, optional
        [Initial directory], by default: current directory, also accessible with `os.getcwd()`
    directory : str, optional
        [Directory], by default None. If this is provided, the method directly returns the list of files 
        (their extension still specified by `ext`)
    ext : str, optional
        [Select all files with that extension], by default '', which means no extension, i.e. binary files

    Returns
    -------
    [files, root_directory]
        [0]: [List of files with the specified ext. By default, list of binary files]
        [1]: root directory selected
    """

    if directory is None:
        from tkinter import filedialog
        import tkinter

        boxtitle = "Select OCT measurement"
        root = tkinter.Tk()
        root.withdraw()

        try:
            root.call('wm', 'attributes', '.', '-topmost', True)
            root.directory = filedialog.askdirectory(
                title=boxtitle, initialdir=initdir)
        except FileNotFoundError:
            print('Something went wrong in the folder selection')
        root.update()
        print(f'Reading all binary files in folder: {root.directory}')
        files = os.listdir(root.directory)
        files = [f for f in files if (os.path.isfile(os.path.join(
            root.directory, f)) and os.path.splitext(f)[1]) == ext and f[0]!='.']
        print(f'Found {len(files)} files \n\n')
        return [files, root.directory]
    else:
        assert os.path.isdir(
            directory), "The path specified is not a valid directory: check the path given is correct!"
        files = os.listdir(directory)
        files = [f for f in files if (os.path.isfile(os.path.join(
            directory, f)) and os.path.splitext(f)[1]) == ext and f[0]!='.']
        print(f'Found {len(files)} files \n\n')
        return [files, directory]

def get_file(initdir='.', ext=''):
    """
    GUI which prompts the selection of a single file, and returns its path.

    Parameters
    ----------
    initdir : str, optional
        [Initial directory], by default: current directory, also accessible with `os.getcwd()`
    ext : str, optional
        [Show only files with that extension], by default ''

    Returns
    -------
    filepath :: str
    """

    boxtitle = "Select file"
    root = tkinter.Tk()
    root.withdraw()
    root.focus_force()
    try:
        root.filename =  filedialog.askopenfilename(initialdir = initdir,title = "Select file",filetypes=[("bz2 Files", ".bz2")]) 
        root.call('wm', 'attributes', '.', '-topmost', True)
    except FileNotFoundError:
        print('Something went wrong in the folder selection')
    root.update()
    path2file = root.filename
    root.destroy()
    return path2file

def get_results(initialdir = None, ext = 'bz2'):
    if initialdir is None:
        initialdir = os.getcwd()
    pckl = get_file(initdir = initialdir, ext=ext)
    basepath = os.path.dirname(pckl)
    with bz2.BZ2File(pckl, 'r') as sfile:
        data = pickle.loads(sfile.read())
    # if the last folder is 'processed', remove it from the basepath which will be returned    
    if os.path.split(basepath)[-1]=='processed':
        basepath = os.path.split(basepath)[0]
    print(f"Dataset: \n{basepath}\nsuccessfully read")

    return data, basepath



def plot_raw(rawimgs: list):
    fig, ax = plt.subplots()
    assert hasattr(
        rawimgs, '__iter__'), "Please provide raw images in a list. Even if it's one image only, put it within square brackets."
    for rawimg in rawimgs:
        if type(rawimg) == bscan:
            rawimg = rawimg.raw
        ax.imshow(rawimg, cmap='gray')
    plt.show()


def plot_profiles(profiles: list):
    fig, ax = plt.subplots()
    assert hasattr(
        profiles, '__iter__'), "Please provide profiles in a list. Even if it's one profile only, put it within square brackets."
    for profile in profiles:
        ax.plot(profile, alpha=0.65)
    plt.show()

def roll_list(lst, N):
    """[Rolls a list by N elements]
    example: 
    my_list = [0,1,2,3,4,5]
    roll_list(my_list, 2) = [2,3,4,5,0,1]
    
    Parameters
    ----------
    lst : [list]
        the list to roll
    N : [int]
        the number of elements to roll around
    
    NOTE: lst is not deep-copied. Any change on this internal variable will be reflected
    onto the variable in the parent scope
    """

    for i in range(N):
        lst.append(lst.pop(0))
    pass
    
def compare_dictionaries(dict_1, dict_2, dict_1_name, dict_2_name, path=""):
    """Compare two dictionaries recursively to find non matching elements
    copied from:
    [https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries]
    
    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2

    Returns:

    """
    err = ''
    key_err = ''
    value_err = ''
    old_path = path
    for k in dict_1.keys():
        path = old_path + "[%s]" % k
        if k not in dict_2:
            key_err += "Key %s%s not in %s\n" % (dict_2_name, path, dict_2_name)
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dictionaries(dict_1[k],dict_2[k],dict_1_name,dict_2_name, path)
            else:
                if dict_1[k] != dict_2[k]:
                    value_err += "%s%s=%s DIFFERENT FROM %s%s(=%s)\n"\
                        % (dict_1_name, path, dict_1[k], dict_2_name, path, dict_2[k])

    for k in dict_2.keys():
        path = old_path + "[%s]" % k
        if k not in dict_1:
            key_err += "Key %s%s not in %s\n" % (dict_2_name, path, dict_1_name)

    return key_err + value_err + err
    
default_settings = dict(
        ## if crop['switch'] is True, the image gets cropped at the specified indices 
        crop = dict(
            # turn the crop switch ON or OFF
            switch = False,
            ## cropping indices
            left = None,
            right = None,
            top = None,
            bottom = None,
        ),
        ## if resize['switch'] is True, the image gets resized with the specified factors 
        resize = dict(
            switch = False,
            width_factor = 1,
            height_factor = 1,
        ),
        ## filter settings to set all the parameters of filtering modes 
        filtering = dict(
                switch = True,
                mode = 'bilateral',
                bilateral = dict(
                    d = 15,
                    sigma_color = 90,
                    sigma_space = 80,
                ),
                NLM = dict(
                    sigma = 6, #parameter for NLM denoising
                ),
            ),
        ## Parameters of segmentation
        segmentation = dict(
            switch = True, 
            mode = "otsu", # or  "chan_vese"
            ignore_top_px = 150,
            N_iter = 20, # used in ACWE segm. (see docs of `morphological snakes`)
        ),
        ## Parameters of Axsun OCT
        OCT = dict(
            axial_res = 5.75, # Vertical Resolution of OCT, in micron per pixel
            aperture_size = 4, # horizontal dimension of the Bscan (i.e. size of the aperture)
            bypassmode = 8, # (AXSUN property) this translates to #bytes/px (defaul 1b/px): See manual
        ),           
        save_png = True, # saves B-scans in .png format in folder \processed\
        make_res = True, # stores processed results in a dictionary for immediate use
        save_zip = True, # saves the dictionary of processed results in a pickled .bz2 
        save_settings = True, # saves "analysis settings" in a JSON file
    )

def print_about_settings(p):
    a = compare_dictionaries(p, default_settings, 'p', 'Default_Settings')
    if a:
        print("===================\nFollowing settings are non-default:\n")
        print(a)    
        print("===================")

    onoff = lambda flag: "ON" if flag else "OFF"
    out_str = f'Analysis settings: ' \
     '\n==================='\
    f'\n    Cropping is {onoff(p["crop"]["switch"])}'\
    f'\n    Resizing is {onoff(p["resize"]["switch"])}'\
    f'\n   Filtering is {onoff(p["filtering"]["mode"])}'\
    f'\nSegmentation is {onoff(p["segmentation"]["switch"])}'\
    f'\n .png making is {onoff(p["save_png"])}'\
    f'\n    Savefile is {onoff(p["save_zip"])}'

    return out_str



# the selection `default_setttings['OCT']['bypassmode']` is the key for the following dictionary
datatype_bypass = {
            # reading a file as dataype 3 (which is a complex number shaper as (Im(px),Re(px)):
            # self.RAW needs splitting: LabView saves an image made by 2 juxtaposed images of Im and Re
            3: np.dtype('i2'),
            4: np.dtype('u4'),
            5: np.dtype('u2'),
            6: np.dtype('u2'),
            7: np.dtype('u1'),
            8: np.dtype('u1'),
        }


if __name__ == '__main__':
    print('Compiled successfully')
