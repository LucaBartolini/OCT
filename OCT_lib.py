import os

def files_in_folder(initdir = '.', boxtitle="Select OCT measurement"):
    '''
    Prompts the selection of a folder, and returns a list of the binary files in that location.
    A binary file is intended to have no extension
    '''
    from tkinter import filedialog
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    
    try:
        root.directory =  filedialog.askdirectory(title = boxtitle, initialdir = initdir)
    except FileNotFoundError:
        print('Something went wrong in the folder selection')
        
    print(f'Reading all binary files in folder: {root.directory}')
    
    files = os.listdir(root.directory)
    files = [f for f in files if (os.path.isfile(os.path.join(root.directory, f)) and not os.path.splitext(f)[1])]    
    print(f'Found {len(files)} files')
    
    root.destroy()
    return files

def get_level_set(img, **kwargs):
    '''
    Requires an image as first parameter.
    Returns a level-set, also called "mask", i.e. a matrix of booleans with
    the same shape as the `img` parameter.
    Three modes are allowed: "threshold", "checkerboard", and "otsu".
    `mode="threshold"` requires an additional parameter, the level above which a pixel becomes `True`
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
    Returns a filtered image
    
    Filtering mode, specifically "bilateral" or "NLM"
    
    '''
    mode = kwargs.get('mode','bilateral')
    
    if mode in ['None', None]:
        return img
    
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
            print('Warning, no parameter "sigma_est" passed for this type of filter. Automatic estimation takes computational time!')
            sigma_est = np.mean(estimate_sigma(img, multichannel=False))
            print(sigma_est)
        patch_kw = dict(
                patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False)
        return denoise_nl_means(img, h=12*sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    else:
        raise ValueError('Filtering Mode not understood. Pass mode="bilateral" or mode="NLM"')

def plot_raw (rawimgs: list):
    fig, ax = plt.subplots()
    assert type(rawimgs)==list, "Please provide raw images in a list. Even if it's one image only, put it within square brackets."   
    for rawimg in rawimgs:
        ax.imshow(rawimg)
    plt.show()

def plot_profiles (profiles: list):
    fig, ax = plt.subplots()
    assert type(profiles)==list, "Please provide profiles in a list. Even if it's one profile only, put it within square brackets."   
    for profile in profiles:
        ax.plot(profile)
    plt.show()
