# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:44:39 2018

@author: lucab

Axsun Manual:
http://downloads.axsun.com/public/manuals/Axsun_EthernetPCIe_DAQ_User_Manual.pdf

"""

import numpy as np
import os
import ctypes  # to show popup windows in Microsoft Window
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import matplotlib.cm as cm #colormaps
# import skimage.transform

# TODO:
# check big- or little-endianness of the datatypes
# check that when bypass3 is used, the matrix stored in the binary file has to be split in two (go back to Nate Kemp's emails)
# later, define a C-scan class and an A-scan class
# C-scan will need __getitem__ __iter__ and the appropriate __repr__ and plot()

class bscan:

    def __init__(self,**kwargs):

        # later: "bckgnd" kwarg for subtraction and dispersion compensation
        if 'path' in kwargs:
            temp_path = kwargs.get('path')
            if os.path.isfile(temp_path): # when argument is an actual file
                self.path = temp_path
            elif os.path.isdir(temp_path): # if argument is a folder, choosefile dialog starts from there
                self.path = self.get_path(temp_path)
            else: #if no argument given
                raise Exception('The path specified is invalid or non-existent /LB')
        else:
            self.path = self.get_path()

		self.dirname = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)

        self.filesize = os.path.getsize(self.path) #in bytes
        self.debug = kwargs.get('debug', False)
        self.printwidth = kwargs.get('printwidth', 500)
        self.printheight = kwargs.get('printheight', 512)

        self.ascanlength = kwargs.get('ascanlength', 1024)

        self.bypass = kwargs.get('bypass',8)

        self.datatype_bypass = {
        1: np.dtype(np.uint16),
        2: np.dtype([('Im', '>i2'), ('Re', '>i2')]),
#         3: np.dtype([('Im', '<i2'), ('Re', '<i2')]),
        3: np.dtype('<i2'), # it will have to be reshaped though.
        4: np.dtype('<u4'),
        5: np.dtype('<u2'),
        6: np.dtype('<u2'),
        7: np.dtype('<u1'),
        8: np.dtype('>u1')
        }

        self.datatype = self.datatype_bypass[self.bypass]

		# check that remainder is zero! (in other words, make sure that width results in an int
        self.width = int(self.filesize/(self.ascanlength*self.datatype.itemsize))

        # reads from the file and returns a numpy array
        with cbook.get_sample_data(self.path) as dfile:
            self.raw = np.frombuffer(dfile.read(), dtype=self.datatype).reshape((self.width, self.ascanlength))
        print(f'Image shape = {self.raw.shape}')
        
        # self.raw = np.transpose(self.raw) # The OCT saves RAW data in a transposed image (rotated 90deg)

        print(f'"{self.filename}" loaded.\n - {self.width}x{self.ascanlength}px, bypass: {self.bypass}\n')

        # to get to bypass 7 and for imaging purposes
        self.gain_ref = 7 #actual gain as given by Axsun
        self.gain2 = self.gain_ref*2 #gain multiplied by 2, so that in bypassupdate(7), the factor "2" is already accounted for
        self.offset = -32

    def __len__(self):
	'''
	python requires __len__() to return an integer. In this case, without knowing if it's the best solution,
    len() has been defined to be the total number of pizels in the image. Could also be the filesize(bytes)
	'''
		return((self.width*self.ascanlength))
    
    def __call__(self):
	'''
	when an instance (name e.g. "thumb") of this class is created, calling thumb() will execute the lines below
	'''
        return self.raw

    def __getitem__(self,idx):
	'''
	To iterate over the Bscan, we need to access the A-scans.
	TODO self.raw might be stored in Transposed form, make sure that the correct one (between row or column) is returned by this function
	'''
        if ((idx<0) or (idx>self.width)):
            raise IndexError('The specified A-scan in not in the image. /LB')
        return(self.raw[idx,:])

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
		return {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
  
    def __repr__(self):
	'''
	Returns a string that will be printed out when the built-in "print" is called
	Blueprint: "[filename] - [width] x [height] px. Bypass=[bypassmode]"
	'''
        return f'(repr):"{self.filename}" - {self.width}x{self.ascanlength}px. Bypass={self.bypass}'
	
    def resize_img(self, w = int(self.width/2), h = self.printheight, new_bypass = 7):
	'''
	Returns a transformed image
	'''
		if self.debug: print(f'Creating image from "{self.filename}" bypass {self.bypass}, i.e. datatype: {self.datatype_bypass[self.bypass]}.')
        if new_bypass>self.bypass:
             self.update_bypass(bypass)
        return skimage.transform.resize(self.raw, (w, h), mode='reflect', preserve_range=True)


		
		
    def update_bypass(self,new_bypass=6):
        '''
        remember to cast to the appropriate data types!!
        '''

        if new_bypass <= self.bypass:
            print(f'Trying to update from bypass {self.bypass} to {new_bypass}')
            print('but bypass can only be increased.')
            return False

        if self.bypass<new_bypass and self.bypass<5:
            # bypassmode 3 is the lowest useful bypassmode in which data are stored
            # bring it to 5: -> Abs()
            if self.debug: print(f'Updating: from bypass: {self.bypass} to {self.bypass+2}: abs()')
            modulus_square_ref = lambda cmplx: (cmplx[0]*cmplx[0]+cmplx[1]*cmplx[1])
            modulus_square = np.vectorize(modulus_square_ref, otypes=[self.datatype_bypass[4]])
            # temp = np.zeros( (self.ascanlength, self.width), dtype=self.datatype_bypass[5])
            print('`modulus_square` defined')
            temp = np.zeros( len(self.raw), dtype=self.datatype_bypass[4])
            print('`temp` matrix created')
            temp = modulus_square(self.raw)
            print('taken `modulus_square(self.raw)`')
            # temp.shape = (self.width, self.ascanlength)
            print('reshaped temp - COMMENTED out for now')
            self.raw = temp
            print('Assigned temp to self.raw')
            self.bypass = 4

        if self.bypass<new_bypass and self.bypass<6:
            #bring it to 6: -> Log Compressed
            if self.debug: print(f'Updating: from bypass: {self.bypass} to {self.bypass+1}: taking log10')
            # temp = np.zeros( (self.ascanlength, self.width), dtype=self.datatype_bypass[6])
            # temp = 20 * np.log10(self.raw)
            # logarithm10 = np.vectorize(np.log10, otypes=[self.datatype_bypass[6]])
            #self.raw = logarithm10(self.raw)
            self.raw = np.log10(self.raw).astype(self.datatype_bypass[6])
            self.bypass = 6

        if self.bypass<new_bypass and self.bypass<7:
            #bring it to 7: -> Dynamic Range Reduced
            if self.debug: print(f'Updating: from bypass: {self.bypass} to {self.bypass+1}: reducing dynamic range (log2)')
            # temp = np.zeros( (self.ascanlength, self.width), dtype=self.datatype_bypass[7])
            # temp = self.gain * 2 *np.log2(self.raw) + self.offset

            def reduce_dr_ref(x_i):
                return np.add(self.offset, np.multiply(self.gain2,np.log2(x_i) ) )
            reduce_dr = np.vectorize(reduce_dr_ref, otypes=[self.datatype_bypass[7]])

            self.raw = reduce_dr(self.raw)

            self.bypass = 7

        if self.bypass<new_bypass and self.bypass<8:
            #bring it to 8: -> JPEG compressed
            if self.debug: print(f'Updating: from bypass: {self.bypass} to {self.bypass+1}')
            self.bypass = 8
            pass

        self.datatype = self.datatype_bypass[self.bypass]
        #return self.raw
        # return self

    def mod1(self):
        # print ("uses vectorialized func")
        def modulus_ref(cmplx):
            return np.sqrt(np.add(np.square(cmplx[0]),np.square(cmplx[1])))
        modulus = np.vectorize(modulus_ref, otypes=[self.datatype_bypass[5]])
        # return mod_square(self.raw)
        # self.raw = modulus(self.raw)
        self.bypass = 4

    # def mod2(self):
    #     # print("Uses lambda")
    #     temp = np.zeros( (self.ascanlength, self.width), dtype=self.datatype_bypass[4])
    #     modulus_squared = lambda cmplx: np.sqrt(np.square(cmplx[0])+np.square(cmplx[1]))
    #     temp = map(modulus_squared,self.raw)
    #     return temp
    #     self.bypass = 4

    def mod3(self):
        modulus_square_ref = lambda cmplx: (cmplx[0]*cmplx[0]+cmplx[1]*cmplx[1])**0.5
        modulus_square = np.vectorize(modulus_square_ref)
        temp = np.zeros( (self.ascanlength, self.width), dtype=self.datatype_bypass[5])
        temp = modulus_square(self.raw)
        return temp
        self.bypass = 4		

    def this(self):
        # from bypassmode 3 to bypassmode 6
        # consider: https://stackoverflow.com/questions/25870923/how-to-square-or-raise-to-a-power-elementwise-a-2d-numpy-array
        idx_split = int(self.width/2)
        res = np.zeros((idx_split, self.ascanlength), dtype=self.datatype_bypass[5])
        def byp_6_ref(Im_el,Re_el): 
            return np.multiply(np.log10(np.sqrt(np.add(Im_el*Im_el, Re_el*Re_el))), 20)
        byp_6 = np.vectorize(byp_6_ref, otypes=[self.datatype_bypass[5]])
        res = byp_6(self.raw[:idx_split], self.raw[idx_split:])
        self.raw = res.astype(self.datatype_bypass[6])
        self.width = self.raw.shape[0]
        self.bypass = 6
        
    def that(self):
        '''from bypassmode 3 to bypassmode 6
        '''
        idx_split = int(self.raw.shape[0]/2)
        def byp_6(Im_mat,Re_mat): 
            return np.multiply(
                            np.log10(
                                np.sqrt(
                                    np.add(
                                        np.power(Im_mat, 2, dtype = self.datatype_bypass[4]), 
                                        np.power(Re_mat, 2, dtype = self.datatype_bypass[4]),
                                        dtype = self.datatype_bypass[4] ),
                                dtype = self.datatype_bypass[5] ),
                            dtype = self.datatype_bypass[6] ), 
                        20)
        res = byp_6(self.raw[:idx_split], self.raw[idx_split:])
        self.raw = res.astype(self.datatype_bypass[6])
        self.width = self.raw.shape[0]
        self.bypass = 6

		
		
		
		
    @staticmethod
    def get_path(initdir = 'C:\\Users\\user\\Google Drive\\OCT\\', boxtitle="Select OCT measurement"):
        from tkinter import filedialog
        import tkinter

        root = tkinter.Tk()
        root.filename =  filedialog.askopenfilename(initialdir = initdir,title = boxtitle, filetypes = (("all files","*.*"),("bin files","*.bin")))
        root.withdraw()

        if root.filename == '':
            raise ValueError('File invalid or not specified. /LB ')

        return root.filename

    @staticmethod
    def MessageBox(title, text, style):
        return ctypes.windll.user32.MessageBoxW(0, text, title, style)


    #
    pass
    #

# print(a)
# a.__repr__()
# a.__str__()
# print(a.ascanlength)
# print(a.width)
# len(a)
# a.filesize
# print(a)
# np.shape(a.raw)
# a.raw[1,:]
