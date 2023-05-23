import io
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from skimage.measure import block_reduce

class ImageFilter(object):
    
    def __init__(self, kernel=None):
        if kernel is not None:
            self.imgKernel = kernel
    
    def convolve(self, imgTensor):
        imgTensorRGB = imgTensor.copy() 
        outputImgRGB = np.empty_like(imgTensorRGB)

        for dim in range(imgTensorRGB.shape[-1]):  # loop over rgb channels
            outputImgRGB[:, :, dim] = sp.signal.convolve2d (
                imgTensorRGB[:, :, dim], self.imgKernel, mode="same", boundary="symm"
            )

        return outputImgRGB

    def downSample(self, img):
        return block_reduce(image=img, block_size=(2,2,1), func=np.max)
