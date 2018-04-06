import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from scipy import misc,ndimage

def norm(img):
	img = img.convert(mode="RGB")
	img_norm = ImageOps.equalize(img)
	return img_norm
	
def blur(img:np.ndarray, sig:float):
	
	blurred = img.filter(ImageFilter.GaussianBlur(sig))
	return blurred
	
def sharp(img:np.ndarray):
	
	sharpened = img.filter(ImageFilter.SHARPEN)
	return sharpened
