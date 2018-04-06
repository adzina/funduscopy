import sys
import scipy
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy

def blur(img:numpy.ndarray, sig:float):
	blurred = gaussian_filter(img,sigma=sig)
	return blurred
	
def sharp(img:numpy.ndarray, alpha:float):
	img=img.astype(float)
	blurred = gaussian_filter(img, 3)
	filter_blurred = gaussian_filter(blurred, 1)
	sharpened = blurred + alpha * (blurred - filter_blurred)
	return sharpened