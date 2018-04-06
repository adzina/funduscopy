import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy

def blur(img:numpy.ndarray,sig:float):
	blurred = gaussian_filter(img,sigma=sig)
	return blurred