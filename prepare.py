import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def blur():
	img = plt.imread("img1.png")
	plt.imshow(img)
	plt.show()
	blurred = gaussian_filter(img,sigma=0.1)
	plt.imshow(blurred)
	plt.show()