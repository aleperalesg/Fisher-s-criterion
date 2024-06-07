""" This function returns kde from a given distribution
	Params: x is the given distribution
			h is the kernel bandwidth
"""

import numpy as np
import matplotlib.pyplot as plt

#*******************************************************

def gaussian_kernel(x):
	return np.exp(-x**2/2)/np.sqrt(np.pi*2)

#*******************************************************

def areawstep(y,step):
	area = 0
	for i in y:
		lengh = step
		width = i
		area += width*lengh
	return area

#*******************************************************

def area_no_step(x,y):

	num_samples = len(x)
	area = 0

	for i,xi in enumerate(x):

		if i < num_samples-1:
			lengh = x[i+1] - xi
			width = y[i]

			area += lengh*width
		else:
			area += 0

	return area

#*******************************************************

def kde(x,h,lim):
	
	
	n_sample = len(x)
	x_range = np.linspace(x.min() - lim, x.max() + lim, num = 600)
	y=0

	for i,xi in enumerate(x):
		y += gaussian_kernel((x_range - xi) / h)

	return  x_range, y/(h*n_sample)

#*******************************************************















































	

























