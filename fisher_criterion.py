""" 
This code show how LDA (2d and 3d) works and its aim is data visualization.
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from numpy.linalg import inv
import kde
import math

np.random.seed(10)
## projection function for visual representation
def projection2d(c,w,n1):

	pjt = np.zeros((n1,2))
	norma = np.sqrt(sum(w**2)) 

	for i in range(n1):
		pjt[i,:]  = ((np.dot(c[i,:],w)) / ( norma**2 )) * w

	return pjt 

## Set up dataset
dim = 2   # dimensionality of dataset
n1 = 30  # number of elements of class 1
n2 = 30  # number of elements of class 2



## get classes from normal distribution
c1x, c1y = random.multivariate_normal([9,6], [[0.04,0.9],[0.6,2]], n1).T
c2x, c2y = random.multivariate_normal([6,8], [[0.04,0.9],[0.6,2]],n2).T
c1 = np.column_stack((c1x,c1y))
c2 = np.column_stack((c2x,c2y))
dataset = np.concatenate((c1,c2), axis = 0);



## mean of classes
mn1 = np.mean(c1, axis = 0)
mn2 = np.mean(c2, axis = 0)

SB = mn1 - mn2  # mean difference
S1 = np.cov(np.transpose(c1)) # covariance matrix of class 1
S2 = np.cov(np.transpose(c2)) # covariance matrix of class 1

# get SW matrix
SW = (S1*(n1-1) + S2*(n1-1)) / ((n1+n2) - 2) 
SWi = inv(SW)

# Optimal vector
w = np.dot(SWi,SB.T)


# discriminant function
g1 = np.dot((c1 - 0.5*(mn1+mn2)), w.reshape(dim,1))
g2 = np.dot((c2 - 0.5*(mn1+mn2)), w.reshape(dim,1))

# Array predict label,     1 -> class 1 | -1 -> class 2
pl = np.sign(np.dot((dataset - 0.5*(mn1+mn2)), w.reshape(dim,1))) 


# kde of classifcation scores
h = 5    # kernel bandwidth
lim = 5  
x1_kde, y1_kde  = kde.kde(g1,h,lim)
x2_kde, y2_kde  = kde.kde(g2,h,lim)

wes = w*0.4  ## scaled vector w for visual representation 


if dim == 2:
	##getting orthogonal projections for visual representation:
	pjtc1 = projection2d(c1,wes,n1)
	pjtc2 = projection2d(c2,wes,n2)

	### non optimal projection for visual representation:
	nwes = np.array([1,13])
	npjtc1 = projection2d(c1,nwes,n1)
	npjtc2 = projection2d(c2,nwes,n2)


	fig, (plt1, plt2, plt3,plt4) = plt.subplots(1, 4)
	fig.suptitle('Fisher\'s criterion', fontsize=16)
	
	plt1.plot(c1[:,0],c1[:,1],'o',color='salmon')
	plt1.plot(c2[:,0],c2[:,1],'o',color='skyblue')
	plt1.plot([],[],[],color='black')

	plt1.plot(pjtc1[:,0],pjtc1[:,1],'o',color='salmon')
	plt1.plot(pjtc2[:,0],pjtc2[:,1],'o',color='skyblue')

	for i in range(n1):
		x = np.linspace(c1[i,0],pjtc1[i,0],10)
		y = np.linspace(c1[i,1],pjtc1[i,1],10)

		plt1.plot(x,y,'--',color='gray')


	for i in range(n2):
		x = np.linspace(c2[i,0],pjtc2[i,0],10)
		y = np.linspace(c2[i,1],pjtc2[i,1],10)

		plt1.plot(x,y,'--',color='gray')



	start = [0,0]
	#plt.quiver(start[0],start[0],w[0],w[1],color = 'black',scale_units='xy', scale=1)
	plt1.arrow(0, 0, wes[0],wes[1] , width = 0.05,color = "black")
	#plt.quiver(start[0],start[0],-w[0],-w[1],color = 'black',scale_units='xy', scale=1)
	plt1.axhline(y = 0, color = 'black', linestyle = '-') 
	plt1.axvline(x = 0, color = 'black', linestyle = '-')
	#plt1.grid()
	plt.axis([-1, 14 , -4, 14])
	plt1.set_xlabel("x1")
	plt1.set_ylabel("x2")
	#plt1.title(" Fisher's criterion")
	plt1.legend(['Class 1','Class 2','Optimal vector'])


	plt2.plot(c1[:,0],c1[:,1],'o',color='salmon')
	plt2.plot(c2[:,0],c2[:,1],'o',color='skyblue')
	plt2.plot([],[],[],color='black')
	plt2.plot(npjtc1[:,0],npjtc1[:,1],'o',color='salmon')
	plt2.plot(npjtc2[:,0],npjtc2[:,1],'o',color='skyblue')

	for i in range(n1):
		x = np.linspace(c1[i,0],npjtc1[i,0],10)
		y = np.linspace(c1[i,1],npjtc1[i,1],10)

		plt2.plot(x,y,'--',color='gray')


	for i in range(n2):
		x = np.linspace(c2[i,0],npjtc2[i,0],10)
		y = np.linspace(c2[i,1],npjtc2[i,1],10)

		plt2.plot(x,y,'--',color='gray')


	start = [0,0]
	#plt.quiver(start[0],start[0],w[0],w[1],color = 'black',scale_units='xy', scale=1)
	plt2.arrow(0, 0, nwes[0],nwes[1] , width = 0.05,color = "black")
	#plt.quiver(start[0],start[0],-w[0],-w[1],color = 'black',scale_units='xy', scale=1)
	plt2.axhline(y = 0, color = 'black', linestyle = '-') 
	plt2.axvline(x = 0, color = 'black', linestyle = '-')
	#plt2.grid()
	#plt2.ylim(10, 30)
	plt.axis([-1, 14 , -4, 14])
	plt2.set_xlabel("x1")
	plt2.set_ylabel("x2")
	#plt1.title(" Fisher's criterion")
	plt2.legend(['Class 1','Class 2','Non optimal vector'])




	plt3.plot(c1x, c1y,'o',c = 'salmon')
	plt3.plot(c2x, c2y,'o',c = 'skyblue')
	plt3.grid()
	plt3.set_xlabel("x1")
	plt3.set_ylabel("x2")

	plt4.set_xlim([-65, 65])
	plt4.set_ylim([0, 0.055])
	plt4.plot(x1_kde, y1_kde, c = 'salmon')
	plt4.plot(x2_kde, y2_kde, c = 'skyblue')
	plt4.grid()
	plt4.set_xlabel("LDA score")
	plt4.set_ylabel("Probability")

	
	plt.show()
"""
if dim == 2:
	#set subplot
	fig,(plt1,ax2) = plt.subplots(1,2)

	# Second subplot
	plt1.plot(c1[:,0],c1[:,1],'o',color='red')
	plt1.plot(c2[:,0],c2[:,1],'o',color='blue')
	plt1.plot([],[],[],color='green')
	start = [0,0]
	plt1.quiver(start[0],start[1],w[0],w[1],color = 'green',scale_units='xy', scale=1)
	plt1.quiver(start[0],start[1],-w[0],-w[1],color = 'green',scale_units='xy', scale=1)
	plt1.axhline(y = 0, color = 'black', linestyle = '-') 
	plt1.axvline(x = 0, color = 'black', linestyle = '-')
	plt1.grid()
	plt1.axis([-10, 10 , -10, 10])
	plt1.set_xlabel("x1")
	plt1.set_ylabel("x2")
	#plt1.title(" Fisher's criterion")
	plt1.legend(['Class 1','Class 2','Optimal vector'])

	# Second subplot
	ax2.plot(x1_kde,y1_kde)
	ax2.plot(x2_kde,y2_kde)
	ax2.set_xlabel("LDA score")
	ax2.set_ylabel("Probability")
	ax2.legend(['Class 1','Class 2'])
	plt.show()

else:
	#set subplot
	fig = plt.figure(figsize=plt.figaspect(2.))
	fig.suptitle('Data visualization')

	# First subplot
	ax = fig.add_subplot(2, 1, 1)
	ax.plot(x1_kde,y1_kde)
	ax.plot(x2_kde,y2_kde)
	ax.set_xlabel("LDA score")
	ax.set_ylabel("Probability")
	ax.legend(['Class 1','Class 2'])

	# Second subplot
	ax = fig.add_subplot(2, 1, 2, projection='3d')
	ax.scatter(c1[:,0], c1[:,1], c1[:,2],'green')
	ax.scatter(c2[:,0], c2[:,1], c2[:,2],'blue')
	ax.scatter([],[],[],color = 'red')
	start = [0,0,0]
	ax.quiver(start[0],start[1],start[2],w[0],w[1],w[2],color = 'r')
	ax.quiver(start[0],start[1],start[2],-w[0],-w[1],-w[2],color = 'r')
	ax.axes.set_xlim3d(left=-6, right=6) 
	ax.axes.set_ylim3d(bottom=-6, top=6) 
	ax.axes.set_zlim3d(bottom=-6, top=6)
	ax.set_xlabel('x1',fontsize = 10)
	ax.set_ylabel('x2',fontsize = 10)
	ax.set_zlabel('x3',fontsize = 10)
	ax.set_title(" Fisher's criterion")
	ax.legend(['Class 1','Class 2','Optimal vector'])

	plt.show()




"""

















