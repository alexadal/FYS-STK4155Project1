from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import Axes3D
'''
#Initiating data
np.set_printoptions(precision=4)

#Datapoints - how many & DegreeMax

D = 10000
Deg_max = 5
# Make data grid give same shape as x in exercise
#np.random.seed(10)

x1 = np.random.rand(D,1)
y1 = np.random.rand(D,1)

#x1 = x1.ravel()
#y1 = y1.ravel()
#print(x1.shape)

#Sort before printing
X = np.sort(x1,axis=0)
Y = np.sort(y1,axis=0)
#Create meshgrid for plotting
#x_train,y_train = np.meshgrid(X, Y)

#x, y = np.meshgrid(X,Y)

#x1d = x.reshape((D**2, 1))
#y1d = y.reshape((D**2, 1))

#Create 1d arrays
x1d = x1.ravel()
y1d = y1.ravel()


#Obtain true function
#z = FrankeFunc(x_train,y_train)
#True function or train function
sigma = 0.5
z_true = np.zeros(D)
for i in range(D):
    error = np.random.normal(0, sigma)
    z_true[i] = FrankeFunc(x1d[i],y1d[i])+error
#Initiate betas
X = X.ravel()
Y = Y.ravel()
'''

np.set_printoptions(precision=4)

#Datapoints - how many & DegreeMax

D = 100
Deg_max = 5
lamd = 0.0
# Make data grid give same shape as x in exercise
#np.random.seed(10)

#x1 = np.random.rand(D,1)
#y1 = np.random.rand(D,1)
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)


x, y = np.meshgrid(x,y)




#Obtain true function
z = FrankeFunc(x,y)
#True function or train function
sigma = 0.2
error = np.random.normal(0,sigma,size=z.shape)
z = z + error
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


x1d = x.ravel()
y1d = y.ravel()
z_true = z.ravel()


#Find alpha
folds = 5
deg = 5
i = 0
#lamdas = [0, 1, 2, 4, 5, 10,50,100,1000,10**4,10**5,10**8,10**10,10**12,10**13,10**14]
alphas = np.logspace(-6, 0, 100)
MSE_test = np.zeros(len(alphas))
variance= np.zeros(len(alphas))
bias= np.zeros(len(alphas))
MSE_train = np.zeros(len(alphas))
beta = np.zeros(len(alphas))
for alpha in alphas:
    print(alpha)
    MSE_test[i], MSE_train[i], bias[i], variance[i] = k_fold2(x1d, y1d, z_true, deg, folds,'Lasso',alpha=alpha)
    i = i+1
MSE_test = np.sqrt(MSE_test)
MSE_train = np.sqrt(MSE_train)

fig = plt.figure()
line_test, = plt.plot(alphas,MSE_test,label='TEST')
line_train, = plt.plot(alphas,MSE_train,label='TRAINING')

plt.xscale('log')
#plt.yscale('log')

#line_test1, = plt.plot(Degrees,average_MSE_test2,label='TEST2')
#line_train1, = plt.plot(Degrees,average_MSE_train2,label='TRAINING2')
#plt.legend(handles=[line_train,line_test,line_test1,line_train1])
plt.legend(handles=[line_test,line_train])

plt.show()


summ = []
sigmavec = np.full(len(alphas),0)
print(sigmavec)
plt.figure(2)
line_bias, = plt.plot(alphas,bias,label='BIAS')
line_var, = plt.plot(alphas,variance,label='VARIANCE')
line_test, = plt.plot(alphas,MSE_test,label='TEST')

summ =variance+bias+sigmavec
dot_bi_var, = plt.plot(alphas,summ,'ro',label='sum')

plt.legend(handles=[line_bias,line_var,line_test,dot_bi_var])
plt.show()


'''
methods = ['OLS', 'Ridge', 'Lasso']
deg = []
MSE = []
for method in methods:
    MSE[i], MSE[i], bias[i], variance[i], beta[i] = k_fold1(x1d, y1d, z_true, deg, kfold1, method)

end
'''