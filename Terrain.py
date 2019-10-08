import numpy as np
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from random import seed
from numpy.random import rand
from functions import FrankeFunc, ols_svd, OLS_sk, ols_inv, pred_,pred_skl, Ridge_sk, MSE, R_2, OLS5_, pred5_, Ridge_x, Conf_i, k_fold1,BV, k_fold2,bootstrap
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import griddata
# Load the terrain


terrain = imread("SRTM_data_Norway_60_2.tif")

# Show the terrain
plt.figure(1)
ax = plt.gca()
plt.title("Norge")
plt.imshow(terrain, cmap='coolwarm')
#ax.grid(color='w', linestyle='-', linewidth=2)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


#Remove nodata points

#z = terrain
print("Terr",terrain.shape)
n = len(terrain[0])
m = len(terrain[:,1])
x = np.linspace(0,50*m,m)
y = np.linspace(0,50*n,n)
X,Y = np.meshgrid(x,y)
print(X.shape)
print(Y.shape)


# Plot the surface.
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X,Y,terrain.reshape(X.shape), rstride=1, cstride=1, cmap="coolwarm",
linewidth=0, antialiased=False)
fig.colorbar(surf, ax=ax1)
plt.title("Random Patch Norway")
ax2 = fig.add_subplot(122)
cs = ax2.contourf(X, Y,terrain.reshape(X.shape),cmap='coolwarm')
ax2.contour(cs, colors='k')
fig.colorbar(cs, ax=ax2)
plt.show()

z = terrain.ravel()
x = X.ravel()
y = Y.ravel()
#degree of polynomial
degrees = [1,5,10,15,20,25,40]
folds = 5
i = 0

#lamdas = np.logspace(-6, 5, 100)

'''
MSE_test = np.zeros(len(lamdas))
variance= np.zeros(len(lamdas))
bias= np.zeros(len(lamdas))
MSE_train = np.zeros(len(lamdas))
beta = np.zeros(len(lamdas))

for lamd in lamdas:
    MSE_test[i], MSE_train[i], bias[i], variance[i] = k_fold1(x, y, z, deg, folds,'Ridge',lamd=lamd)
    i = i+1

fig = plt.figure()
line_test, = plt.plot(lamdas,MSE_test,label='TEST')
line_train, = plt.plot(lamdas,MSE_train,label='TRAINING')

plt.xscale('log')
plt.legend(handles=[line_test,line_train])

plt.show()
'''
#OLS

MSE_test = np.zeros(len(degrees))
variance= np.zeros(len(degrees))
bias= np.zeros(len(degrees))
MSE_train = np.zeros(len(degrees))
beta = np.zeros(len(degrees))
for deg in degrees:
    MSE_test[i], MSE_train[i], bias[i], variance[i] = k_fold2(x, y, z, deg, folds,'OLS')
    i = i+1

fig = plt.figure()
line_test, = plt.plot(degrees,MSE_test,label='TEST')
line_train, = plt.plot(degrees,MSE_train,label='TRAINING')


plt.legend(handles=[line_test,line_train])

plt.show()


MSE_tst, MSE_tra, bi, va = k_fold1(x,y,z,5,folds,'OLS',shuffle=True)
print('MSE for test with OLS:')
print(MSE_tst)

'''
MSE_test[1], MSE_train[1], bias[1], variance[1] = k_fold1(x,y,z,deg,folds,'OLS',shuffle=True)

#Ridge
MSE_test[2], MSE_train[2], bias[2], variance[2] = k_fold1(x,y,z,deg,folds,'OLS',shuffle=True)
#Lasso
MSE_test[3], MSE_train[3], bias[3], variance[3] = k_fold1(x,y,z,deg,folds,'OLS',shuffle=True)
'''
#Compare