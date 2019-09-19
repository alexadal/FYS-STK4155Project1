
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import seed
from numpy.random import rand
from functions import FrankeFunc, ols_svd, OLS_sk, ols_inv, pred_,pred_skl, Ridge_sk, MSE, R_2, OLS5_, pred5_, Ridge_x, Conf_i, k_fold
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

np.set_printoptions(precision=4)

#Datapoints - how many & DegreeMax

D = 100
Deg_max = 5
lamd = 0.0
# Make data grid give same shape as x in exercise
np.random.seed(10)

x1 = np.random.rand(D,1)
y1 = np.random.rand(D,1)

print(x1.shape)

#Sort before printing
X = np.sort(x1,axis=0)
Y = np.sort(y1,axis=0)
#Create meshgrid for plotting
x_train,y_train = np.meshgrid(X, Y)

x, y = np.meshgrid(X,Y)

#x1d = x.reshape((D**2, 1))
#y1d = y.reshape((D**2, 1))

#Create 1d arrays
x1d = x.ravel()
y1d = y.ravel()


#Obtain true function
z = FrankeFunc(x_train,y_train,0.2,False)
#True function or train function

z_true = FrankeFunc(x1d,y1d,0.2,False)

print(z.shape)

#Initiate betas
X = X.ravel()
Y = Y.ravel()

beta1 = OLS_sk(X,Y,z,Deg_max)
beta2 = ols_svd(x1d,y1d,z_true,Deg_max)
beta3 = Ridge_x(x1d,y1d,z_true,lamd,Deg_max)
beta4 = OLS5_(X,Y,z)

#Create predicxtors/fitted values

z1_ = pred_skl(X,Y,beta1,Deg_max)
z2_ = pred_(x1d,y1d,beta2,Deg_max)
z3_ = pred_(x1d,y1d,beta3,Deg_max)
z4_ = pred5_(X,Y,beta4)


"""
------------------------------------------------------------------------------------
Part a)
------------------------------------------------------------------------------------

"""



"""
#Plot figures
fig = plt.figure()
ax1 = fig.gca(projection='3d')

# Plot the surface.
surf1 = ax1.plot_surface(x_train, y_train, z3_, cmap=cm.coolwarm,
                       linewidth=0,alpha = 0.8, antialiased=False)

surf2 = ax1.plot_surface(x_train, y_train, z4_, cmap=cm.winter,
                       linewidth=0,alpha = 0.0, antialiased=False)

ax1.scatter(x_train, y_train,z,s=0.3, color = 'black', alpha =0.7)

# Customize the z axis.
ax1.set_zlim(-0.10, 1.40)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf1,ax=ax1, shrink=0.5, aspect=5)
fig.colorbar(surf2, shrink=0.5, aspect=5)

plt.show()
"""

#Find Statistical Properties
print("R2: ", R_2(z,z3_))
print("MSE: ",MSE(z, z3_))

print("R2: ", R_2(z,z2_))
print("MSE: ",MSE(z, z2_))


#
C_i = Conf_i(z_true,z2_,x1d,y1d,beta2,Deg_max)

print("Confidence Intervals OLS",C_i)


C_i_ridge = Conf_i(z_true,z3_,x1d,y1d,beta3,Deg_max)

print("Confidence Intervals Ridge",C_i_ridge)






"""
------------------------------------------------------------------------------------
Initiate k-fold
------------------------------------------------------------------------------------

"""

kfold1 = 5


predictr_f = k_fold(x1d,y1d,z_true,Deg_max,kfold1,'Ridge',shuffle=False,lamd=lamd)

predictr_t = k_fold(x1d,y1d,z_true,Deg_max,kfold1,'Ridge',shuffle=False,lamd=lamd)

predictr_of = k_fold(x1d,y1d,z_true,Deg_max,kfold1,'OLS',shuffle=False)

predictr_ot = k_fold(x1d,y1d,z_true,Deg_max,kfold1,'OLS')

print("Predictor Matrix False",predictr_f)
print("Predictor Matrix True",predictr_t)
print("Predictor Matrix OLS false",predictr_of)
print("Predictor Matrix OLS true",predictr_ot)

"""
------------------------------------------------------------------------------------
Test k-fold with sklearn
------------------------------------------------------------------------------------

"""


k = 5
kfold = KFold(n_splits = k,shuffle=True)
kfold2 = KFold(n_splits = k,shuffle=False)



x_deg = np.c_[x1d, y1d]
poly = PolynomialFeatures(degree=5)
X_ = poly.fit_transform(x_deg)


lmb = 0.0

ridge = Ridge(alpha=lmb)

estimated_mse_folds = cross_val_score(ridge, X_, z_true, scoring='neg_mean_squared_error', cv=kfold)
estimated_mse_folds2 = cross_val_score(ridge, X_, z_true, scoring='neg_mean_squared_error', cv=kfold2)

# cross_val_score return an array containing the estimated negative mse for every fold.
# we have to the the mean of every array in order to get an estimate of the mse of the model
estimated_mse_sklearn = np.abs(estimated_mse_folds)
estimated_mse_sklearn2 = np.abs(estimated_mse_folds2)

print("SK K-Fold True",np.average(estimated_mse_sklearn))
print("SK K-Fold False",np.average(estimated_mse_sklearn2))


