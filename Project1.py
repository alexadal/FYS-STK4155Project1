
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import seed
from numpy.random import rand
from functions import FrankeFunc, ols_svd, OLS_sk, ols_inv, pred_,pred_skl, Ridge_sk, MSE, R_2, OLS5_, pred5_, Ridge_x, Conf_i, k_fold, ols_svd_X
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=4)

#Datapoints - how many & DegreeMax

D = 2000
Deg_max = 5
lamd = 0.0
# Make data grid give same shape as x in exercise
np.random.seed(10)

x1d = np.random.rand(D,1)
y1d = np.random.rand(D,1)


#Sort before printing
#X = np.sort(x1,axis=0)
#Y = np.sort(y1,axis=0)
#Create meshgrid for plotting
#x_train,y_train = np.meshgrid(X, Y)

#x, y = np.meshgrid(X,Y)

#x1d = x.reshape((D**2, 1))
#y1d = y.reshape((D**2, 1))

#Create 1d arrays
#x1d = x.ravel()
#y1d = y.ravel()

#Obtain true function
#z = FrankeFunc(x_train,y_train,0.5,False)
#True function or train function
#1D
sigma = 0.5
z_true = np.zeros((D,1))
err = [0]*D
for i in range(D):
 err[i] = np.random.normal(0, sigma)

 z_true[i,0] = FrankeFunc(x1d[i,0],y1d[i,0]) + err[i]

#print(1-np.mean(err)/np.mean(z_true))


'''
print(z.shape)

#Initiate betas
X = X.ravel()
Y = Y.ravel()

# beta1 = OLS_sk(X,Y,z,Deg_max) not used
beta2 = ols_svd(x1d,y1d,z_true,Deg_max)
beta3 = Ridge_x(x1d,y1d,z_true,lamd,Deg_max)
# beta4 = OLS5_(X,Y,z) not used

#Create predicxtors/fitted values

#z1_ = pred_skl(X,Y,beta1,Deg_max) #Test
z2_ = pred_(x1d,y1d,beta2,Deg_max)
z3_ = pred_(x1d,y1d,beta3,Deg_max)
#z4_ = pred5_(X,Y,beta4) #not needed

'''


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


'''
#Find Statistical Properties
print("R2: ", R_2(z_true,z3_))
print("MSE: ",MSE(z_true, z3_))

print("R2: ", R_2(z_true,z2_))
print("MSE: ",MSE(z_true, z2_))


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

predictr_t = k_fold(x1d,y1d,z_true,Deg_max,kfold1,'Ridge',lamd=lamd)

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

'''


#C
degrees = [0,1,3,5,10,15,20]
#degrees = [3]
MSE_test = [None] * len(degrees)
MSE_train = [None] * len(degrees)
bias = [None] * len(degrees)
variance = [None] * len(degrees)
i = 0
for deg in degrees:
    print('Polynomial:')
    MSE_test[i],MSE_train[i],bias[i],variance[i] = k_fold(x1d,y1d,z_true,deg,5,'OLS',shuffle=False)

    print('Error:', MSE_test[i])
    print('Bias^2:', bias[i])
    print('Var:', variance[i])
    print('{} >= {} + {} = {}'.format(MSE_test[i], bias[i], variance[i], bias[i] + variance[i]))
    i = i+1



'''
degrees = [3]
i = 0
MSE_test = [None] * len(degrees)
MSE_train = [None] * len(degrees)
bias = [None] * len(degrees)
variance = [None] * len(degrees)
x_deg = np.c_[x1d, y1d]
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    X_train, X_test, z_train, z_test = train_test_split(X_, z_true, shuffle=True, test_size=0.2)
    beta = ols_svd_X(X_train,z_train)
    z_pred = X_test@beta
    z_pred_train = X_train@beta
    bias[i] = np.mean((z_test - np.mean(z_pred)) ** 2)
    variance[i] = np.mean(np.var(z_pred))

    MSE_test[i] = MSE(z_test,z_pred)
    MSE_train[i] = MSE(z_train,z_pred_train)
    i = i+1

print('Error:', MSE_test[i-1])
print('Bias^2:', bias[i-1])
print('Var:', variance[i-1])
print('{} >= {} + {} = {}'.format(MSE_test[i-1], bias[i-1], variance[i-1], bias[i-1] + variance[i-1]))

#MSE_test = [0.023627402480447975, 0.001947726079861693, 9.563008799530615e-05, 5.51709066080293e-06 2.423663893175628e-07]
#MSE_train = [0.02204046931212681, 0.001893659106094801, 8.586638895464984e-05, 5.417059451890584e-06 2.237914804468885e-07]

#[0.023627402480447975, 0.0076237815693330655, 0.001902432258179857, 9.065887267684283e-05, 1.0734291011466178e-06]
#[0.02204046931212681, 0.0072182664081734986, 0.0019064223899577426, 8.70357375672621e-05, 1.1218033142965828e-06]

'''
plt.figure(1)
line_test, = plt.plot(degrees,MSE_test,label='TEST')
line_train, = plt.plot(degrees,MSE_train,label='TRAINING')
plt.legend(handles=[line_train,line_test])
plt.show()

plt.figure(2)
line_bias, = plt.plot(degrees,bias,label='BIAS')
line_var, = plt.plot(degrees,variance,label='VARIANCE')
line_test, = plt.plot(degrees,MSE_test,label='TEST')
plt.legend(handles=[line_bias,line_var,line_test])
plt.show()



'''
#Error_avg = [0.02236807748763603, 0.015159384007576104, 0.007315949223080988, 0.0034092598605627495, 0.0019129888406493483]
Error_avg.append(k_fold(x1d,y1d,z_true,50,5,'OLS'))
plt.plot(degrees,Error_avg)
plt.show()
'''
'''
#Recreation of figure..
#Seps:
#  1 find data
# shuffle data
#split data into trainig and test data
#Datapoints
n = 100
x = np.arange(100)
x = x/100
x = np.shuffle(x)
sigma = np.arrange(0,0.5,0.05)
degrees = np.arrange(1,10,1)
i = 0
for s in sigma:
    epsilon(i) = np.random.normal(0, s)
    y[:,i] = x + epsilon(i)
    x_train[:,i], x_test[:,i], y_train[:,i], y_test[:,i] = train_test_split(x, y[:,i], test_size=0.2)
    i = i+1

for deg in degrees
    poly = PolynomialFeatures(degree=deg)
    X_ = np.polyfit(x_train,y_train,deg)

'''
