from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import Axes3D

# Make data grid give same shape as x in exercise
x = np.arange(0, 1, 0.001)
y = np.arange(0, 1, 0.001)

sigma = 1
z = FrankeFunc(x,y)
error = np.random.normal(0,sigma,size=z.shape)
print(error.shape)
z_true = z+error
x1d = x
y1d = y

#Find alpha
folds = 5
deg = 5
i = 0
#lamdas = [0, 1, 2, 4, 5, 10,50,100,1000,10**4,10**5,10**8,10**10,10**12,10**13,10**14]
alphas = np.logspace(-4, 0, 100)
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


'''
methods = ['OLS', 'Ridge', 'Lasso']
deg = []
MSE = []
for method in methods:
    MSE[i], MSE[i], bias[i], variance[i], beta[i] = k_fold1(x1d, y1d, z_true, deg, kfold1, method)

end
'''