from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=4)

# Make data grid give same shape as x in exercise
x = np.arange(0, 1, 0.001)
y = np.arange(0, 1, 0.001)
X, Y = np.meshgrid(x,y)

'''
#Obtain true function
z = FrankeFunc(X,Y)
#True function or train function
sigma = 1
error = np.random.normal(0,sigma,size=z.shape)
z = z + error
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



print(z.shape)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''
sigma = 1
z = FrankeFunc(x,y)
error = np.random.normal(0,sigma,size=z.shape)
z_true = z+error
x1d = x
y1d = y
#z_true = z.ravel()
print(z_true.shape)
#Find lambda
folds = 5
deg = 5
i = 0
#lamdas = [0, 1, 2, 4, 5, 10,50,100,1000,10**4,10**5,10**8,10**10,10**12,10**13,10**14]
#lamdas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
lamdas = np.logspace(-6, 5, 100)
MSE_test = np.zeros(len(lamdas))
variance= np.zeros(len(lamdas))
bias= np.zeros(len(lamdas))
MSE_train = np.zeros(len(lamdas))
beta = np.zeros(len(lamdas))
for lamd in lamdas:
    MSE_test[i], MSE_train[i], bias[i], variance[i] = k_fold2(x1d, y1d, z_true, deg, folds,'Ridge',lamd=lamd)
    i = i+1
MSE_test = np.sqrt(MSE_test)
MSE_train = np.sqrt(MSE_train)

fig = plt.figure()
line_test, = plt.plot(lamdas,MSE_test,label='TEST')
line_train, = plt.plot(lamdas,MSE_train,label='TRAINING')

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