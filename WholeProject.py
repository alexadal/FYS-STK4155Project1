from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=4)
np.random.seed(25)
#Create datapoints:
#Number of samples in each dimension:
N = 5000
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)

sigma = 1
z = FrankeFunc(x,y)
error = np.random.normal(0,sigma,size=z.shape)
z = z+error
folds = 5
deg = 5
i = 0

#OLS: MSE vs Complexity
'''
degrees = np.arange(30)
MSE_test = np.zeros(len(degrees))
variance= np.zeros(len(degrees))
bias= np.zeros(len(degrees))
MSE_train = np.zeros(len(degrees))
beta = np.zeros(len(degrees))
for deg in degrees:
    MSE_test[i], MSE_train[i], bias[i], variance[i] = k_fold1(x, y, z, deg, folds,'OLS')
    i = i+1
MSE_test = np.sqrt(MSE_test)
MSE_train = np.sqrt(MSE_train)

fig = plt.figure()
line_test, = plt.plot(degrees,MSE_test,label='TEST')
line_train, = plt.plot(degrees,MSE_train,label='TRAINING')
plt.legend(handles=[line_test,line_train])
plt.show()
'''

'''
#Ridge: MSE vs Complexity and lambda

degrees = np.arange(1,30,2)
lambdas = [10**-6,10**-5,10**-4,10**-3,10**-1,1,10,100]
MSE_test = np.zeros((len(lambdas),len(degrees)))
variance= np.zeros((len(lambdas),len(degrees)))
bias= np.zeros((len(lambdas),len(degrees)))
MSE_train = np.zeros((len(lambdas),len(degrees)))
beta = np.zeros((len(lambdas),len(degrees)))

i = 0
for lamb in lambdas:
    j = 0
    for deg in degrees:
        MSE_test[i,j], MSE_train[i,j], bias[i,j], variance[i,j] = k_fold1(x, y, z, deg, folds,'Ridge',lamb=lamb,shuffle=True)
        j = j+1
    i = i+1


fig = plt.figure()
i = 0
lines = np.zeros(len(lambdas))
linenames = [None]*len(lambdas)
print(MSE_test[0,:])
for i in range(len(lambdas)):
    linenames[i] = 'Lambda = ' + str(lambdas[i])
    lines = plt.plot(degrees, MSE_test[i,:],label=linenames[i])

plt.legend(linenames)
plt.show()
'''


#MSE vs Complexity for Lasso
degrees = np.arange(1,10,1)
lambdas = [10**-4,10**-3,10**-1,1,10,100,10**3]
MSE_test = np.zeros((len(lambdas),len(degrees)))
variance= np.zeros((len(lambdas),len(degrees)))
bias= np.zeros((len(lambdas),len(degrees)))
MSE_train = np.zeros((len(lambdas),len(degrees)))
beta = np.zeros((len(lambdas),len(degrees)))

i = 0
for lamb in lambdas:
    j = 0
    for deg in degrees:
        MSE_test[i,j], MSE_train[i,j], bias[i,j], variance[i,j] = k_fold1(x, y, z, deg, folds,'Lasso',lamb=lamb,shuffle=True)
        j = j+1
    i = i+1


fig = plt.figure()
i = 0
lines = np.zeros(len(lambdas))
linenames = [None]*len(lambdas)
print(MSE_test[0,:])
for i in range(len(lambdas)):
    linenames[i] = 'Lambda = ' + str(lambdas[i])
    lines = plt.plot(degrees, MSE_test[i,:],label=linenames[i])

plt.legend(linenames)
plt.show()



'''
#Ridge: Beta vs lambda
deg = 5
lambdas = [10**-6, 10**-5, 10**-4,10**-3,10**-2,10**-1,1,10,100, 1000, 10**4]
MSE_test = np.zeros((len(lambdas)))
variance= np.zeros((len(lambdas)))
bias= np.zeros((len(lambdas)))
MSE_train = np.zeros((len(lambdas)))
beta = np.zeros((len(lambdas),20))

i = 0
for lamb in lambdas:
    beta[i,:] = k_fold1(x, y, z, deg, folds,'Ridge',lamb=lamb,shuffle=True,beta_out=True)
    print(beta.shape)
    i = i+1


fig = plt.figure()
lines = np.zeros(20)
linenames = [None]*len(lines)
for j in range(20):
    linenames[j] = 'Beta nr = ' + str([j])
    lines = plt.plot(lambdas, beta[:,j],label=linenames[j])
plt.xscale('log')
plt.axhline()
plt.legend(linenames)
plt.show()

'''

'''
#Lasso: Beta vs lambda
deg = 5
#lambdas = [10**-6, 10**-5, 10**-4,10**-3,10**-2,10**-1,1,10,100, 1000, 10**4]
lambdas = np.logspace(-6,4,30)
MSE_test = np.zeros((len(lambdas)))
variance= np.zeros((len(lambdas)))
bias= np.zeros((len(lambdas)))
MSE_train = np.zeros((len(lambdas)))
beta = np.zeros((len(lambdas),20))

i = 0
for lamb in lambdas:
    beta[i,:] = k_fold1(x, y, z, deg, folds,'Lasso',lamb=lamb,shuffle=True,beta_out=True)
    print(beta.shape)
    i = i+1


fig = plt.figure()
lines = np.zeros(20)
linenames = [None]*len(lines)
for j in range(20):
    linenames[j] = 'Beta nr = ' + str([j])
    lines = plt.plot(lambdas, beta[:,j],label=linenames[j])
plt.xscale('log')
plt.axhline()
#plt.legend(linenames)
plt.show()

'''

'''
#MSE vs noise on Francefunk for OLS,Lasso and Ridge
folds = 5
deg = 5
lamb = 10**-3
sigmas = np.linspace(0,1,10)
MSE_test = np.zeros((3,len(sigmas)))
variance= np.zeros((3,len(sigmas)))
bias= np.zeros((3,len(sigmas)))
MSE_train = np.zeros((3,len(sigmas)))
i = 0
for sigma in sigmas:
    z = FrankeFunc(x,y)
    error = np.random.normal(0,sigma,size=z.shape)
    z = z+error
    MSE_test[0,i], MSE_train[0,i], bias[0,i], variance[0,i] = k_fold1(x, y, z, deg, folds,'OLS', shuffle=True)
    MSE_test[1, i], MSE_train[1, i], bias[1, i], variance[1, i] = k_fold1(x, y, z, deg, folds, 'Ridge', lamb=lamb, shuffle=True)
    MSE_test[2, i], MSE_train[2, i], bias[2, i], variance[2, i] = k_fold1(x, y, z, deg, folds, 'Lasso', lamb=100, shuffle=True)
    i = i+1

fig = plt.figure()
lines = np.zeros(3)
linenames = [None]*3
linenames[0] = 'OLS'
linenames[1] = 'Ridge'
linenames[2] = 'Lasso'
for j in range(3):
    lines = plt.plot(sigmas, MSE_test[j,:], label=linenames[j])
#plt.axhline()
plt.legend(linenames)
plt.show()
'''