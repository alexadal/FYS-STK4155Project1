from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X

import numpy as np
import matplotlib.pyplot as plt

#Initiating data
np.set_printoptions(precision=4)

#Datapoints - how many & DegreeMax

D = 300
Deg_max = 5
# Make data grid give same shape as x in exercise
np.random.seed(10)

x1 = np.random.rand(D,1)
y1 = np.random.rand(D,1)

x1 = x1.ravel()
y1 = y1.ravel()
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
z = FrankeFunc(x_train,y_train)
#True function or train function
sigma = 0.5
error = np.random.normal(0,sigma,D**2)
z_true = FrankeFunc(x1d,y1d).ravel()+error
#Initiate betas
X = X.ravel()
Y = Y.ravel()


#Find lambda
folds = 5
deg = 5
i = 0
lamdas = [0, 1, 2, 4, 5, 10,50,100,1000,10000,100000]
MSE_test = np.zeros(len(lamdas))
variance= np.zeros(len(lamdas))
bias= np.zeros(len(lamdas))
MSE_train = np.zeros(len(lamdas))
beta = np.zeros(len(lamdas))
for lamd in lamdas:
    MSE_test[i], MSE_train[i], bias[i], variance[i] = k_fold1(x1d, y1d, z_true, deg, folds, 'Ridge',lamd=lamd)
    i = i+1

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


summ = []
sigmavec = np.full(len(lamdas),0)
print(sigmavec)
plt.figure(2)
line_bias, = plt.plot(lamdas,bias,label='BIAS')
line_var, = plt.plot(lamdas,variance,label='VARIANCE')
line_test, = plt.plot(lamdas,MSE_test,label='TEST')

summ =variance+bias+sigmavec
dot_bi_var, = plt.plot(lamdas,summ,'ro',label='sum')

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