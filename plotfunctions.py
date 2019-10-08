from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=4)

#OLS: MSE vs Complexity
def plotOlsMSEVsComplexity(x,y,z,degrees,folds, Franke = False):
    MSE_test = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))
    bias = np.zeros(len(degrees))
    MSE_train = np.zeros(len(degrees))
    R2 = np.zeros(len(degrees))

    i = 0
    for deg in degrees:
        MSE_test[i], MSE_train[i], bias[i], variance[i],R2[i] = k_fold1(x, y, z, deg, folds,'OLS',shuffle=True, Franke=Franke)
        i = i+1
    minInd = np.where(MSE_test == np.amin(MSE_test))
    optDeg = degrees[minInd]
    lab = 'Optimal degreen = ' + str(optDeg)
    plt.figure()
    line_test, = plt.plot(degrees,MSE_test,label='TEST')
    #line_train, = plt.plot(degrees,MSE_train,label='TRAINING')
    dot, = plt.plot(optDeg,MSE_test[minInd],'bo',label = lab)
    plt.legend(handles=[dot])
    plt.title('OLS')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.show()



#Ridge: MSE vs Complexity and lambda
def plotRidgeLambdaAnalysis(x,y,z,degrees,lambdas,folds, Franke = False):
    MSE_test = np.zeros((len(lambdas),len(degrees)))
    variance= np.zeros((len(lambdas),len(degrees)))
    bias= np.zeros((len(lambdas),len(degrees)))
    MSE_train = np.zeros((len(lambdas),len(degrees)))
    R2 = np.zeros((len(lambdas),len(degrees)))
    i = 0
    for lamb in lambdas:
        j = 0
        for deg in degrees:
            MSE_test[i,j], MSE_train[i,j], bias[i,j], variance[i,j], R2[i,j] = k_fold1(x, y, z, deg, folds,'Ridge',lamb=lamb,shuffle=True, Franke= Franke)
            j = j+1
        i = i+1

    #optDeg = degrees[minInd]

    fig = plt.figure()
    i = 0
    lines = np.zeros(len(lambdas))
    linenames = [None]*len(lambdas)
    print(MSE_test[0,:])
    for i in range(len(lambdas)):
        linenames[i] = 'Lambda = ' + str(lambdas[i])
        lines = plt.plot(degrees, MSE_test[i,:],label=linenames[i])
    plt.title('Ridge')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.legend(linenames)
    plt.show()
    linenames = [None]*len(degrees)

    fig2 = plt.figure()
    for i in range(len(degrees)):
        linenames[i] = 'Degree = ' + str(degrees[i])
        lines = plt.plot(lambdas, MSE_test[:,i],label=linenames[i])
    #plt.plot(lambdas,MSE_test[:,7])
    plt.title('Ridge')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.legend(linenames)

    plt.xscale('log')
    plt.show()




#MSE vs Complexity and lambda for Lasso
def plotLassoLambdaAnalysis(x,y,z,degrees,lambdas,folds, Franke = False):
    MSE_test = np.zeros((len(lambdas),len(degrees)))
    variance= np.zeros((len(lambdas),len(degrees)))
    bias= np.zeros((len(lambdas),len(degrees)))
    MSE_train = np.zeros((len(lambdas),len(degrees)))
    R2 = np.zeros((len(lambdas),len(degrees)))
    i = 0
    for lamb in lambdas:
        j = 0
        for deg in degrees:
            MSE_test[i,j], MSE_train[i,j], bias[i,j], variance[i,j],R2[i,j] = k_fold1(x, y, z, deg, folds,'Lasso',lamb=lamb,shuffle=True, Franke=Franke)
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
    plt.title('Lasso')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.show()

    linenames = [None]*len(degrees)

    fig2 = plt.figure()
    for i in range(len(degrees)):
        linenames[i] = 'Degree = ' + str(degrees[i])
        lines = plt.plot(lambdas, MSE_test[:,i],label=linenames[i])
    plt.plot(lambdas,MSE_test[:,:])
    plt.title('Lasso')
    plt.xlabel('Lambda')

    plt.ylabel('MSE')
    plt.xscale('log')
    plt.show()


#Ridge: Beta vs lambda
def plotRidgeBetaVsLambda(x,y,z,deg,lambdas,folds, Franke = False):
    p = int(0.5 * (deg + 2) * (deg + 1))
    beta = np.zeros((len(lambdas),p))
    i = 0
    for lamb in lambdas:
        beta[i,:] = k_fold1(x, y, z, deg, folds,'Ridge',lamb=lamb,shuffle=True,beta_out=True, Franke = Franke)
        i = i+1
    fig = plt.figure()
    lines = np.zeros(p)
    linenames = [None]*len(lines)
    for j in range(p):
        linenames[j] = 'Beta nr = ' + str([j])
        lines = plt.plot(lambdas, beta[:,j],label=linenames[j])
    plt.xscale('log')
    plt.title('Ridge')
    plt.xlabel('Lambda')
    plt.ylabel('Beta')
    plt.axhline(color='k')
    #plt.legend(linenames)
    plt.show()


#Lasso: Beta vs lambda
def plotLassoBetaVsLambda(x,y,z,deg,lambdas,folds, Franke=False):
    p = int(0.5 * (deg + 2) * (deg + 1))
    beta = np.zeros((len(lambdas),p))
    i = 0
    for lamb in lambdas:
        beta[i,:] = k_fold1(x, y, z, deg, folds,'Lasso',lamb=lamb,shuffle=True,beta_out=True, Franke = Franke)
        i = i+1
    fig = plt.figure()
    lines = np.zeros(p)
    linenames = [None]*len(lines)
    for j in range(p):
        linenames[j] = 'Beta'
        lines = plt.plot(lambdas, beta[:,j],label=linenames[j])
    plt.xscale('log')
    plt.title('Lasso')
    plt.xlabel('Lambda')
    plt.ylabel('Beta')
    plt.axhline(color='k')
    #plt.legend(linenames)
    plt.show()




#MSE vs noise on Francefunk for OLS,Lasso and Ridge
def plotMSEvsNoise(x,y,deg,sigmas,lamb,folds, Franke = False):
    MSE_test = np.zeros((3,len(sigmas)))
    variance = np.zeros((3,len(sigmas)))
    bias = np.zeros((3,len(sigmas)))
    MSE_train = np.zeros((3,len(sigmas)))
    i = 0
    for sigma in sigmas:
        z = FrankeFunc(x,y)
        error = np.random.normal(0,sigma,size=z.shape)
        z = z+error
        MSE_test[0,i], MSE_train[0,i], bias[0,i], variance[0,i], R2[0, i] = k_fold1(x, y, z, deg, folds,'OLS', shuffle=True, Franke=Franke)
        MSE_test[1, i], MSE_train[1, i], bias[1, i], variance[1, i], R2[1, i] = k_fold1(x, y, z, deg, folds, 'Ridge', lamb=lamb, shuffle=True, Franke=Franke)
        MSE_test[2, i], MSE_train[2, i], bias[2, i], variance[2, i], R2[2, i] = k_fold1(x, y, z, deg, folds, 'Lasso', lamb=100, shuffle=True, Franke=Franke)
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
    plt.xlabel('Noise')
    plt.ylabel('MSE')
    plt.legend(linenames)
    plt.show()

def compareRegressionMethodsAtOptimalConditions(x,y,z,degOLS,degRidge,degLasso,lambRidge,lambLasso,folds,Franke = False):
    MSE_test = np.zeros(3)
    variance= np.zeros(3)
    bias= np.zeros(3)
    MSE_train = np.zeros(3)
    R2 = np.zeros(3)
    i = 0
    MSE_test[0], MSE_train[0], bias[0], variance[0], R2[0] = k_fold1(x, y, z, degOLS, folds,'OLS', shuffle=True, Franke=Franke)
    MSE_test[1], MSE_train[1], bias[1], variance[1], R2[1] = k_fold1(x, y, z, degRidge, folds, 'Ridge', lamb=lambRidge, shuffle=True, Franke=Franke)
    MSE_test[2], MSE_train[2], bias[2], variance[2], R2[2] = k_fold1(x, y, z, degLasso, folds, 'Lasso', lamb=lambLasso, shuffle=True, Franke=Franke)
    print('-----------------')
    print('MSE OLS: ' + str(MSE_test[0]))
    print('R2 OLS: ' + str(R2[0]))
    print('MSE Ridge: ' + str(MSE_test[1]))
    print('R2 Ridge: ' + str(R2[1]))
    print('MSE Lasso: ' + str(MSE_test[2]))
    print('R2 Lasso: ' + str(R2[2]))

def fitTerrainData(x,y,terrain,deg=4,lamb = 0.001,folds = 5,method = 'OLS', Franke = False):
    p = int(0.5 * (deg + 2) * (deg + 1))
    beta = np.zeros(p)
    z = terrain.ravel()
    x_ = x.ravel()
    y_ = y.ravel()
    beta = k_fold1(x_, y_, z, deg, folds,method,lamb=lamb,shuffle=True,beta_out=True, Franke=Franke)
    x_deg = np.c_[x_, y_]
    beta = np.transpose(beta)
    poly = PolynomialFeatures(degree=deg)
    X = poly.fit_transform(x_deg)
    #z_est = np.zeros(z.shape)
    z_est = X @ beta
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, terrain, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter(x,y,z_est,marker='.',c='k')
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    titlestring = 'Terrain data fitted with data from a ' + str(deg) +'th degree polynomial, \n created using ' + method + ' regression with lambda = ' + str(lamb)
    plt.title(titlestring)
    plt.show()
