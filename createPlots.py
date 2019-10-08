from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from plotfunctions import plotOlsMSEVsComplexity,plotRidgeLambdaAnalysis,plotLassoLambdaAnalysis,plotRidgeBetaVsLambda,plotLassoBetaVsLambda,plotMSEvsNoise,compareRegressionMethodsAtOptimalConditions,fitTerrainData
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator

np.set_printoptions(precision=4)
np.random.seed(25)
#Create datapoints:
#Number of samples in each dimension:

use_terrain_data = False

if(use_terrain_data):
    terrain = imread("SRTM_data_Norway_25.tif")
    n = len(terrain[0])
    m = len(terrain[:, 1])
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)

    '''
    # Show the terrain
    plt.figure(1)
    ax = plt.gca()
    plt.title("Terrain Data")
    plt.imshow(terrain, cmap='coolwarm')
    # ax.grid(color='w', linestyle='-', linewidth=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    X, Y = x,y

    # Plot the surface.

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, terrain.reshape(X.shape), rstride=1, cstride=1, cmap="coolwarm",
                            linewidth=0, antialiased=False)
    fig.colorbar(surf, ax=ax1)

    ax2 = fig.add_subplot(122)
    cs = ax2.contourf(X, Y, terrain.reshape(X.shape), cmap='coolwarm')
    ax2.contour(cs, colors='k')
    fig.colorbar(cs, ax=ax2)
    plt.show()
    '''
    z = terrain.ravel()
    #z = (z-min(z))/(max(z)-min(z))
    #standardize the data
    #z = (z-np.mean(z))/np.std(z)
    print(min(z))
    fitTerrainData(x, y, terrain, deg=9, lamb=0, folds=5, method='OLS', Franke=False)
    fitTerrainData(x, y, terrain, deg=14, lamb=10**-9, folds=5, method='Ridge', Franke=False)
    fitTerrainData(x, y, terrain, deg=20, lamb=10**-12, folds=5, method='Lasso', Franke=False)
    x = x.ravel()
    y = y.ravel()

else:
    N = 50
    x = np.arange(0, 1, 1/N)
    y = np.arange(0, 1, 1/N)
    x,y = np.meshgrid(x,y)
    sigma = 1
    z = FrankeFunc(x,y)
    error = np.random.normal(0,sigma,size=z.shape)
    z = z+error
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    deg = 3
    i = 0

folds = 5







'''
#OLS: MSE vs Complexity
#degrees = np.arange(20)
degrees = np.arange(1,21,2)
plotOlsMSEVsComplexity(x,y,z,degrees,folds)

#Ridge: MSE vs Complexity and lambda
#degrees = np.arange(2,16,2)
#degrees = [35]
#lambdas = [10**-12,10**-9,10**-6,10**-3,10**-1,10**0]


#Franke
degrees = np.arange(2,20,2)
#lambdas = np.logspace(-10,1,10)
lambdas = [10**-10, 10**-8, 10**-6, 10**-4, 10**-3,10]
#degrees = np.arange(1,20,1)
degrees = [3]
plotRidgeLambdaAnalysis(x,y,z,degrees,lambdas,folds)


#MSE vs Complexity and lambda for Lasso
#degrees = [3]
#egrees = np.arange(1,10,1)
lambdas = np.logspace(-12,0,50)
#degrees = np.arange(4,23,2)
degrees = [20]
#lambdas = [10**-12,10**-9,10**-6]
#lambdas = [10**-1]
#lambdas = [10**-2,10**-1,10**0,10]
plotLassoLambdaAnalysis(x,y,z,degrees,lambdas,folds)

#Ridge: Beta vs lambda
deg = 3
lambdas = [10**-8, 10**-6, 10**-5, 10**-4,10**-3,10**-2,10**-1,1,10,100, 1000, 10**4]
plotRidgeBetaVsLambda(x,y,z,deg,lambdas,folds)


#Lasso: Beta vs lambda
deg = 3
#lambdas = [10**-6, 10**-5, 10**-4,10**-3,10**-2,10**-1,1,10,100, 1000, 10**4]
lambdas = np.logspace(-4,2,50)
plotLassoBetaVsLambda(x,y,z,deg,lambdas,folds)


#MSE vs noise on Francefunk for OLS,Lasso and Ridge
folds = 5
deg = 5
lamb = 10**-3
sigmas = np.linspace(0,1,10)
plotMSEvsNoise(x,y,deg,sigmas,lamb,folds)
'''
if(use_terrain_data):
    degOLS = 9
    degRidge = 14
    degLasso = 20
    lambRidge = 10**-9
    lambLasso = 10**-12
else:
    degOLS = 4
    degRidge = 5
    degLasso = 3
    lambRidge = 10**-3
    lambLasso = 10**-4
compareRegressionMethodsAtOptimalConditions(x,y,z,degOLS,degRidge,degLasso,lambRidge,lambLasso,folds,Franke=True)



