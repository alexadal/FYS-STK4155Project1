from functions import MSE,k_fold1,FrankeFunc,k_fold2,Ridge_sk_X
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from plotfunctions import plotOlsMSEVsComplexity,plotRidgeLambdaAnalysis,plotLassoLambdaAnalysis,plotRidgeBetaVsLambda,plotLassoBetaVsLambda,plotMSEvsNoise,compareRegressionMethodsAtOptimalConditions
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator

np.set_printoptions(precision=4)
np.random.seed(25)

def print_menu():
    print('Options:')
    print('1: Plot MSE vs Complexity for OLS')
    print('2: Plot MSE vs Complexity and Lambda for Ridge')
    print('3: Plot MSE vs Complexity and Lambda for Lasso')
    print('4: Plot Betas vs Lambda for Ridge')
    print('5: Plot MSE vs Complexity for Lasso')
    print('q: Quit')
    print('')
    inp = str(input('What would you like to do? (1 - 5, q) '))
    return inp

def get_degrees():
    print('Please provide polynomial degrees as integers:')
    sta = int(input('Minimum degree: '))
    end = int(input('Maximum degree: '))
    step_size = int(input('Step-size: '))
    return np.arange(sta,end,step_size)
def get_lambdas():
    print('Please provide lambdas')
    print('The lambdas will be logarithmically spaced betwen 10^(min) and 10^(max)')
    sta = int(input('Min (integer): '))
    end = int(input('Max (integer): '))
    leng = int(input('Number of points(integer): '))
    return np.logspace(sta,end,leng)
def main():
    use_terrain_data = True
    print('Welcome to this solution of Project 1 in ...')
    print('When plots are shown, please close them to continue the program.')
    user_input = str(input('Do you want to use terrain data(y) or Franke Function(n)? (y/n) '))

    while (user_input != 'y' and user_input != 'n'):
        user_input = input('Invalid input... enter y to use terrain data or n for Franke Function? (y/n) ')
    print('---------------------------------------------------------------------------------------------')
    if user_input == 'y':
        use_terrain_data = True
        Franke = False
        terrain = imread("SRTM_data_Norway_25.tif")
        n = len(terrain[0])
        m = len(terrain[:, 1])
        x = np.linspace(0, 1, m)
        y = np.linspace(0, 1, n)
        x, y = np.meshgrid(x, y)
        print('You have chosen the terrain data. The analysis will be performed on data that looks like this:')
        print('Please close the plot-window to continue the program.')
        # Plot the surface.
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(x, y, terrain.reshape(x.shape), rstride=1, cstride=1, cmap="coolwarm", linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax1)
        plt.title('Terrain data')
        ax2 = fig.add_subplot(122)
        cs = ax2.contourf(x, y, terrain.reshape(x.shape), cmap='coolwarm')
        ax2.contour(cs, colors='k')
        fig.colorbar(cs, ax=ax2)
        plt.title('Contours')
        plt.suptitle('Please close this window to continue the program.')
        plt.show()
        z = terrain.ravel()
        x = x.ravel()
        y = y.ravel()
    else:
        use_terrain_data = False
        Franke = True
        N = 50
        x = np.arange(0, 1, 1/N)
        y = np.arange(0, 1, 1/N)
        x,y = np.meshgrid(x,y)
        sigma = 1
        z = FrankeFunc(x,y)
        error = np.random.normal(0,sigma,size=z.shape)
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # Customize the z axis.
        ax1.set_zlim(-0.10, 1.40)
        ax1.zaxis.set_major_locator(LinearLocator(10))
        ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf1, shrink=0.5, aspect=5)
        plt.title('Franke Function with no noise')

        z = z+error
        ax2 = fig.add_subplot(122,projection='3d')
        ax2.zaxis.set_major_locator(LinearLocator(10))
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        surf2 = ax2.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # Customize the z axis.
        ax2.set_zlim(-4, 4)
        ax2.zaxis.set_major_locator(LinearLocator(10))
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf2, shrink=0.5, aspect=5)
        plt.title('Franke Function with noise (sigma = 1)')
        plt.suptitle('Please close this window to continue the program.')
        plt.show()
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
    #Data is now created and descided.
    folds = 5
    user_input = ''
    valid_inputs = ['1','2','3','4','5','q']
    while (user_input != 'q'):
        user_input = print_menu()
        while user_input not in valid_inputs:
            user_input = str(input('Please refer to the above menu and enter a valid input. '))
        if user_input == '1':
            degrees = get_degrees()
            if len(degrees) == 0:
                degrees = [2, 4, 6, 10]
            print('Thinking about it...')
            plotOlsMSEVsComplexity(x, y, z, degrees, folds, Franke)
        if user_input == '2':
            degrees = get_degrees()
            lambdas = get_lambdas()
            if len(degrees) == 0:
                degrees = [2, 4, 6, 10]
            if len(lambdas) == 0:
                lambdas = [10**-4,10**-2,10**-1,10**0,10]
            print('Thinking about it...')
            plotRidgeLambdaAnalysis(x, y, z, degrees, lambdas, folds, Franke)
        if user_input == '3':
            degrees = get_degrees()
            lambdas = get_lambdas()
            if len(degrees) == 0:
                degrees = [2, 4, 6, 10]
            if len(lambdas) == 0:
                lambdas = [10**-4,10**-2,10**-1,10**0,10]
            print('Thinking about it...')
            plotLassoLambdaAnalysis(x, y, z, degrees, lambdas, folds, Franke)
        if user_input == '4':
            deg = 0
            deg = int(input('Please enter the degree of the polynomial you want: '))
            lambdas = get_lambdas()
            if deg == 0:
                deg  = 3
            if len(lambdas) == 0:
                lambdas = [10**-4,10**-2,10**-1,10**0,10]
            print('Thinking about it...')
            plotRidgeBetaVsLambda(x, y, z, deg, lambdas, folds, Franke)
        if user_input == '5':
            deg = ''
            deg = int(input('Please enter the degree of the polynomial you want: '))
            lambdas = get_lambdas()
            if deg == 0:
                deg = 3
            if len(lambdas) == 0:
                lambdas = [10**-4,10**-2,10**-1,10**0,10]
            print('Thinking about it...')
            plotLassoBetaVsLambda(x, y, z, deg, lambdas, folds, Franke)
        print('---------------------------------------------------------------------------------------------')


main()
