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

# Load the terrain


terrain1 = imread("SRTM_data_Norway_1.tif")
# Show the terrain
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap='coolwarm')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x = terrain1[0]
y = terrain1[:,1]

print(x.shape)
print(y.shape)
#Downsample
xd = x[0::2]
yd = y[0::2]

print(xd.shape)
print(yd.shape)
print(terrain1.shape)

X,Y = np.meshgrid(xd,yd)

x1 = x.ravel()
y1 = x.ravel()

