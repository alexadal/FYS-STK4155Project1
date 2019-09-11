from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from random import random, seed
import scipy.linalg as scl




def FrankeFunc(x,y,sigma,noise=False):
    err = np.random.normal(0,sigma)*noise
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*(y+err)-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*(y+err)+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*(y+err)-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*(y+err)-7)**2)
    return term1 + term2 + term3 + term4


def ols_svd(x: np.ndarray, y: np.ndarray,z,deg) -> np.ndarray:
    x_deg = np.column_stack((x, y))
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    print(X_)
    u, s, v = scl.svd(X_)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ z


def ols_inv(x: np.ndarray, y: np.ndarray,z,deg) -> np.ndarray:
    x_deg = np.column_stack((x, y))
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    print(X_)
    return scl.inv(X_.T @ X_) @ (X_.T @ z)




def OLS_sk(x,y,z,deg):
    x_deg = np.column_stack((x,y))
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    clf = LinearRegression()
    return clf.fit(X_,z)


def pred_(x,y,b):



