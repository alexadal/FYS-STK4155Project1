from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from random import random, seed
import scipy.linalg as scl
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=4)

def FrankeFunc(x, y, sigma, noise=False):
    err = [0] * len(x)
    if noise:
     for i in range(len(x)):
      err[i] = np.random.normal(0, sigma)
#    print("Error", err)
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4 + err



def DesignMatrix5(x,y):
    return np.c_[np.ones((len(x))),x,y,x**2,y**2,x*y,
           x** 3, x**2*y, x*y**2, y**3,
           x**4, x**3*y, x**2*y**2, x*y**3, y**4,
           x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]


def ols_svd(x: np.ndarray, y: np.ndarray, z, deg) -> np.ndarray:
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    u, s, v = scl.svd(X_)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ z

def ols_svd_X(X_, z):
    u, s, v = scl.svd(X_)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ z


def ols_inv(x: np.ndarray, y: np.ndarray, z, deg) -> np.ndarray:
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    return scl.inv(X_.T @ X_) @ (X_.T @ z)


def OLS_sk(x, y, z, deg):
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    clf = LinearRegression()
    return clf.fit(X_, z)

def OLS5_(x, y, z):
    x_deg = DesignMatrix5(x, y)
    u, s, v = scl.svd(x_deg)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ z

def Ridge_sk(x, y, z, lamd, deg):
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    return Ridge(alpha=lamd).fit(X_, z)


def Ridge_x(x, y, z, lamd, deg):
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    l_vec = np.eye(len(X_[0])) * lamd
    return np.linalg.inv(X_.T.dot(X_)+l_vec).dot(X_.T).dot(z)

def Ridge_X(X_, z, lamd):
    l_vec = np.eye(len(X_[0])) * lamd
    #print("Lvec",l_vec)
    return np.linalg.inv(X_.T.dot(X_)+l_vec).dot(X_.T).dot(z)

def pred5_(x, y, beta):
    x_deg = DesignMatrix5(x,y)
    return x_deg @ beta


def pred_(x, y, beta, deg):
    # Create design matrix
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    return X_ @ beta

def pred_X(X_, beta):
    # Return beta times designmatrix
    return X_ @ beta

#Function to use with Lasso
def pred_skl(x, y, beta, deg):
    #Create design matrix
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    #Return beta times designmatrix
    return beta.predict(X_)


def MSE(z, z_):
    #As defined
    z, z_ = z.flatten(), z_.flatten()
    n = np.size(z_)
    return np.sum((z - z_) ** 2) / n


def R_2(z, z_):
    #As defined
    z, z_ = z.flatten(), z_.flatten()
    return 1 - np.sum((z - z_) ** 2) / np.sum((z - np.mean(z_)) ** 2)


def Conf_i(z,z_,x,y,beta,deg):
    #Create design matrix
    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
    #Sigma as defined by book/slides - 1D
    sigma = (1./(len(z)-len(X_[0,:])-1))*sum(((z-z_)**2))
    #Obtain covar matrix
    covar = np.linalg.inv(X_.T.dot(X_))*sigma
    #Standard error of betas are diagonal elements of covar matrix
    se_beta = np.sqrt(np.diagonal(covar))
    #Make beta to be 1col times as many rows necesarry to get 1col
    se_beta.reshape(-1,1)
    interval = np.zeros((len(beta), 3))
    for i in range((len(beta))):
        #Save first row beta
        interval[i][0] = beta[i]
        #Save interval, lowwer first, 1.96 as we want 95% CI
        interval[i][1] = beta[i] - 1.96 * se_beta[i]
        interval[i][2] = beta[i] + 1.96 * se_beta[i]
    return interval


def random_indices(indices,X_fold,z_fold):
    #create designemratrix
    #Shuffle the index-vector from input raondomly
    np.random.shuffle(indices)
    #Obtain original Matrices
    X_fold = X_fold[indices]
    z_fold = z_fold[indices]
    return X_fold,z_fold

def k_fold(x,y,z,deg,folds,reg_type,shuffle=True,lamd=0):
    #create designermatrix

    x_deg = np.c_[x, y]

    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(x_deg)
#    if split:
#        X_, X_val, z, z_val = train_test_split(X_,z,test_size =0.2)
    indices = np.arange(len(z))
    #Shuffle the dataset randomly if chosen
    if shuffle:
        X_shuffle,z_shuffle = random_indices(indices,X_,z)
        z = z_shuffle
        X_ = X_shuffle

#    print("Zfold",np.matrix(X_[:][1].tolist()))
    #Split data into k-folds
    X_fold = np.array_split(X_,folds)
    z_fold = np.array_split(z,folds)

    #Initiate arrays
    R2_tot = []
    MSE_test = []
    MSE_train = []
    bias = []
    variance = []
    for i in range(folds):
        X_fold_i = X_fold
        z_fold_i = z_fold
        #take out test vars
        X_test = X_fold[i]
        z_test = z_fold[i]
        #delete row --> 0 from train set
        X_train = np.delete(X_fold_i,i,0)
        z_train = np.delete(z_fold_i,i,0)
        X_train = np.vstack(X_train)
        z_train = np.vstack(z_train)
        #Make z_train 1D
        z_train = z_train.ravel()
        #Choose Model type
        if reg_type == 'OLS':
            beta = ols_svd_X(X_train,z_train)

        elif reg_type == 'Ridge':
            beta = Ridge_X(X_train,z_train,lamd)
        elif reg_type == 'Lasso':
            clf = Lasso(alpha=0.1)
            clf.fit(X_train,z_train)
            beta = clf.coef_
        else:
            print("Not a valid Regression model")
            return 0
        #Create values based on training predictors
        z_pred = X_test@beta
        z_pred_train = X_train@beta
        MSE_train = np.append(MSE_train, MSE(z_train, z_pred_train))
        MSE_test = np.append(MSE_test,MSE(z_test, z_pred))
        bias = np.append(bias, (z_test-np.mean(z_pred,keepdims=True)) ** 2)
        variance = np.append(variance, np.var(z_pred,keepdims=True))

    print('variance')

    print(variance)
    print('MSE_test')

    print(MSE_test)
    print('bias')

    print(bias)
    MSE_test_avg = np.mean(MSE_test,keepdims=True)
    MSE_train_avg = np.mean(MSE_train,keepdims=True)
    bias_avg = np.mean(bias,keepdims=True)
    variance_avg = np.mean(variance,keepdims=True)
    return MSE_test_avg, MSE_train_avg, bias_avg, variance_avg

