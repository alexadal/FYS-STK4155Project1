from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from random import random, seed
import scipy.linalg as scl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import resample

np.set_printoptions(precision=4)

def FrankeFunc(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4



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

def k_fold1(x,y,z,deg,folds,reg_type,shuffle=True,lamd=0):
    #create designermatrix

    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=deg)
    X_t = poly.fit_transform(x_deg)
    z_t = z
    print(X_t.shape)

    if shuffle:
        indices = np.arange(len(z_t))
        X_t, z_t = random_indices(indices,X_t,z_t)
        print("True")
    else:
        print("False")

    #print("Zfold",np.matrix(X_[:][1].tolist()))
    #Split data into k-folds
    X_fold = np.array_split(X_t,folds)
    z_fold = np.array_split(z_t,folds)

    #Initiate arrays
    R2_tot = []
    MSE_tot = []
    MSE_train = []
    MSE_tot_avg = []
    MSE_train_avg = []
    bias = []
    variance = []
    beta_avg = []


    for i in range(folds):
        X_train_i = X_fold
        z_train_i = z_fold
        #take out test vars
        X_test = X_fold[i]
        z_test = z_fold[i]
        #delete row --> 0 from train set
        X_train = np.delete(X_train_i,i,0)
        z_train = np.delete(z_train_i,i,0)
        X_train = np.vstack(X_train)
        z_train = np.vstack(z_train)
        #Make z_train 1D
        z_train = z_train.ravel()
        #Choose Model type
        if reg_type == 'OLS':
            beta = ols_svd_X(X_train,z_train)

        elif reg_type == 'Ridge':
            beta = Ridge_X(X_train,z_train,lamd)
        else:
            print("Not a valid Regression model")
            return 0
        #Create values based on training predictors
        z_pred = X_test@beta
        z_train_pred = X_train@beta

        MSE_tot = np.append(MSE_tot,MSE(z_test, z_pred))
        MSE_train = np.append(MSE_train, MSE(z_train, z_train_pred))
        bias = np.append(bias, np.mean((z_test - np.mean(z_pred)) ** 2))
        variance = np.append(variance, np.var(z_pred))

    bias_avg = np.average(bias)
    variance_avg = np.average(variance)
    MSE_tot_avg = np.average(MSE_tot)
    MSE_train_avg = np.average(MSE_train)

    print("Beta_avg",beta_avg)

    return MSE_tot_avg, MSE_train_avg, bias_avg, variance_avg


def k_fold2(x,y,z,deg,folds,reg_type,lamd=0):

    kf = KFold(folds,True,5)



    #Initiate arrays
    R2_tot = []
    MSE_tot = []
    MSE_train = []
    MSE_tot_avg = []
    MSE_train_avg = []
    bias = []
    variance = []

    betas = []

    i = 0
    for train_inds,test_inds in kf.split(x):
        xtrain = x[train_inds]
        ytrain = y[train_inds]
        ztrain = z[train_inds]


        xtest = x[test_inds]
        ytest = y[test_inds]
        ztest = z[test_inds]


        #Make z_train 1D
        ztrain = ztrain.ravel()
        x_ = np.c_[xtrain,ytrain]
        poly = PolynomialFeatures(degree=deg)
        X_train = poly.fit_transform(x_)

        #Choose Model type
        if reg_type == 'OLS':
            beta = ols_svd_X(X_train,ztrain)

        elif reg_type == 'Ridge':
            beta = Ridge_X(X_train,ztrain,lamd)
        else:
            print("Not a valid Regression model")
            return 0

        x_t = np.c_[xtest, ytest]
        poly = PolynomialFeatures(degree=deg)
        X_test = poly.fit_transform(x_t)
        #Create values based on training predictors
        z_pred = X_test@beta
        z_train_pred = X_train@beta

        MSE_tot = np.append(MSE_tot,MSE(ztest, z_pred))
        MSE_train = np.append(MSE_train, MSE(ztrain, z_train_pred))
        bias = np.append(bias, np.mean((ztest - np.mean(z_pred)) ** 2))
        variance = np.append(variance, np.var(z_pred))

    bias_avg = np.average(bias)
    variance_avg = np.average(variance)
    MSE_tot_avg = np.average(MSE_tot)
    MSE_train_avg = np.average(MSE_train)

    return MSE_tot_avg, MSE_train_avg, bias_avg, variance_avg




def BV(x,y,z,deg,folds,reg_type,lamb = 0):


    xt, x_test, yt, y_test, zt, z_test = train_test_split(x, y, z, test_size=1./folds,shuffle=True)

    kf = KFold(folds, True, 5)

    # Initiate arrays

    error = []
    bias = []
    variance = []
    z_pred = np.empty((z_test.shape[0], folds))
    # betas = np.zeros((deg,p))
    betas = []

    i = 0
    for train_inds, test_inds in kf.split(xt):
        xtrain = xt[train_inds]
        ytrain = yt[train_inds]
        ztrain = zt[train_inds]

        xtest = xt[test_inds]
        ytest = yt[test_inds]
        ztest = zt[test_inds]

        # Make z_train 1D
        ztrain = ztrain.ravel()
        x_ = np.c_[xtrain, ytrain]
        poly = PolynomialFeatures(degree=deg)
        X_train = poly.fit_transform(x_)

        # Choose Model type
        if reg_type == 'OLS':
            beta = ols_svd_X(X_train, ztrain)

        elif reg_type == 'Ridge':
            beta = Ridge_X(X_train, ztrain, lamd)
        else:
            print("Not a valid Regression model")
            return 0

        x_t = np.c_[x_test, y_test]
        poly = PolynomialFeatures(degree=deg)
        X_test = poly.fit_transform(x_t)
        # Create values based on training predictors
        z_pred[:,i] = X_test @ beta.ravel()
        #z_train_pred = X_train @ beta.ravel()

    z_test = z_test.reshape(z_test.shape[0],1)
    error = np.mean(np.mean((z_test - z_pred) ** 2, axis=1, keepdims=True))
    bias = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(z_pred, axis=1, keepdims=True))

    return error, bias, variance

















def bootstrap(x,y,z,deg,reg_type,n_bootstraps,lambd=0):

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)
    z_test = z_test.reshape(z_test.shape[0],1)

    poly = PolynomialFeatures(degree=deg)
    x_tst = np.c_[x_test,y_test]
    X_tst = poly.fit_transform(x_tst)

    print("z_test", z_test.shape)

    print("Straps",n_bootstraps)

    z_pred = np.empty((z_test.shape[0], n_bootstraps))
    z_pred_train = np.empty((z_train.shape[0], n_bootstraps))
    z_t =  np.empty((z_train.shape[0], n_bootstraps))
    print("Straps", z_pred.shape)
    err = []
    bi = []
    vari = []
    MSE_train = []
    for i in range(n_bootstraps):

        x_, y_, z_ = resample(x_train, y_train,z_train)
        # Choose Model type
        x_bot = np.c_[x_,y_]
        X_train = poly.fit_transform(x_bot)
        if reg_type == 'OLS':
            beta = ols_svd_X(X_train, z_)

        elif reg_type == 'Ridge':
            beta = Ridge_X(X_train, z_, lambd)
        else:
            print("Not a valid Regression model")
            return 0

        # Evaluate the new model on the same test data each time.
        z_pred[:, i] = X_tst@beta.ravel()
        z_pred_train[:, i] = X_train @ beta.ravel()
        z_t[:, i] = z_

    error = np.mean(np.mean((z_test - z_pred) ** 2, axis=1, keepdims=True))
    bias = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(z_pred, axis=1, keepdims=True))
    z_train = z_train.reshape(z_train.shape[0], 1)
    MSE_t= np.mean(np.mean((z_t - z_pred_train) ** 2, axis=1, keepdims=True))
    err.append(error)
    bi.append(bias)
    vari.append(variance)
    MSE_train.append(MSE_t)
    print(MSE_train)
    print(err)
    print(bi)
    return error, MSE_t, bias, variance