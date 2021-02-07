"""
FOT
Functional Optimal Transport:
Mapping Estimation and Domain Adaptation for Functional Data
Jiacheng Zhu
jzhu4@andrew.cmu.edu
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
import scipy


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''
    Computes the suffifient statistics of the GP posterior predictive distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def nll_fn(X_train, Y_train, noise, naive=True):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (7), if
               False use a numerically more stable implementation.
    Returns:
        Minimization objective.
    '''

    def nll_naive(theta):
        # Naive implementation of Eq. (7). Works well for the examples
        # in this article but is numerically less stable compared to
        # the implementation in nll_stable below.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise ** 2 * np.eye(len(X_train))
        error = 0.5 * np.log(det(K)) + \
               0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2 * np.pi)
        # print('error.shape =', error.shape)
        return error[0, 0]

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise ** 2 * np.eye(len(X_train))
        L = cholesky(K)
        error = np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
               0.5 * len(X_train) * np.log(2 * np.pi)
        return error[0, 0]

    if naive:
        return nll_naive
    else:
        return nll_stable


class GP_model:
    def __init__(self, x_train, y_train, noise=0.4):
        self.x_train = x_train
        self.y_train = y_train

        self.l_opt = 0
        self.signa_opt = 0

        self.noise = noise

    def train(self):
        res = minimize(nll_fn(self.x_train, self.y_train, self.noise), [1, 1],
                         bounds=((1e-5, None), (1e-5, None)),
                         method='L-BFGS-B')
        self.l_opt, self.signa_opt = res.x

    def predict(self, x_test):
        mu, K = posterior_predictive(x_test, self.x_train, self.y_train,
                             l=self.l_opt, sigma_f=self.signa_opt, sigma_y= self.noise)
        return mu, K