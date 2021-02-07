'''
Experiments on (The General Formulation/ RKHS)
of GP mapping

Find the the map for a single realization
and
Visualize the eign functions

'''

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


def data_domain_1():
    x = np.linspace(0, 10, 10).reshape(-1, 1)

    y_11 = 6 * np.sin(0.2 * (x + 4)) - x / 10 - 3.5
    y_12 = 6 * np.sin(0.2 * (x + 4.5)) + x / 20 - 3.7

    X_train = np.concatenate([x, x], axis=0)
    Y_train = np.concatenate([y_11, y_12], axis=0)

    return X_train, Y_train


def data_domain_2():
    x = np.linspace(0, 10, 10).reshape(-1, 1)

    y_21 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 3.9
    y_22 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 4.2

    X_train = np.concatenate([x, x], axis=0)
    Y2_train = np.concatenate([y_21, y_22], axis=0)

    return X_train, Y2_train


if __name__ == '__main__':

    print()
    print('scipy.__version__ =', scipy.__version__)
    # Notice: Manually get some random data.
    #   Create two trajectories

    x = np.linspace(0, 10, 10).reshape(-1, 1)
    y_11 = 6 * np.sin(0.2 * (x + 4)) - x / 10 - 3.5
    y_12 = 6 * np.sin(0.2 * (x + 4.5)) + x / 20 - 3.7
    y_21 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 3.9
    y_22 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 4.2

    # Notice: Plot it
    fig = plt.figure()
    plt.scatter(x, y_11, marker='+', c='tomato', label='y_11')
    plt.scatter(x, y_12, marker='+', c='fuchsia', label='y_12')

    plt.scatter(x, y_21, marker='s', c='aqua', label='y_21')
    plt.scatter(x, y_22, marker='s', c='teal', label='y_22')

    X_train, Y_train = data_domain_1()

    noise = 0.4

    # Notice: Get the data for training GP_1
    X_test = np.linspace(0, 10, 10).reshape(-1, 1)

    # print('X_train.shape =', X_train.shape)
    # print(Y_train.shape)
    GP_1 = GP_model(x_train=X_train, y_train=Y_train, noise=noise)
    GP_1.train()
    mu_1, K_1 = GP_1.predict(x_test=X_test)


    # Notice: Get the data for training GP_2
    X2_train, Y2_train = data_domain_2()

    GP_2 = GP_model(x_train=X2_train, y_train=Y2_train, noise=noise)
    GP_2.train()
    mu_2, K_2 = GP_2.predict(x_test=X_test)

    # Notice: Plot the GP regression
    diag = np.sqrt(np.diag(K_1))
    diag2 = np.sqrt(np.diag(K_2))

    # plt.plot(X_test, mu_1, c='r', label='Mean')
    plt.fill_between(X_test[:, 0], (mu_1 + diag)[:, 0], (mu_1 - diag)[:, 0], alpha=0.5, color='orange')

    # plt.plot(X_test, mu_2, c='b', label='Mean')
    plt.fill_between(X_test[:, 0], (mu_2 + diag2)[:, 0], (mu_2 - diag2)[:, 0], alpha=0.5, color='navy')

    # Notice:
    #   Start to work on OUR PROPOSED METHOD!
    #   The
    U, D1, UT = np.linalg.svd(K_1)

    V, D2, VT = np.linalg.svd(K_2)

    alpha_test = 1 * np.eye(10)
    alpha_test[1, 1] = 1

    T_test = np.dot(np.dot(V, alpha_test), UT)

    # push_test = np.dot(T_test, y_11)

    # plt.plot(X_test, push_test, marker='+', c='fuchsia', label='Push forward from f_12')


    # Notice: The cost function is
    #   C_lk = d(V alpha U^T f_1l, f_2k)
    #   alpha = (10, )
    def cost_func(alpha, f_1, f_2):
        alpha_T = np.multiply(np.eye(10), alpha)
        T = np.dot(np.dot(V, alpha_T), UT)
        f_sharp = np.dot(T, f_1)
        # print('f_sharp.shape =', f_sharp.shape)
        # print('f_2.shape =', f_2.shape)
        return np.sqrt(np.sum(np.square(f_sharp - f_2)))

    cost = cost_func(alpha_test, y_11, y_22)

    def objective_function(arg):
        alpha = arg

        cost = cost_func(alpha, y_11, y_22) + np.sum(np.square(alpha))
        # print('cost =', cost)
        return cost

    # bounds =
    res_test = minimize(objective_function, np.zeros(10, ),
                     method='L-BFGS-B')
    alpha_opti = res_test.x
    print('alpha_opti =', alpha_opti)
    print('alpha_opti.shape =', alpha_opti.shape)

    # print('np.eye(3) =', np.eye(3))

    # alpha_opti[0] = 0.34
    # alpha_opti[1] = -0.3
    # alpha_opti[2] = 1
    manual_cost = cost_func(alpha_opti, y_11, y_22)
    print('manual_cost =', manual_cost)
    alpha_mat = np.multiply(np.eye(10), alpha_opti)
    # print('alpha_mat.shape =', alpha_mat.shape)

    # print('alpha_mat =', alpha_mat)

    T_test = np.dot(np.dot(V, alpha_mat), UT)
    print('T_test.shape =', T_test.shape)

    push_test = np.dot(T_test, y_11)

    # distance = np.mean(np.abs(push_test - ))
    plt.plot(X_test, push_test, c='orange', label='Push forward from y_11 to y_22')
    plt.legend()

    # Notice: Some results for better understanding
    fig_2 = plt.figure(2)
    # plt.scatter(x, y_11, marker='+', c='tomato', label='y_11 source')
    # plt.scatter(x, y_12, marker='+', c='fuchsia', label='y_12')
    # plt.scatter(x, y_21, marker='s', c='aqua', label='y_21')
    # plt.scatter(x, y_22, marker='s', c='teal', label='y_22 target')

    plt.fill_between(X_test[:, 0], (mu_1 + diag)[:, 0], (mu_1 - diag)[:, 0], alpha=0.2, color='orange')

    # plt.plot(X_test, mu_2, c='b', label='Mean')
    plt.fill_between(X_test[:, 0], (mu_2 + diag2)[:, 0], (mu_2 - diag2)[:, 0], alpha=0.2, color='navy')

    # Notice: Visualize the eignfunctions
    alpha_mat_I = np.eye(10)
    T_test_I = np.dot(np.dot(V, alpha_mat_I), UT)
    push_test_I = np.dot(T_test_I, y_11)
    plt.plot(X_test, push_test_I, c='orange', label='I')

    for i in range(10):
        alpha_mat_iterate = np.zeros((10, 10))
        alpha_mat_iterate[i, i] = 1
        T_test_it = np.dot(np.dot(V, alpha_mat_iterate), UT)
        push_test_it = np.dot(T_test_it, y_11)
        plt.plot(X_test, push_test_it, label='alpha' + str(i + 1) +'=1')

    plt.legend()
    plt.show()















