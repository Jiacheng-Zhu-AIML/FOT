'''
Experiments on (The Generic Functional Optimal Transport)
of GP mapping

According to the derivation of gradients.
Validate the derived gradient of A and Pi

01:
Experiment on 3D plot

Notice:
    need to put the optimizer together

Todo:
    1. Implement and validate the gradient
        a. Diagonal case    [Done]
        b. General case     [Done]
        c. Compare with Scipy optimization results
        .
    2. Think of a better toy example that explores the properties
        a. Non-GP realizations
        b. Imbalance dataset

Notice:
    A good example of realizations:
    data_len = 10
    x = np.linspace(0, 10, data_len).reshape(-1, 1)
    y_12 = 6 * np.sin(0.2 * (x + 4)) - x / 10 - 3.4
    y_11 = 6 * np.sin(0.2 * (x + 4.5)) + x / 20 - 3.6
    l_num = 2
    F1_list = [y_11, y_12]
    y_21 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 3.6
    y_22 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 4.0
    k_num = 2
    F2_list = [y_21, y_22]
'''

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
import scipy

# Notice: Self defined functions
from General_Integral_GP_test import GP_model, data_domain_1, data_domain_2

# Notice:
#  A class for the alternating minimization
#  of toy experiments for
#   Generic Functional Optimal Transport
#   Todo:
#       Minimize A \in (n, n) and Pi \in (l, k)
#       subject to boundaries and constraints
#       where l is the data num of source domain
#             k is the data num of target domain
#       Pi is the coupling/transport plan
#       VAU^T is the transform map
class GFOT_optimization:
    def __init__(self, F1_list, F2_list):
        self.F1_list = F1_list
        self.F2_list = F2_list




if __name__ == '__main__':

    # Notice: Create the test dataset
    data_len = 10  # Notice, this should be consistent all through the process
    x = np.linspace(0, 10, data_len).reshape(-1, 1)

    y_11 = 6 * np.sin(0.2 * (x + 4.5)) + x / 20 - 3.2
    y_12 = 6 * np.sin(0.2 * (x + 4)) - x / 10 - 3.6
    y_13 = 6 * np.sin(0.2 * (x + 4)) - x / 10 - 3.4
    l_num = 3
    F1_list = [y_11, y_12, y_13]

    y_21 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 3.6
    y_22 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 4.0
    y_23 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 4.4
    k_num = 3
    F2_list = [y_21, y_22, y_23]

    # Notice: Plot it
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    layer_1 = np.zeros_like(x)

    layer_2 = np.ones_like(x)

    # ax.scatter3D(layer_1, x, y_11, marker='+', c='tomato', label='y_11')
    ax.scatter3D(layer_1, x, y_11, marker='+', c='tomato', label='y_11')
    ax.scatter3D(layer_1, x, y_12, marker='+', c='fuchsia', label='y_12')
    ax.scatter3D(layer_1, x, y_13, marker='+', c='coral', label='y_13')

    ax.scatter3D(layer_2, x, y_21, marker='s', c='aqua', label='y_21')
    ax.scatter3D(layer_2, x, y_22, marker='s', c='teal', label='y_22')
    ax.scatter3D(layer_2, x, y_23, marker='s', c='dodgerblue', label='y_23')

    # plt.scatter(x, y_11, marker='+', c='tomato', label='y_11')
    # plt.scatter(x, y_12, marker='+', c='fuchsia', label='y_12')
    # plt.scatter(x, y_13, marker='+', c='coral', label='y_13')
    #
    # plt.scatter(x, y_21, marker='s', c='aqua', label='y_21')
    # plt.scatter(x, y_22, marker='s', c='teal', label='y_22')
    # plt.scatter(x, y_23, marker='s', c='dodgerblue', label='y_23')

    # Notice:
    #   Step 0:
    #   Process the data for training
    # X1_train, Y1_train = data_domain_1()
    # X2_train, Y2_train = data_domain_2()

    X1_train = np.concatenate([x, x, x], axis=0)
    Y1_train = np.concatenate([y_11, y_12, y_13], axis=0)

    X2_train = np.concatenate([x, x, x], axis=0)
    Y2_train = np.concatenate([y_21, y_22, y_23], axis=0)

    # Notice:
    #   Step 1: Use GP regression to get K

    # Notice: Get the data for training GP_1

    X_test = np.linspace(0, 10, data_len).reshape(-1, 1)
    # print('X_test =', X_test)

    noise = 0.5
    GP_1 = GP_model(x_train=X1_train, y_train=Y1_train, noise=noise)
    GP_1.train()
    mu_1, K_1 = GP_1.predict(x_test=X_test)

    GP_2 = GP_model(x_train=X2_train, y_train=Y2_train, noise=noise)
    GP_2.train()
    mu_2, K_2 = GP_2.predict(x_test=X_test)

    # Notice: Visualize the learned kernel
    # diag = np.sqrt(np.diag(K_1))
    # diag2 = np.sqrt(np.diag(K_2))
    # plt.plot(X_test, mu_1, c='r', label='Mean')
    # plt.fill_between(X_test[:, 0], (mu_1 + diag)[:, 0], (mu_1 - diag)[:, 0], alpha=0.5, color='orange')
    #
    # plt.plot(X_test, mu_2, c='b', label='Mean')
    # plt.fill_between(X_test[:, 0], (mu_2 + diag2)[:, 0], (mu_2 - diag2)[:, 0], alpha=0.5, color='navy')

    # Notice:
    #   Start to work on OUR PROPOSED METHOD!
    U, D1, UT = np.linalg.svd(K_1)
    V, D2, VT = np.linalg.svd(K_2)  # Notice: V and U and then fixed

    # print('V.T @ V =', V.T @ U)
    # Notice: The objective function
    # Notice: This is the cost: C_ij
    #   alpha:  numpy   (len, )
    #   f_i:    numpy   (len, 1)
    #   f_j:    numpy   (len, len)
    #   V:      numpy   (len, len)
    #   UT:     numpy   (len, len)
    #   return  numpy   scalar   ()
    def cost_func_all(alpha_T, f_l, f_k, V, U):
        return np.sqrt(np.sum(np.square(V @ alpha_T @ U.T @ f_l - f_k)))

    # Notice: The step update method
    #   Case 1. The diagonal A
    initial_alpha = np.eye(10)

    initial_Pi = 1.0/3 * np.ones((l_num, k_num))
    # initial_Pi

    A_mat = initial_alpha
    Pi = initial_Pi

    # Notice: Check the initial A
    # print('alpha_vec.shape =', A_mat.shape)
    # print('Pi.shape =', Pi.shape)
    # f_sharp_11 = V @ A_mat @ U.T @ y_11
    # f_sharp_12 = V @ A_mat @ U.T @ y_12
    # plt.plot(X_test, f_sharp_11, c='red', label='Initial push of y_11', alpha=0.3)
    # plt.plot(X_test, f_sharp_12, c='red', label='Initial push of y_12', alpha=0.3)


    # Notice: Start to do the optimization
    #   The parameters for A
    learning_rate_A = 1e-2

    #   The parameters for Pi
    def cost_function(C, Pi):
        return np.sum(np.multiply(C, Pi))

    def cost_matrix(A, F1_list, F2_list, U=U, V=V):
        C_mat = np.zeros((l_num, k_num))
        for l in range(l_num):
            for k in range(k_num):
                C_mat[l, k] = np.sum(np.square(V @ A @ U.T @ F1_list[l] - F2_list[k]))
        return C_mat

    # Notice: The lbd
    lbd_k = 0.1 * np.ones((k_num, ))
    lbd_l = 0.1 * np.ones((l_num, ))

    # Notice: The augumentation variable
    rho_k = 40 * np.ones((k_num,))
    rho_l = 40 * np.ones((l_num,))

    # Notice:
    #   Start to compute the gradient

    ite_num = 3000
    lr = 0.0001

    # Notice:
    #   To obtain the C_initial
    init_C_matrix = cost_matrix(A=initial_alpha, F1_list=F1_list, F2_list=F2_list)
    last_cost_out = cost_function(C=init_C_matrix, Pi=Pi)


    ite_num = 50
    for ite in range(ite_num):
        print()
        print('ite =', ite)
        # Notice: One step of updating A
        #   Compute the gradient grad_A
        grad_A = np.zeros((data_len, data_len))
        for l in range(l_num):
            for k in range(k_num):
                # Notice: Compute the d C_lk / dA \in R{nxn}
                d_C_lk = (A_mat @ U.T @ F1_list[l] - V.T @ F2_list[k]) @ F1_list[l].T @ U
                grad_A = grad_A + Pi[l, k] * d_C_lk

        grad_A += 2 * A_mat

        # Notice: Only take the diagonal gradient
        grad_A_diag = np.eye(10) * grad_A
        # print('grad_A_diag =', grad_A_diag)
        A_mat = A_mat - learning_rate_A * grad_A_diag

        # Notice: one step of updating Pi
        last_Pi = Pi
        C_matrix_temp = cost_matrix(A=A_mat, F1_list=F1_list, F2_list=F2_list)
        init_cost = cost_function(C=C_matrix_temp, Pi=Pi)
        # print('init_cost =', init_cost)
        cost_difference = 100   # Set a large number
        last_cost = init_cost

        while cost_difference > 0.01:

            grad_Pi = np.zeros((l_num, k_num))
            for l in range(l_num):
                for k in range(k_num):
                    d_pi_lk = C_matrix_temp[l, k] + lbd_k[k] + lbd_l[l] + \
                              1 * rho_k[k] * np.sum(Pi[:, k]) + \
                              1 * rho_l[l] * np.sum(Pi[l, :]) \
                              - 1 * (rho_k[k] + rho_l[l])*Pi[l, k]

                    grad_Pi[l, k] = d_pi_lk

            # Notice: Now obtained the grad of Pi

            Pi_temp = Pi - lr * grad_Pi
            # Notice: The boundary is applied here
            if (np.max(Pi_temp) >= 1.0) or (np.min(Pi_temp) <= 0):
                break
            else:
                Pi = Pi_temp
            # print('Pi =', Pi)
            cost_temp = cost_function(C=C_matrix_temp, Pi=Pi)
            cost_difference = np.abs(last_cost - cost_temp)
            last_cost = cost_temp

        lbd_k = lbd_k + np.multiply(rho_k, np.sum(Pi, axis=0) - 1)
        lbd_l = lbd_l + np.multiply(rho_l, np.sum(Pi, axis=1) - 1)

        # notice: Terminate the optimization
        # break
        this_cost = cost_function(C=C_matrix_temp, Pi=Pi)
        # if np.abs(this_cost - init_cost) < 1e-12:
        #     break


    print('A_mat =', A_mat)
    print('Pi =', Pi)

    f_sharp_11 = V @ A_mat @ U.T @ y_11
    f_sharp_12 = V @ A_mat @ U.T @ y_12
    f_sharp_13 = V @ A_mat @ U.T @ y_13

    ax.scatter3D(layer_2, X_test, f_sharp_11, c='orange', label='T#y_11', alpha=0.3)
    ax.scatter3D(layer_2, X_test, f_sharp_12, c='orange', label='T#y_12', alpha=0.3)
    ax.scatter3D(layer_2, X_test, f_sharp_13, c='orange', label='T#y_13', alpha=0.3)

    # plt.plot(X_test, f_sharp_11, c='orange', label='T#y_11', alpha=0.3)
    # plt.plot(X_test, f_sharp_12, c='orange', label='T#y_12', alpha=0.3)
    # plt.plot(X_test, f_sharp_13, c='orange', label='T#y_13', alpha=0.3)
    plt.legend()
    plt.show()













