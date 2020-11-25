'''
Experiments on (The Generic Functional Optimal Transport)
of GP mapping

According to the derivation of gradients.
Validate the derived gradient of A and Pi

02:
First FOT vs GPOT, The variance issue

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

from WGPOT.wgpot import expmap, logmap

# Notice: Self defined functions
from General_Integral_GP_test import GP_model, data_domain_1, data_domain_2

from General_Functional_OT_Optimization import plot_origin_domain_data
from General_Functional_OT_Optimization import GFOT_optimization


if __name__ == '__main__':

    # Notice: Create the test dataset

    x_start = -10
    x_end = 10

    data_len = 11  # Notice, this should be consistent all through the process
    x = np.linspace(x_start, x_end, data_len).reshape(-1, 1)

    y_11 = 6 * np.sin(0.15 * (x + 5.5)) - x / 10 - 3.6
    y_12 = 6 * np.sin(0.15 * (x + 5.5)) - x / 10 - 4.2
    y_13 = 6 * np.sin(0.15 * (x + 5.5)) - x / 10 - 4.8
    l_num = 3
    F1_list = [y_11, y_12, y_13]

    y_21 = 6 * np.sin(0.15 * (x + 1.7)) - x / 15 - 3.0
    y_22 = 6 * np.sin(0.15 * (x + 1.7)) - x / 15 - 3.6
    y_23 = 6 * np.sin(0.15 * (x + 1.7)) - x / 15 - 4.2
    k_num = 3
    F2_list = [y_21, y_22, y_23]

    # Notice: Plot it
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    layer_1 = np.zeros_like(x)

    layer_2 = np.ones_like(x)

    # ax.scatter3D(layer_1, x, y_11, marker='+', c='tomato', label='y_11')

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

    X_test = np.linspace(x_start, x_end, data_len).reshape(-1, 1)

    noise = 0.2
    GP_1 = GP_model(x_train=X1_train, y_train=Y1_train, noise=noise)
    GP_1.train()
    mu_1, K_1 = GP_1.predict(x_test=X_test)

    GP_2 = GP_model(x_train=X2_train, y_train=Y2_train, noise=noise)
    GP_2.train()
    mu_2, K_2 = GP_2.predict(x_test=X_test)

    # Notice:
    #   Start to work on OUR PROPOSED METHOD!
    U, D1, UT = np.linalg.svd(K_1)
    V, D2, VT = np.linalg.svd(K_2)  # Notice: V and U and then fixed

    # Notice: *********Use the optimizer*********
    GFOT_optimizer = GFOT_optimization(F1_list=F1_list, F2_list=F2_list,
                                       V=V, U=U, l_num=l_num, k_num=k_num, data_len=data_len)

    # notice: Set initial values
    ini_A = np.eye(data_len)
    ini_Pi =1.0/3 * np.ones((l_num, k_num))
    # ini_Pi = np.eye(3) + 1e-16

    lbd_k = 0.1 * np.ones((k_num,))
    lbd_l = 0.1 * np.ones((l_num,))
    lbd_i = 0.1 * np.ones((l_num, k_num))
    s_mat = 0.1 * np.ones((l_num, k_num))

    GFOT_optimizer.Set_Initial_Variables(ini_A=ini_A, ini_Pi=ini_Pi,
                                         ini_lbd_k=lbd_k, ini_lbd_l=lbd_l,
                                         ini_lbd_i=lbd_i, s_mat=s_mat)

    # Notice: Set lagrangian parameters
    rho_k = 20 * np.ones((k_num,))
    rho_l = 20 * np.ones((l_num,))
    rho_i = 10
    gamma_h = 20
    gamma_A = 0.00
    gamma_power = -30
    l_power = 3
    GFOT_optimizer.Set_Parameters(rho_k=rho_k, rho_l=rho_l, rho_i=rho_i,
                                  gamma_A=gamma_A, gamma_h=gamma_h,
                                  gamma_power=gamma_power, l_power=l_power)

    # Notice: Do the optimization
    lr_A = 0.005
    lr_Pi = 0.0001
    ite_num = 300
    A_mat, Pi = GFOT_optimizer.Optimize(lr_A=lr_A, lr_Pi=lr_Pi, tho=1e-5,
                                        diagonal=False, max_iteration=ite_num,
                                        entropy=True, fix_Pi=False,
                                        inequality=False)

    f_sharp_11 = V @ A_mat @ U.T @ y_11
    f_sharp_12 = V @ A_mat @ U.T @ y_12
    f_sharp_13 = V @ A_mat @ U.T @ y_13

    f_sharp_list = [f_sharp_11, f_sharp_12, f_sharp_13]
    # # Notice: Conduct the GPOT
    # v_mu, v_T = logmap(mu_2, K_2, mu_1, K_1)
    # f_shp_11_gpot, _ = expmap(y_11, K_1, v_mu, v_T)
    # f_shp_12_gpot, _ = expmap(y_12, K_1, v_mu, v_T)
    # f_shp_13_gpot, _ = expmap(y_13, K_1, v_mu, v_T)
    #
    # Plot
    # Notice: Give the 3D plot
    #   The mapped data by FOT
    ax.scatter3D(layer_1, x, y_11, marker='+', c='tomato', label='y_11')
    ax.scatter3D(layer_1, x, y_12, marker='+', c='fuchsia', label='y_12')
    ax.scatter3D(layer_1, x, y_13, marker='+', c='coral', label='y_13')

    ax.scatter3D(layer_2, x, y_21, marker='s', c='aqua', label='y_21')
    ax.scatter3D(layer_2, x, y_22, marker='s', c='teal', label='y_22')
    ax.scatter3D(layer_2, x, y_23, marker='s', c='dodgerblue', label='y_23')
    ax.scatter3D(layer_2, X_test, f_sharp_11, c='orange', label='T#y_11', alpha=0.3)
    ax.scatter3D(layer_2, X_test, f_sharp_12, c='orange', label='T#y_12', alpha=0.3)
    ax.scatter3D(layer_2, X_test, f_sharp_13, c='orange', label='T#y_13', alpha=0.3)

    # Notice: Connect the mapped points

    for l in range(l_num):
        for i in range(data_len):
            # for k in range(k_num):
            k = np.argmax(Pi[l, :])
            layer_3d = [0, 1]
            X_test_3d = [X_test[i, 0], X_test[i, 0]]
            f_3d = [F1_list[l][i, 0], f_sharp_list[k][i, 0]]
            alpha_3d = Pi[l, k]
            ax.plot3D(layer_3d, X_test_3d, f_3d, c=(0.8, l*0.3 + 0.2, 1 - l*0.3), alpha=alpha_3d)

    plt.legend()

    # Notice: Give the 2D plot
    #   The original data
    figure_2d = plt.figure(2)
    plt.scatter(x, y_11, marker='+', c='r', label='y_11', alpha=0.5, s=42)
    plt.scatter(x, y_12, marker='+', c='r', label='y_12', alpha=0.5, s=42)
    plt.scatter(x, y_13, marker='+', c='r', label='y_13', alpha=0.5, s=42)

    plt.scatter(x, y_21, marker='x', c='b', label='y_21', alpha=0.5, s=42)
    plt.scatter(x, y_22, marker='x', c='b', label='y_22', alpha=0.5, s=42)
    plt.scatter(x, y_23, marker='x', c='b', label='y_23', alpha=0.5, s=42)

    # Notice: Visualize the learned kernel
    diag = np.sqrt(np.diag(K_1))
    diag2 = np.sqrt(np.diag(K_2))
    # plt.plot(X_test, mu_1, c='r', label='Mean')
    plt.fill_between(X_test[:, 0], (mu_1 + diag**0.5)[:, 0], (mu_1 - diag**0.5)[:, 0], alpha=0.2, color='pink')

    # plt.plot(X_test, mu_2, c='b', label='Mean')
    plt.fill_between(X_test[:, 0], (mu_2 + diag2**0.5)[:, 0], (mu_2 - diag2**0.5)[:, 0], alpha=0.2, color='aqua')

    # plt.plot(x, y_11, c='r', label='y_11', alpha=0.5, markersize=12)
    # plt.plot(x, y_12, c='r', label='y_12', alpha=0.5, markersize=12)
    # plt.plot(x, y_13, c='r', label='y_13', alpha=0.5, markersize=12)
    # plt.plot(x, y_21, c='b', label='y_21', alpha=0.5, markersize=12)
    # plt.plot(x, y_22, c='b', label='y_22', alpha=0.5, markersize=12)
    # plt.plot(x, y_23, c='b', label='y_23', alpha=0.5, markersize=12)

    #  Notice: The mapped data by FOT
    plt.plot(X_test, f_sharp_11, c='orange', label='FOT T#y_11', alpha=0.9)
    plt.plot(X_test, f_sharp_12, c='orange', label='FOT T#y_12', alpha=0.9)
    plt.plot(X_test, f_sharp_13, c='orange', label='FOT T#y_13', alpha=0.9)

    # notice: The mapped data by GPOT
    # plt.plot(X_test, f_shp_11_gpot, c='violet', label='GPOT T#y_11', alpha=0.7)
    # plt.plot(X_test, f_shp_12_gpot, c='violet', label='GPOT T#y_12', alpha=0.7)
    # plt.plot(X_test, f_shp_13_gpot, c='violet', label='GPOT T#y_13', alpha=0.7)
    plt.legend()
    plt.show()













