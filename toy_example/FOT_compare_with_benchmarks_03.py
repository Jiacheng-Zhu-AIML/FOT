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
import ot


import sys
sys.path.append("..")

from fot.FOT_Solver import GFOT_optimization, sinkhorn_plan
from fot.fot_utils import loss_weighted_l2_average, loss_l2_Wasserstein, \
    plot_origin_domain_data, plot_functions, Generate_Sine_Mixture, plot_origin_domain_data_line
from fot.learn_kernel import GP_model

from WGPOT.wgpot import expmap, logmap
from benchmarks.LSOT_StochasticOTDiscrete import PyTorchStochasticDiscreteOT

import ot


if __name__ == '__main__':

    # Notice: Create the test dataset
    data_len = 40  # Notice, this should be consistent all through the process
    x_start = -5
    x_end = 5
    x = np.linspace(x_start, x_end, data_len).reshape(-1, 1)

    # Notice: Sample the source domain data
    #   Generated from a mixture of GP-like functions
    #   GP-like functions are sine functions with noise
    l_num = 12

    mix_ctr_list_1 = [-0.5, 0.5]
    mix_var_list_1 = [0.2, 0.2]
    sine_scale_list_1 = [0.5, 0.5]
    sine_scale_var_list_1 = [0.2, 0.2]
    sine_amp_list_1 = [0.5, 0.5]
    sine_amp_var_list_1 = [0.2, 0.2]
    sine_shift_list_1 = [1, 1]
    sine_shift_var_list_1 = [0.5, 0.5]

    F1_list, F1_x_list = Generate_Sine_Mixture(mix_center_list=mix_ctr_list_1, mix_var_list=mix_var_list_1,
                                 sine_scale_list=sine_scale_list_1, sine_scale_var_list=sine_scale_var_list_1,
                                 sine_amp_list=sine_amp_list_1, sine_amp_var_list=sine_amp_var_list_1,
                                 sine_shift_list=sine_shift_list_1, sine_shift_var_list=sine_shift_var_list_1,
                                 x_list=x, traj_num=l_num, mix_type='uniform')

    k_num = 12
    mix_ctr_list_2 = [-1, 2]
    mix_var_list_2 = [0.3, 0.3]
    sine_scale_list_2 = [1.0, 1.0]
    sine_scale_var_list_2 = [1.0, 1.0]
    sine_amp_list_2 = [0.4, 0.4]
    sine_amp_var_list_2 = [0.3, 0.3]
    sine_shift_list_2 = [1, 1]
    sine_shift_var_list_2 = [0.5, 0.5]

    F2_list, F2_x_list = Generate_Sine_Mixture(mix_center_list=mix_ctr_list_2, mix_var_list=mix_var_list_2,
                                               sine_scale_list=sine_scale_list_2,
                                               sine_scale_var_list=sine_scale_var_list_2,
                                               sine_amp_list=sine_amp_list_2, sine_amp_var_list=sine_amp_var_list_2,
                                               sine_shift_list=sine_shift_list_2,
                                               sine_shift_var_list=sine_shift_var_list_2,
                                               x_list=x, traj_num=k_num, mix_type='uniform')

    # Notice: Prepare to train the GP
    X1_train = np.concatenate(F1_x_list, axis=0)
    Y1_train = np.concatenate(F1_list, axis=0)

    X2_train = np.concatenate(F2_x_list, axis=0)
    Y2_train = np.concatenate(F2_list, axis=0)

    # Notice:
    print("F1_list[0].shape =", F1_list[0].shape)   # (data_len, 1)
    print("Y1_train.shape =", Y1_train.shape)       # (data_len * l/k_num, 1)

    # Notice: ############### Mapping Estimation (Pointcloud)
    ot_mapping_linear = ot.da.MappingTransport(
        kernel="linear", mu=1e0, eta=1e-8, bias=True,
        max_iter=20, verbose=True)
    ot_mapping_linear.fit(Xs=Y1_train, Xt=Y2_train)
    transp_Xs_linear = ot_mapping_linear.transform(Xs=Y1_train)

    print("transp_Xs_linear.shape =", transp_Xs_linear.shape)

    # Notice: ############# Mapping Estimation (High-D vector)
    Y1_train_vect = np.concatenate(F1_list, axis=1).T
    Y2_train_vect = np.concatenate(F2_list, axis=1).T

    print("Y1_train_discrete.shape =", Y1_train_vect.shape)     # (data_len, data_num)

    vect_ot_mapping =  ot.da.MappingTransport(
        kernel="gaussian", mu=1e-1, eta=1e-8, bias=True,
        max_iter=20, verbose=True)
    vect_ot_mapping.fit(Xs=Y1_train_vect, Xt=Y2_train_vect)
    transp_Xs_nonlinear_vect = vect_ot_mapping.transform(Xs=Y1_train_vect).T

    print("transp_Xs_nonlinear_vect.shape =", transp_Xs_nonlinear_vect.shape)

    # exit()

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
    ini_Pi = 1.0 / (l_num * k_num) * np.ones((l_num, k_num))
    # ini_Pi = np.eye(l_num)

    lbd_k = 100 * np.ones((k_num,))
    lbd_l = 100 * np.ones((l_num,))
    lbd_i = 0.1 * np.ones((l_num, k_num))
    s_mat = 0.1 * np.ones((l_num, k_num))

    GFOT_optimizer.Set_Initial_Variables(ini_A=ini_A, ini_Pi=ini_Pi,
                                         ini_lbd_k=lbd_k, ini_lbd_l=lbd_l,
                                         ini_lbd_i=lbd_i, s_mat=s_mat)

    # Notice: Set lagrangian parameters
    rho_k = 800 * np.ones((k_num,))
    rho_l = 800 * np.ones((l_num,))
    rho_i = 10
    gamma_h = 40
    gamma_A = 0.001
    gamma_power = -10
    l_power = 3
    GFOT_optimizer.Set_Parameters(rho_k=rho_k, rho_l=rho_l, rho_i=rho_i,
                                  gamma_A=gamma_A, gamma_h=gamma_h,
                                  gamma_power=gamma_power, l_power=l_power)

    # Notice: Do the optimization
    lr_A = 0.0004
    lr_Pi = 0.00001
    ite_num = 500
    A_mat, Pi = GFOT_optimizer.Optimize(lr_A=lr_A, lr_Pi=lr_Pi, tho=1e-5,
                                        diagonal=False, max_iteration=ite_num,
                                        entropy=True, fix_Pi=False,
                                        inequality=False)

    # Notice: #####################Do the FOT (Diag)####################
    GFOT_optimizer_Diag = GFOT_optimization(F1_list=F1_list, F2_list=F2_list,
                                       V=V, U=U, l_num=l_num, k_num=k_num, data_len=data_len)

    # Notice: Use the same initial
    GFOT_optimizer_Diag.Set_Initial_Variables(ini_A=ini_A, ini_Pi=ini_Pi,
                                         ini_lbd_k=lbd_k, ini_lbd_l=lbd_l,
                                         ini_lbd_i=lbd_i, s_mat=s_mat)

    # Notice: Use the same parameters
    GFOT_optimizer_Diag.Set_Parameters(rho_k=rho_k, rho_l=rho_l, rho_i=rho_i,
                                  gamma_A=gamma_A, gamma_h=gamma_h,
                                  gamma_power=gamma_power, l_power=l_power)

    A_mat_daig, Pi_diag = GFOT_optimizer_Diag.Optimize(lr_A=lr_A, lr_Pi=lr_Pi, tho=1e-5,
                                        diagonal=True, max_iteration=ite_num,
                                        entropy=True, fix_Pi=False,
                                        inequality=False)

    # Notice: ################# Do the LSOT ###########################
    index_src = np.concatenate(F1_x_list)
    n_my = l_num * data_len
    src_X = np.concatenate(F1_list)
    src_w = np.ones(n_my, ) / n_my

    tgt_X = np.concatenate(F2_list)
    tgt_w = np.ones(n_my, ) / n_my

    # Notice: Dual OT Stochastic Optimization (alg.1 of ICLR 2018 paper "Large-Scale Optimal Transport and Mapping Estimation")
    reg_val = 0.02
    reg_type = 'l2'
    device_type = 'cpu'
    device_index = 0

    discreteOTComputer = PyTorchStochasticDiscreteOT(src_X, src_w, tgt_X, tgt_w, reg_type, reg_val, device_type=device_type,
                                                     device_index=device_index)
    history = discreteOTComputer.learn_OT_dual_variables(epochs=200, batch_size=50, lr=0.0005)

    # Compute the reg-OT objective
    d_stochastic = discreteOTComputer.compute_OT_MonteCarlo(epochs=20, batch_size=50)

    # Learn Barycentric Mapping (alg.2 of ICLR 2018 paper "Large-Scale Optimal Transport and Mapping Estimation")
    bp_history = discreteOTComputer.learn_barycentric_mapping(epochs=300, batch_size=50, lr=0.000002)
    xsf = discreteOTComputer.evaluate_barycentric_mapping(src_X)

    # print('xsf.shape = ', xsf.shape)

    xsf_reshaped = xsf.reshape((-1, data_len))  # (l_num, data_len)

    #   Notice:
    #       ################Conduct the Mapping########################
    #       ###########################################################

    # Notice: Conduct the GFOT
    GFOT_f_shp_list = []
    for f1 in F1_list:
        f1_sharp_fot = V @ A_mat @ U.T @ f1
        GFOT_f_shp_list.append(f1_sharp_fot)

    # Notice: Conduct the GFOT(Diag)
    GFOT_Diag_f_shp_list = []
    for f1 in F1_list:
        f1_Diag_sharp_fot = V @ A_mat_daig @ U.T @ f1
        GFOT_Diag_f_shp_list.append(f1_Diag_sharp_fot)

    # Notice: Conduct the LSOT
    LSOT_f_shp_list = []
    for l in range(l_num):
        LSOT_f_shp_list.append(xsf_reshaped[l, :])

    # Notice: Conduct the GPOT
    v_mu, v_T = logmap(mu_2, K_2, mu_1, K_1)
    GPOT_f_shp_list = []
    for f1 in F1_list:
        f1_sharp_gpot, _ = expmap(f1, K_1, v_mu, v_T)
        GPOT_f_shp_list.append(f1_sharp_gpot)

    # Notice: Conduct the Mapping estimation
    map_est_list = []
    for l in range(l_num):
        map_est_list.append(transp_Xs_linear[l * data_len:l * data_len + data_len, :])

    # Notice: conduct the mapping estimation (highD vector)
    map_est_vector_list = []
    for l in range(l_num):
        map_est_vector_list.append(transp_Xs_nonlinear_vect[:, l])

    # Notice: Compute the loss

    gfot_loss = loss_weighted_l2_average(F2_list, GFOT_f_shp_list, Pi)
    gfot_loss_nmlzd = gfot_loss/data_len
    print('gfot_loss_nmlzd =', gfot_loss_nmlzd)

    gfot_diag_loss = loss_weighted_l2_average(F2_list, GFOT_Diag_f_shp_list, Pi)
    gfot_diag_loss_nmlzd = gfot_diag_loss / data_len
    print('gfot_diag_loss_nmlzd =', gfot_diag_loss_nmlzd)

    gp_coupling = np.ones_like(Pi)/l_num
    gpot_loss = loss_weighted_l2_average(F2_list, GPOT_f_shp_list, gp_coupling)
    gpot_loss_nmlzd = gpot_loss/data_len
    print('gpot_loss_nmlzd =', gpot_loss_nmlzd)

    lsot_loss = loss_weighted_l2_average(F2_list, LSOT_f_shp_list, gp_coupling)
    lsot_loss_nmlzd = lsot_loss/data_len
    print('lsot_loss_nmlzd =', lsot_loss_nmlzd)

    print()
    print('np.sum(Pi) =', np.sum(Pi))
    print('np.sum(gp_coupling) =', np.sum(gp_coupling))


    # Notice: #####################PLOT####################
    #   Here, illustrate the difference of 4 maps
    plot_x_low = x_start - 1
    plot_x_high = x_end + 1

    plot_y_low = -3
    plot_y_high = 3

    # Notice: 2D plot
    #   Just the Original Data
    #   "The first one"
    figure_first = plt.figure(11)
    figure_first.tight_layout()

    plot_origin_domain_data_line(plt, x, F1_list, marker='s',
                                color='r', label='F1, source', alpha=0.99,
                                 linewidth=1.5, markersize=5)

    plot_origin_domain_data_line(plt, x, F2_list, marker='o',
                                color='b', label='F2, target', alpha=0.99,
                                 linewidth=1.5, markersize=5)

    diag = np.sqrt(np.diag(K_1))
    diag2 = np.sqrt(np.diag(K_2))


    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend()

    # Notice: 2D plot
    #   "The second one"
    #   Plot the FOT result
    figure_second = plt.figure(12)

    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label='F1, source', alpha=0.3, s=10)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label='F2, target', alpha=0.3, s=10)

    #  Notice: The mapped data by FOT
    plot_functions(plt, X_test, GFOT_f_shp_list, "FOT", "orange", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend()

    # Notice: 2D plot
    #   "The Third one"
    #   Plot the FOT (Diag) result
    figure_third = plt.figure(13)

    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label='F1, source', alpha=0.3, s=10)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label='F2, target', alpha=0.3, s=10)

    plot_functions(plt, X_test, GFOT_Diag_f_shp_list, "FOT(Diag)", "tan", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend()

    # Notice: 2D plot
    #   "The Forth one"
    #   Plot the GPOT result
    figure_forth = plt.figure(14)

    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label='F1, source', alpha=0.3, s=10)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label='F2, target', alpha=0.3, s=10)

    plot_functions(plt, X_test, GPOT_f_shp_list, "GPOT", "Green", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend()

    # Notice: 2D plot
    #   "The Forth one"
    #   Plot the GPOT result
    figure_fifth = plt.figure(15)

    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label='F1, source', alpha=0.3, s=10)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label='F2, target', alpha=0.3, s=10)

    plot_functions(plt, X_test, LSOT_f_shp_list, "LSOT", "violet", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend()

    # Notice: 2D Plot : discrete mapping estimation
    fig_map_est = plt.figure(23333)
    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label='F1, source', alpha=0.3, s=10)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label='F2, target', alpha=0.3, s=10)

    plot_functions(plt, X_test, map_est_list, "Mapping_est", "green", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend()

    # Notice: discrete mapping estimation (HighD vector)
    fig_map_est_vec = plt.figure(23333666)
    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label='F1, source', alpha=0.3, s=10)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label='F2, target', alpha=0.3, s=10)

    plot_functions(plt, X_test, map_est_vector_list, "Mapping_est_vec", "teal", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend()


    # Notice: Visualize the Pi
    fig_m = plt.figure(10)
    plt.imshow(Pi)
    plt.colorbar()

    fig_A = plt.figure(999999)
    plt.imshow(A_mat)
    plt.colorbar()

    plt.show()













