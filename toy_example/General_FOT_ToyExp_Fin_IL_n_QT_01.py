'''
Toy examples of the Generic Functional Optimal Transport

Illustrative (IL)
Appear on the paper!!!
Stochastic Toy Examples: Multiple mixture of sine to Multiple mixture of sine

1. Multimodal
2. Compress Illustration


IL&QT 01:
Appear on the paper
Stochastic Toy examples: One Mixture of Sines to 2 Mixture of Sines

The data generation parameters are finalized

GPOT,
LSOT,
FOT(Diag),
FOT,
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits import mplot3d
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
import scipy
import pickle


from WGPOT.wgpot import expmap, logmap

# Notice: Self defined functions
from General_Integral_GP_test import GP_model, data_domain_1, data_domain_2

from General_Functional_OT_Optimization import plot_origin_domain_data
from General_Functional_OT_Optimization import GFOT_optimization, plot_functions, \
    loss_l2_average, Generate_Sine_Mixture, plot_origin_domain_data_line

from LSOT_StochasticOTDiscrete import PyTorchStochasticDiscreteOT

import sys
sys.path.append("..")
from fot.GFOT_Solver_HS import GFOT_optimization, sinkhorn_plan, loss_weighted_l2_average, \
    loss_l2_Wasserstein


if __name__ == '__main__':

    gfot_nmlzd_loss_list = []
    gfot_diag_nmlzed_loss_list = []
    gpot_nmlzed_loss_list = []
    lsot_nmlzed_loss_list = []

    test_time = 1

    for t in range(test_time):
        # Notice: Create the test dataset
        data_len = 20  # Notice, this should be consistent all through the process
        x_start = -5
        x_end = 5
        x = np.linspace(x_start, x_end, data_len).reshape(-1, 1)

        # Notice: Sample the source domain data
        #   Generated from a mixture of GP-like functions
        #   GP-like functions are sine functions with noise
        l_num = 16

        # mix_ctr_list_1 = [-1]
        # mix_var_list_1 = [0.3]
        # sine_scale_list_1 = [0.5]
        # sine_scale_var_list_1 = [0.2]
        # sine_amp_list_1 = [0.5]
        # sine_amp_var_list_1 = [0.2]
        # sine_shift_list_1 = [1]
        # sine_shift_var_list_1 = [0.5]

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

        k_num = 16

        # mix_ctr_list_2 = [1]
        # mix_var_list_2 = [2]
        # sine_scale_list_2 = [1.0]
        # sine_scale_var_list_2 = [1.0]
        # sine_amp_list_2 = [0.4]
        # sine_amp_var_list_2 = [0.3]
        # sine_shift_list_2 = [1]
        # sine_shift_var_list_2 = [0.5]

        mix_ctr_list_2 = [-1.8, 1.8]
        mix_var_list_2 = [0.3, 0.3]
        sine_scale_list_2 = [0.85, 1.5]
        sine_scale_var_list_2 = [0.5, 0.5]
        sine_amp_list_2 = [0.5, 0.5]
        sine_amp_var_list_2 = [0.2, 0.2]
        sine_shift_list_2 = [1, 0.0]
        sine_shift_var_list_2 = [0.4, 0.4]

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
        gamma_h = 40    #
        gamma_A = 0.001
        gamma_power = -20
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

        xsf_reshaped = xsf.reshape((-1, data_len))  # (l_num, data_len)

        # Notice: ################Conduct the Mapping########################

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


        # Notice: Compute the loss
        gfot_loss = loss_weighted_l2_average(F2_list, GFOT_f_shp_list, Pi)
        gfot_loss_nmlzd = gfot_loss / data_len
        print('gfot_loss_nmlzd =', gfot_loss_nmlzd)

        gfot_diag_loss = loss_weighted_l2_average(F2_list, GFOT_Diag_f_shp_list, Pi)
        gfot_diag_loss_nmlzd = gfot_diag_loss / data_len
        print('gfot_diag_loss_nmlzd =', gfot_diag_loss_nmlzd)

        # Notice: Compute the loss with the Sinkhorn coupling
        gp_coupling = np.ones_like(Pi) / l_num
        gpot_loss = loss_weighted_l2_average(F2_list, GPOT_f_shp_list, gp_coupling)
        gpot_loss_nmlzd = gpot_loss / data_len
        print('gpot_loss_nmlzd =', gpot_loss_nmlzd)
        gpot_w_loss = loss_l2_Wasserstein(F2_list, GPOT_f_shp_list, lam=1/gamma_h, epsilon=1e-3)
        gpot_w_loss_nmlzd = gpot_w_loss / data_len
        print("gpot_w_loss_nmlzd =", gpot_w_loss_nmlzd)
        print()

        lsot_loss = loss_weighted_l2_average(F2_list, LSOT_f_shp_list, gp_coupling)
        lsot_loss_nmlzd = lsot_loss / data_len
        print('lsot_loss_nmlzd =', lsot_loss_nmlzd)
        lsot_w_loss = loss_l2_Wasserstein(F2_list, LSOT_f_shp_list, lam=1/gamma_h, epsilon=1e-3)
        lsot_w_loss_nmlzd = lsot_w_loss / data_len
        print("lsot_w_loss_nmlzd =", lsot_w_loss_nmlzd)

        print()
        print('np.sum(Pi) =', np.sum(Pi))
        print('np.sum(gp_coupling) =', np.sum(gp_coupling))

        gfot_nmlzd_loss_list.append(gfot_loss_nmlzd)
        gfot_diag_nmlzed_loss_list.append(gfot_diag_loss_nmlzd)
        gpot_nmlzed_loss_list.append(gpot_w_loss_nmlzd)
        lsot_nmlzed_loss_list.append(lsot_w_loss_nmlzd)


    gfot_loss_mean = np.mean(gfot_nmlzd_loss_list)
    gfot_loss_var = np.var(gfot_nmlzd_loss_list)

    gfot_diag_loss_mean = np.mean(gfot_diag_nmlzed_loss_list)
    gfot_diag_loss_var = np.var(gfot_diag_nmlzed_loss_list)

    gpot_loss_mean = np.mean(gpot_nmlzed_loss_list)
    gpot_loss_var = np.var(gpot_nmlzed_loss_list)

    lsot_loss_mean = np.mean(lsot_nmlzed_loss_list)
    lsot_loss_var = np.var(lsot_nmlzed_loss_list)

    print()
    print('gfot_loss_mean =', gfot_loss_mean)
    print('gfot_loss_var =', gfot_loss_var)
    print()
    print('gfot_diag_loss_mean =', gfot_diag_loss_mean)
    print('gfot_diag_loss_var =', gfot_diag_loss_var)
    print()
    print('gpot_w_loss_mean =', gpot_loss_mean)
    print('gpot_w_loss_var =', gpot_loss_var)
    print()
    print('lsot_w_loss_mean =', lsot_loss_mean)
    print('lsot_w_loss_var =', lsot_loss_var)

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

    plot_origin_domain_data_line(plt, x, F1_list, marker=None,
                                color='r', label='Source', alpha=0.99,
                                 linewidth=3, markersize=5)

    plot_origin_domain_data_line(plt, x, F2_list, marker=None,
                                color='b', label='Target', alpha=0.99,
                                 linewidth=3, markersize=5)

    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.legend(loc="upper left", prop={'size': 20})
    plt.xticks([])
    # plt.xlabel("x")
    plt.yticks([])
    # plt.title("Target and source sample functions")
    figure_first.tight_layout(pad=0.1)
    plt.savefig("ToyExp_IL_figures/toy_exp_IL_1_data.png", dpi=300)

    # Notice: 2D plot
    #   "The second one"
    #   Plot the FOT result
    figure_second = plt.figure(12)

    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label=None, alpha=0.15, s=6)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label=None, alpha=0.13, s=6)

    #  Notice: The mapped data by FOT
    plot_functions(plt, X_test, GFOT_f_shp_list, "FOT", "darkorange", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper left", prop={'size': 20})
    figure_second.tight_layout(pad=0.1)
    plt.savefig("ToyExp_IL_figures/toy_exp_IL_1_FOT.png", dpi=300)

    # Notice: 2D plot
    #   "The Third one"
    #   Plot the FOT (Diag) result
    figure_third = plt.figure(13)

    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label=None, alpha=0.15, s=6)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label=None, alpha=0.13, s=6)

    plot_functions(plt, X_test, GFOT_Diag_f_shp_list, "FOT(Diag)", "tan", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper left", prop={'size': 20})
    figure_third.tight_layout(pad=0.1)
    plt.savefig("ToyExp_IL_figures/toy_exp_IL_1_FOT_diag.png", dpi=300)

    # Notice: 2D plot
    #   "The Forth one"
    #   Plot the GPOT result
    figure_forth = plt.figure(14)

    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label=None, alpha=0.15, s=6)

    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label=None, alpha=0.13, s=6)

    plot_functions(plt, X_test, GPOT_f_shp_list, "GPOT", "Green", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper left", prop={'size': 20})
    figure_forth.tight_layout(pad=0.1)
    plt.savefig("ToyExp_IL_figures/toy_exp_IL_1_GPOT.png", dpi=300)

    # Notice: 2D plot
    #   "The Fifth one"
    #   Plot the LSOT result
    figure_fifth = plt.figure(15)
    plot_origin_domain_data(plt, x, F1_list, marker='s',
                            color='r', label=None, alpha=0.15, s=6)
    plot_origin_domain_data(plt, x, F2_list, marker='o',
                            color='b', label=None, alpha=0.13, s=6)

    plot_functions(plt, X_test, LSOT_f_shp_list, "LSOT", "violet", 0.9, linewidth=3)
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper left", prop={'size': 20})
    figure_fifth.tight_layout(pad=0.1)
    plt.savefig("ToyExp_IL_figures/toy_exp_IL_1_LSOT.png", dpi=300)


    # Notice: Visualize the Pi
    # fig_m = plt.figure(10)
    # plt.imshow(Pi)
    # plt.colorbar()
    plt.show()













