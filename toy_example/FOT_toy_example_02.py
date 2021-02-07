"""
FOT
Functional Optimal Transport:
Mapping Estimation and Domain Adaptation for Functional Data
Jiacheng Zhu
jzhu4@andrew.cmu.edu
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.linalg import inv

import sys
sys.path.append("..")

from fot.FOT_Solver import GFOT_optimization, sinkhorn_plan
from fot.fot_utils import loss_weighted_l2_average, loss_l2_Wasserstein, \
    plot_origin_domain_data, plot_functions, Generate_Sine_Mixture, plot_origin_domain_data_line
from fot.learn_kernel import GP_model

from WGPOT.wgpot import expmap, logmap
from benchmarks.LSOT_StochasticOTDiscrete import PyTorchStochasticDiscreteOT


if __name__ == '__main__':

    # Notice: Create the test dataset
    data_len = 20  # Notice, this should be consistent all through the process
    x_start = -5
    x_end = 5
    x = np.linspace(x_start, x_end, data_len).reshape(-1, 1)

    # Notice: Sample the source domain data
    #   Generated from a mixture of GP-like functions
    #   GP-like functions are sine functions with noise
    l_num = 16

    mix_ctr_list_1 = [-1]
    mix_var_list_1 = [1.8]
    sine_scale_list_1 = [0.4]
    sine_scale_var_list_1 = [0.2]
    sine_amp_list_1 = [0.5]
    sine_amp_var_list_1 = [0.2]
    sine_shift_list_1 = [0.8]
    sine_shift_var_list_1 = [0.5]

    F1_list, F1_x_list = Generate_Sine_Mixture(mix_center_list=mix_ctr_list_1, mix_var_list=mix_var_list_1,
                                 sine_scale_list=sine_scale_list_1, sine_scale_var_list=sine_scale_var_list_1,
                                 sine_amp_list=sine_amp_list_1, sine_amp_var_list=sine_amp_var_list_1,
                                 sine_shift_list=sine_shift_list_1, sine_shift_var_list=sine_shift_var_list_1,
                                 x_list=x, traj_num=l_num, mix_type='uniform')

    k_num = 16
    mix_ctr_list_2 = [-1, 1]
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
    ini_Pi = 0.9 * 1.0 / (l_num * k_num) * np.ones((l_num, k_num))
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
    # gamma_power = -10
    gamma_power = -10
    l_power = 3
    GFOT_optimizer.Set_Parameters(rho_k=rho_k, rho_l=rho_l, rho_i=rho_i,
                                  gamma_A=gamma_A, gamma_h=gamma_h,
                                  gamma_power=gamma_power, l_power=l_power)

    # Notice: Do the optimization
    lr_A = 0.0004
    lr_Pi = 0.00001
    ite_num = 1000
    A_mat, Pi = GFOT_optimizer.Optimize(lr_A=lr_A, lr_Pi=lr_Pi, tho=1e-5,
                                        diagonal=False, max_iteration=ite_num,
                                        entropy=True, fix_Pi=False,
                                        inequality=False)
    # Notice: Conduct the GFOT
    GFOT_f_shp_list = []
    for f1 in F1_list:
        f1_sharp_fot = V @ A_mat @ U.T @ f1
        GFOT_f_shp_list.append(f1_sharp_fot)

    # Notice: Conduct the GPOT
    v_mu, v_T = logmap(mu_2, K_2, mu_1, K_1)
    GPOT_f_shp_list = []
    for f1 in F1_list:
        f1_sharp_gpot, _ = expmap(f1, K_1, v_mu, v_T)
        GPOT_f_shp_list.append(f1_sharp_gpot)

    # Notice: Compute the loss
    gfot_loss = loss_weighted_l2_average(F2_list, GFOT_f_shp_list, Pi)
    gfot_loss_nmlzd = gfot_loss/data_len
    print('np.sum(Pi) =', np.sum(Pi))
    print('gfot_loss_nmlzd =', gfot_loss_nmlzd)

    gp_coupling = np.ones_like(Pi)/l_num
    gpot_loss = loss_weighted_l2_average(F2_list, GPOT_f_shp_list, gp_coupling)
    gpot_loss_nmlzd = gpot_loss/data_len
    print('np.sum(gp_coupling) =', np.sum(gp_coupling))
    print('gpot_loss_nmlzd =', gpot_loss_nmlzd)

    # Notice: #####################PLOT####################
    plot_x_low = x_start - 1
    plot_x_high = x_end + 1

    plot_y_low = -3
    plot_y_high = 3

    # Notice: 2D plot
    #   Just the Original Data
    #   "The first one"
    figure_first = plt.figure(11, figsize=(12, 3))
    gs = GridSpec(1, 11)

    ax1 = figure_first.add_subplot(gs[0, 0:3])
    plot_origin_domain_data_line(plt, x, F1_list, marker=None,
                                 color='r', label='Source', alpha=0.99,
                                 linewidth=2, markersize=5)
    plot_origin_domain_data_line(plt, x, F2_list, marker=None,
                                 color='b', label='Target', alpha=0.99,
                                 linewidth=2, markersize=5)

    ax1.set_xlim([plot_x_low, plot_x_high])
    ax1.set_ylim([plot_y_low, plot_y_high])
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Notice: 2D plot
    #   Plot the mapping,
    #   "The second one"
    # figure_second = plt.figure(12)
    ax2 = figure_first.add_subplot(gs[0, 3])
    # Get the source mean vector
    src_data = np.concatenate(F1_list, axis=1)
    mpd_data = np.concatenate(GFOT_f_shp_list, axis=1)
    # print('mpd_data.shape =', mpd_data.shape)   # (data_len, l_num)
    src_mean = np.mean(src_data, axis=0, keepdims=1)
    mpd_mean = np.mean(mpd_data, axis=0, keepdims=1)
    mapping_value = np.concatenate([src_mean, mpd_mean], axis=0).T    # (l_num, 2)

    for l in range(l_num):
        dplacm_y_start = mapping_value[l, 0]
        dplacm_y_end = mapping_value[l, 1]
        # plt.plot([0, 1], mapping_value[l, :], c='orange',
        #          linewidth=3, marker='>', markersize=10)
        ax2.arrow(x=0, y=dplacm_y_start, dx=1, dy=dplacm_y_end - dplacm_y_start,
                  color='orange', width=0.04, head_width=0.15, head_length=0.3)
        # ax2.plot([0, 1], mapping_value[l, :], c='orange',
        #         linewidth=3, marker='>', markersize=10)

    ax2.set_xlim([-0.1, 1.3])
    ax2.set_ylim([plot_y_low, plot_y_high])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Notice: Give the 2D plot
    #   The original data and the pushforward result
    #   "The third one"
    # figure_2d = plt.figure(13)
    ax3 = figure_first.add_subplot(gs[0, 4:7])
    plot_origin_domain_data(ax3, x, F1_list, marker='s',
                            color='r', label='Source', alpha=0.15, s=6)

    plot_origin_domain_data(ax3, x, F2_list, marker='o',
                            color='b', label='Target', alpha=0.15, s=6)

    #  Notice: The mapped data by FOT
    plot_functions(ax3, X_test, GFOT_f_shp_list, "FOT", "darkorange", 0.9, linewidth=3)
    # notice: The mapped data by GPOT
    # plot_functions(ax3, X_test, GPOT_f_shp_list, "GPOT", "lime", 0.4)

    # Notice: Visualize the learned kernel

    ax3.set_xlim([plot_x_low, plot_x_high])
    ax3.set_ylim([plot_y_low, plot_y_high])
    ax3.legend()
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = figure_first.add_subplot(gs[0, 7:9])
    ax4.imshow(Pi)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title(r"Coupling: $\Pi$")
    # ax4.colorbar()

    ax5 = figure_first.add_subplot(gs[0, 9:11])
    ax5.imshow(A_mat)
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_title(r"$\Lambda$ matrix")
    # ax5.colorbar()
    plt.tight_layout(pad=0.1)

    # Notice: Visualize the Pi
    # fig_m = plt.figure(10)
    # plt.imshow(Pi)
    # plt.colorbar()
    # plt.savefig("ToyExp_viz_matrix_figures/ToyExp_viz_mat_2.png", dpi=600)
    plt.show()