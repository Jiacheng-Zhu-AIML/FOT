"""
Lol
sigh

The experiment that shows
The index-invariant property
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

# Notice:
#   Evaluate from different index sets
#   from random continuous functions
def generate_curves(line_num, para_list, x_np_list):

    amp_mean = para_list[0]
    amp_var = para_list[1]

    slop_mean = para_list[2]
    slop_var = para_list[3]

    scale_mean = para_list[4]
    scale_var = para_list[5]

    mix_mean = para_list[6]
    mix_var = para_list[7]

    # Notice: Get the number of domain
    domain_num = len(x_np_list)
    data_list_domain = [[] for i in range(domain_num)]

    for traj in range(line_num):
        amp = np.random.normal(loc=amp_mean, scale=amp_var)
        slop = np.random.normal(loc=slop_mean, scale=slop_var)
        scale = np.random.normal(loc=scale_mean, scale=scale_var)
        mix = np.random.normal(loc=mix_mean, scale=mix_var)
        for domain_id, x_domain in enumerate(x_np_list):
            y = amp * np.sin(scale * x_domain) + mix + slop * x_domain
            data_list_domain[domain_id].append(y)

    return data_list_domain


# Notice: directly learn the kernels
def get_UV_from_x(F_1_list_in, F1_x_list_in, F_2_list_in, F2_x_list_in, X_test):

    noise = 0.2
    GP_1 = GP_model(x_train=np.concatenate(F1_x_list_in, axis=0),
                    y_train=np.concatenate(F_1_list_in, axis=0), noise=noise)
    GP_1.train()
    _, K_1 = GP_1.predict(x_test=X_test)

    GP_2 = GP_model(x_train=np.concatenate(F2_x_list_in, axis=0),
                    y_train=np.concatenate(F_2_list_in, axis=0), noise=noise)
    GP_2.train()
    _, K_2 = GP_2.predict(x_test=X_test)

    # Notice:
    #   Start to work on OUR PROPOSED METHOD!
    U, D1, UT = np.linalg.svd(K_1)
    V, D2, VT = np.linalg.svd(K_2)  # Notice: V and U and then fixed
    return U, V


if __name__ == "__main__":

    # Notice:
    #   [amp_mean, amp_var, slop_mean, slop_var, scale_mean, scale_var, mix_mean, mix_var]

    np.random.seed(14)

    param_list_src = [0.6, 0.1, 0.01, 0.02, 0.3, 0.05, 1.5, 0.9]
    param_list_tgt = [1, 0.1, -0.05, 0.05, 1.4, 0.02, 0.8, 0.1]

    l_num = 16

    data_len = train_data_len = 40  # notice: 40
    ood_data_len = 30
    x_np_global = np.linspace(-15, 15, 150).reshape(-1, 1)
    x_np_train = np.linspace(-10, 10, train_data_len).reshape(-1, 1)
    x_np_ood = np.linspace(5, 15, ood_data_len).reshape(-1, 1)    # Notice: Out of domain
    x_input_list = [x_np_global, x_np_train, x_np_ood]

    # Notice: Parameters for plotting
    plot_x_low = -16
    plot_x_high = 16
    plot_y_low = -3
    plot_y_high = 5


    # Notice: Generate source data
    data_domain_list = generate_curves(line_num=l_num, para_list=param_list_src, x_np_list=x_input_list)
    y_src_np_list_gloabl = data_domain_list[0]
    F_1_list = data_domain_list[1]
    F_1_list_ood = data_domain_list[2]

    # Notice: Generate target data
    k_num = 16
    data_tgt_domain_list = generate_curves(line_num=k_num, para_list=param_list_tgt, x_np_list=x_input_list)
    y_tgt_np_list_gloabl = data_tgt_domain_list[0]
    F_2_list = data_tgt_domain_list[1]
    F_2_list_ood = data_tgt_domain_list[2]

    # Notice: out-of-domian-out-of-sample
    #     oos_data_len = 30
    #     x_np_oos = np.linspace(-13, 5, oos_data_len).reshape(-1, 1)
    oos_data_len = 30
    x_np_oos = np.linspace(-13, 5, oos_data_len).reshape(-1, 1)
    x_input_list_oos = [x_np_oos]
    oos_num = 10
    # Notice: Generate source data
    data_src_domain_list_oos = generate_curves(line_num=oos_num, para_list=param_list_src, x_np_list=x_input_list_oos)
    F_1_list_oos = data_src_domain_list_oos[0]

    data_tgt_domain_list_oos = generate_curves(line_num=oos_num, para_list=param_list_tgt, x_np_list=x_input_list_oos)
    F_2_list_oos = data_tgt_domain_list_oos[0]



    # Notice: FOT mapping
    # Notice: Prepare to train the GP
    F1_x_list = [x_np_train for i in range(l_num)]
    F2_x_list = [x_np_train for i in range(k_num)]
    U, V = get_UV_from_x(F_1_list, F1_x_list, F_2_list, F2_x_list, x_np_train)

    # Notice: *********Use the optimizer*********
    GFOT_optimizer = GFOT_optimization(F1_list=F_1_list, F2_list=F_2_list,
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
    gamma_h = 40  #
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

    # Notice: ################Conduct the Mapping########################

    # Notice: Conduct the GFOT
    GFOT_f_shp_list = []
    for f1 in F_1_list:
        f1_sharp_fot = V @ A_mat @ U.T @ f1
        GFOT_f_shp_list.append(f1_sharp_fot)

    print()
    print('np.sum(Pi) =', np.sum(Pi))

    # Notice: Plot the first figure
    fig_1 = plt.figure(1, figsize=(12, 6))

    plot_functions(plt, x_np_global, y_src_np_list_gloabl, "All source function space", "r", 0.2, linewidth=2)
    plot_functions(plt, x_np_global, y_tgt_np_list_gloabl, "All taget function space", "b", 0.2, linewidth=2)

    plot_functions(plt, x_np_train, F_1_list,
                   "Source function train", "r", 0.9, linewidth=2, marker="o", markersize=4)
    plot_functions(plt, x_np_train, F_2_list,
                   "target function train", "b", 0.9, linewidth=2, marker="s", markersize=4)

    plot_functions(plt, x_np_train, GFOT_f_shp_list, "FOT", "darkorange", 0.9, linewidth=3)

    plt.legend(loc="upper left", prop={'size': 12})
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    fig_1.tight_layout(pad=0.1)


    # Notice: Out of domain (in the sample)

    F1_x_list_ood = [x_np_ood for i in range(l_num)]
    F2_x_list_ood = [x_np_ood for i in range(k_num)]

    noise = 0.2

    # U_ood, V_ood = get_UV_from_x(F_1_list_ood, F1_x_list_ood, F_2_list_ood, F2_x_list_ood, x_np_ood)

    GP_1_ood = GP_model(x_train=np.concatenate(F1_x_list_ood, axis=0),
                    y_train=np.concatenate(F_1_list_ood, axis=0), noise=noise)
    GP_1_ood.train()
    _, K_1_ood = GP_1_ood.predict(x_test=x_np_ood)

    GP_2_ood = GP_model(x_train=np.concatenate(F2_x_list_ood, axis=0),
                    y_train=np.concatenate(F_2_list_ood, axis=0), noise=noise)
    GP_2_ood.train()
    _, K_2_ood = GP_2_ood.predict(x_test=x_np_ood)

    # Notice:
    #   Start to work on OUR PROPOSED METHOD!
    U_ood, _, _ = np.linalg.svd(K_1_ood)
    V_ood, _, _ = np.linalg.svd(K_2_ood)  # Notice: V and U and then fixed


    A_mat_ood = A_mat[0:ood_data_len, 0:ood_data_len]

    GFOT_f_shp_ood_list = []
    for f1_ood in F_1_list_ood: # Notice something wrong????
        f1_sharp_fot_ood = V_ood @ A_mat_ood @ U_ood.T @ f1_ood
        # f1_sharp_fot_ood = f1_ood
        GFOT_f_shp_ood_list.append(f1_sharp_fot_ood)

    # Notice: Plot ood(in sample)
    fig_2 = plt.figure(2, figsize=(12, 6))
    plot_functions(plt, x_np_global, y_src_np_list_gloabl, "All source function space", "r", 0.2, linewidth=2)
    plot_functions(plt, x_np_global, y_tgt_np_list_gloabl, "All taget function space", "b", 0.2, linewidth=2)

    plot_functions(plt, x_np_ood, F_1_list_ood,
                   "Source function out-of-domain", "salmon", 0.9, linewidth=2, marker="o", markersize=4)
    plot_functions(plt, x_np_ood, F_2_list_ood,
                   "target function out-of-domain", "skyblue", 0.9, linewidth=2, marker="s", markersize=4)

    plot_functions(plt, x_np_ood, GFOT_f_shp_ood_list, "FOT out-of-domain", "gold", 0.9, linewidth=3)

    plt.legend(loc="upper left", prop={'size': 10})
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    fig_2.tight_layout(pad=0.1)


    # Notice: out-of-domian-out-of-sample
    F1_x_list_oos = [x_np_oos for i in range(oos_num)]
    F2_x_list_oos = [x_np_oos for i in range(oos_num)]
    U_oos, V_oos = get_UV_from_x(F_1_list_oos, F1_x_list_oos, F_2_list_oos, F2_x_list_oos, x_np_oos)
    A_mat_oos = A_mat[0:oos_data_len, 0:oos_data_len]
    GFOT_f_shp_oos_list = []
    for f1_oos in F_1_list_oos:  # Notice something wrong????
        f1_sharp_fot_oos = V_oos @ A_mat_oos @ U_oos.T @ f1_oos
        # f1_sharp_fot_ood = f1_ood
        GFOT_f_shp_oos_list.append(f1_sharp_fot_oos)

    # Notice: plot out-of-domian-out-of-sample (oos)
    fig_3 = plt.figure(3, figsize=(12, 6))
    plot_functions(plt, x_np_global, y_src_np_list_gloabl, "All source function space", "r", 0.2, linewidth=2)
    plot_functions(plt, x_np_global, y_tgt_np_list_gloabl, "All taget function space", "b", 0.2, linewidth=2)

    plot_functions(plt, x_np_oos, F_1_list_oos,
                   "Source function oos", "brown", 0.9, linewidth=2, marker="o", markersize=4)
    plot_functions(plt, x_np_oos, F_2_list_oos,
                   "target function oos", "navy", 0.9, linewidth=2, marker="s", markersize=4)

    plot_functions(plt, x_np_oos, GFOT_f_shp_oos_list, "FOT oos", "olive", 0.9, linewidth=3)

    plt.legend(loc="upper left", prop={'size': 10})
    plt.xlim([plot_x_low, plot_x_high])
    plt.ylim([plot_y_low, plot_y_high])
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    fig_3.tight_layout(pad=0.1)


    # Notice: Plot all the parameters
    fig_A = plt.figure(666)
    plt.imshow(A_mat)
    plt.colorbar()
    plt.title("A_mat")

    fig_U = plt.figure(7771)
    plt.imshow(U)
    plt.colorbar()
    plt.title("U")
    fig_V = plt.figure(7772)
    plt.imshow(V)
    plt.colorbar()
    plt.title("V")

    fig_U_ood = plt.figure(7773)
    plt.imshow(U_ood)
    plt.colorbar()
    plt.title("U_ood")
    fig_V_ood = plt.figure(7774)
    plt.imshow(V_ood)
    plt.colorbar()
    plt.title("V_ood")

    plt.show()
