"""
Multi Dimensional Output FOT

Use Gpytorch to do the regression and get the Kernel Matrix

"""
import numpy as np
import scipy.linalg
import scipy
# from nd_GP_test import nd_kernel_function, logmap, expmap, GP_2d
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# from GP_byct_test import Wasserstein_GP
import pickle
import torch
import gpytorch
import dill
import math
import os


from WGPOT.wgpot import expmap, logmap

# Notice: Self defined functions
from General_Integral_GP_test import GP_model, data_domain_1, data_domain_2

from General_Functional_OT_Optimization import plot_origin_domain_data
from General_Functional_OT_Optimization import GFOT_optimization, plot_functions, Generate_MO_Sine
from General_Functional_OT_Optimization import plot_origin_domain_data
from General_Functional_OT_Optimization import GFOT_optimization, plot_functions, \
    loss_l2_average, loss_weighted_l2_average, Generate_Sine_Mixture, plot_origin_domain_data_line


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultiOutputGP:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        self.model = MultitaskGPModel(self.train_x, self.train_y, self.likelihood)

    def train(self, iteration=50):
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(iteration):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            self.optimizer.step()

        self.model.eval()
        self.likelihood.eval()

    def predict_np(self, test_x):
        with torch.no_grad():
            self.model.eval()
            self.likelihood.eval()

            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean
            cov_test = predictions.covariance_matrix
            lower, upper = predictions.confidence_region()

            mean_np = mean.numpy()  # (51, 2)
            # print('mean_np.shape =', mean_np.shape)
            lower_np = lower.numpy()  # (51, 2)
            # print('lower_np.shape =', lower_np.shape)
            upper_np = upper.numpy()  # (51,2)

            # print('cov_test =', cov_test)

            return mean_np, upper_np, lower_np, cov_test.detach().numpy()


if __name__ == '__main__':

    '''
    There are only 1 trajectories in the pkl?
    '''
    # data_address = 'pkl_files/MOGP_data.pkl'
    # data_file = open(data_address, 'rb')
    # mogp_data = pickle.load(data_file)
    #
    # print('len(mogp_data) =', len(mogp_data))
    #
    # # Notice: First deal with the first trajectory
    #
    # data = mogp_data[0][:2, :]
    #
    # print('data.shape =', data.shape)   # (2, 161)
    #
    # print('mogp_data[0].shape =', mogp_data[0].shape)
    #
    # # Notice: Important, need to normalize the data
    #
    # data = data - np.mean(data, axis=1, keepdims=1)

    data_len = 20  # Notice, this should be consistent all through the process
    t_start = -10
    t_end = 10
    t_list = np.linspace(t_start, t_end, data_len).reshape(-1, 1)

    l_num = 10

    mix_ctr_1_list = [0, 0]
    mix_var_list = [0.01, 0.01]
    sine_scale_list = [0.04, 0.04]
    sine_scale_var_list = [0.001, 0.001]
    sine_amp_list = [0.4, 0.5]
    sine_amp_var_list = [0.02, 0.02]
    sine_shift_list = [1, -1]
    sine_shift_var_list = [0.1, 0.1]

    Y_1_data_list, X_1_data_list = Generate_MO_Sine(mix_center_list=mix_ctr_1_list, mix_var_list=mix_var_list,
                                                sine_scale_list=sine_scale_list,
                                                sine_scale_var_list=sine_scale_var_list,
                                                sine_amp_list=sine_amp_list, sine_amp_var_list=sine_amp_var_list,
                                                sine_shift_list=sine_shift_list,
                                                sine_shift_var_list=sine_shift_var_list,
                                                t_list=t_list, traj_num=l_num)

    k_num = 10

    mix_ctr_1_list = [0, 0]
    mix_var_list = [0.01, 0.01]
    sine_scale_list = [0.04, 0.04]
    sine_scale_var_list = [0.001, 0.001]
    sine_amp_list = [0.4, 0.5]
    sine_amp_var_list = [0.02, 0.02]
    sine_shift_list_2 = [2, 0]         # Notice: Only need to change this
    sine_shift_var_list = [0.1, 0.1]

    Y_2_data_list, X_2_data_list = Generate_MO_Sine(mix_center_list=mix_ctr_1_list, mix_var_list=mix_var_list,
                                                sine_scale_list=sine_scale_list,
                                                sine_scale_var_list=sine_scale_var_list,
                                                sine_amp_list=sine_amp_list, sine_amp_var_list=sine_amp_var_list,
                                                sine_shift_list=sine_shift_list_2,
                                                sine_shift_var_list=sine_shift_var_list,
                                                t_list=t_list, traj_num=k_num)

    # Notice: Prepare the source and target dataset for training GP
    Y_src_np = np.concatenate(Y_1_data_list, axis=0)
    X_src_np = np.concatenate(X_1_data_list, axis=0)
    print('Y_src_np.shape =', Y_src_np.shape)
    print('X_src_np.shape =', X_src_np.shape)
    Y_src_torch = torch.tensor(Y_src_np).float()
    X_src_torch = torch.tensor(X_src_np).float()

    Y_tgt_np = np.concatenate(Y_2_data_list, axis=0)
    X_tgt_np = np.concatenate(X_2_data_list, axis=0)
    print('Y_tgt_np.shape =', Y_tgt_np.shape)
    print('X_tgt_np.shape =', X_tgt_np.shape)

    Y_tgt_torch = torch.tensor(Y_tgt_np).float()
    X_tgt_torch = torch.tensor(X_tgt_np).float()

    # Notice: Need to have the same target set (X_star)
    X_star_torch = torch.tensor(t_list).float()
    print('X_star_torch.size() =', X_star_torch.size())

    # Notice: GP regression for both source and target data
    mo_gp_src = MultiOutputGP(train_x=X_src_torch, train_y=Y_src_torch)
    mo_gp_src.train(iteration=50)

    mo_gp_tgt = MultiOutputGP(train_x=X_src_torch, train_y=Y_src_torch)
    mo_gp_src.train(iteration=50)

    # Notice: Get the Kernel matrix
    mean_np, upper_np, lower_np, K_1_np = mo_gp_src.predict_np(test_x=X_star_torch)

    print('mean_np.shape =', mean_np.shape)
    print('K_1_np.shape =', K_1_np.shape)

    mean_np_tgt, upper_np_tgt, lower_np_tgt, K_2_np = mo_gp_tgt.predict_np(test_x=X_star_torch)

    # Notice:
    #   Start to work on OUR PROPOSED METHOD!
    U, D1, UT = np.linalg.svd(K_1_np)
    V, D2, VT = np.linalg.svd(K_2_np)  # Notice: V and U and then fixed

    # Notice: Learn the map

    # Notice: *********Use the optimizer*********
    GFOT_optimizer = GFOT_optimization(F1_list=Y_1_data_list, F2_list=Y_2_data_list,
                                       V=V, U=U, l_num=l_num, k_num=k_num, data_len=data_len,
                                       Y_dim=2)

    # notice: Set initial values
    ini_A = np.eye(data_len * 2)
    ini_Pi = 1.0 / l_num * np.ones((l_num, k_num))
    # ini_Pi = np.eye(l_num)

    lbd_k = 0.1 * np.ones((k_num,))
    lbd_l = 0.1 * np.ones((l_num,))
    lbd_i = 0.1 * np.ones((l_num, k_num))
    s_mat = 0.1 * np.ones((l_num, k_num))

    GFOT_optimizer.Set_Initial_Variables(ini_A=ini_A, ini_Pi=ini_Pi,
                                         ini_lbd_k=lbd_k, ini_lbd_l=lbd_l,
                                         ini_lbd_i=lbd_i, s_mat=s_mat)

    # Notice: Set lagrangian parameters
    rho_k = 80 * np.ones((k_num,))
    rho_l = 80 * np.ones((l_num,))
    rho_i = 10
    gamma_h = 30
    gamma_A = 1
    gamma_power = -30
    l_power = 3
    GFOT_optimizer.Set_Parameters(rho_k=rho_k, rho_l=rho_l, rho_i=rho_i,
                                  gamma_A=gamma_A, gamma_h=gamma_h,
                                  gamma_power=gamma_power, l_power=l_power)

    # Notice: Do the optimization
    lr_A = 0.01
    lr_Pi = 0.0001
    ite_num = 500
    A_mat, Pi = GFOT_optimizer.Optimize(lr_A=lr_A, lr_Pi=lr_Pi, tho=1e-5,
                                        diagonal=False, max_iteration=ite_num,
                                        entropy=True, fix_Pi=False,
                                        inequality=False, multi_output=True)

    F_shp_list = []

    for data_1 in Y_1_data_list:
        data_temp = data_1.reshape((-1, 1))
        data_shp = V @ A_mat @ U.T @ data_temp
        data_2d = data_shp.reshape((-1, 2))
        F_shp_list.append(data_2d)

    # Notice: Plot
    #   Plot the origin data
    fig_1 = plt.figure(1)
    for data in Y_1_data_list:
        plt.plot(data[:, 0], data[:, 1], c='r')

    for data in Y_2_data_list:
        plt.plot(data[:, 0], data[:, 1], c='b')

    # Notice: Plot the mapped data
    fig_2 = plt.figure(2)
    for data in Y_1_data_list:
        plt.plot(data[:, 0], data[:, 1], c='r', alpha=0.3)

    for data in Y_2_data_list:
        plt.plot(data[:, 0], data[:, 1], c='b', alpha=0.3)

    for data in F_shp_list:
        plt.plot(data[:, 0], data[:, 1], c='orange', alpha=1.0)

    fig_3 = plt.figure(3)
    for data in F_shp_list:
        plt.plot(data[:, 0], data[:, 1], c='orange', alpha=1.0, linewidth=4)
    plt.show()

    # train_y = torch.tensor(data.T).float()      # (17, 2)
    #
    # train_x = torch.tensor(np.linspace(0, 1, train_y.size()[0]).reshape(-1, 1)).float()  # (17, 1)
    #
    # train_x_np = train_x.numpy()
    # train_y_np = train_y.numpy()
    #
    # # Notice: Teh target trajectory
    # data_tgt = mogp_data[1][:2, :]
    # data_tgt = data_tgt - np.mean(data_tgt, axis=1, keepdims=1)
    #
    # train_y_tgt = torch.tensor(data_tgt.T).float()
    # train_x_tgt = torch.tensor(np.linspace(0, 1, train_y_tgt.size()[0]).reshape(-1, 1)).float()  # (17, 1)
    #
    # train_x_tgt_np = train_x_tgt.numpy()
    # train_y_tgt_np = train_y_tgt.numpy()
    #
    # # Notice:
    # #   The same test X for both GP
    # test_x = torch.linspace(0, 1, 30)
    #
    # # Notice:
    # #   GP Regression for the (source) data
    # mo_gp = MultiOutputGP(train_x=train_x, train_y=train_y)
    # mo_gp.train(iteration=50)
    #
    # mean_np, upper_np, lower_np, cov_np = mo_gp.predict_np(test_x=test_x)
    #
    # print('mean_np.shape =', mean_np.shape)
    # print('cov_np.shape =', cov_np.shape)
    #
    # # Notice:
    # #   GP R for the target data
    # mo_gp_tgt = MultiOutputGP(train_x=train_x_tgt, train_y=train_y_tgt)
    # mo_gp_tgt.train(iteration=50)
    #
    # mean_np_tgt, upper_np_tgt, lower_np_tgt, cov_np_tgt = mo_gp_tgt.predict_np(test_x=test_x)
    #
    # # Notice:
    # f, ax = plt.subplots(1, 1, figsize=(8, 6))
    #
    # # Notice: Plot the original data
    # plt.scatter(train_y_np[:, 0], train_y_np[:, 1], label='Original src data')
    #
    # plt.scatter(train_y_tgt_np[:, 0], train_y_tgt_np[:, 1], label='Original tgt data')
    #
    # # Notice: Plot the predicted trajectory
    # plt.plot(mean_np[:, 0], mean_np[:, 1], c='b', label='Predicted Mean src')
    #
    # plt.plot(mean_np_tgt[:, 0], mean_np_tgt[:, 1], c='r', label='Predicted Mean tgt')
    #
    # n_e = mean_np.shape[0]
    # for i in range(n_e):
    #     ellipse = Ellipse((mean_np[i, 0], mean_np[i, 1]),
    #                       width=(upper_np[i, 0] - mean_np[i, 0]) * 2,
    #                       height=(upper_np[i, 1] - mean_np[i, 1]) * 2,
    #                       facecolor='blue', alpha=0.1)
    #     ax.add_patch(ellipse)
    #
    #     # ellipse = Ellipse((tgt_mean_x_array[i], tgt_mean_y_array[i]),
    #     #                   width=(tgt_upper_x_array[i] - tgt_mean_x_array[i]) * 2,
    #     #                   height=(tgt_upper_y_array[i] - tgt_mean_y_array[i]) * 2,
    #     #                   facecolor='r', alpha=0.1)
    #     # ax.add_patch(ellipse)
    #
    # # Notice: Apply the OT Mapping
    #
    # src_mean_vector = mean_np.reshape((-1, 1))
    #
    # tgt_mean_vector = mean_np_tgt.reshape((-1, 1))
    #
    # v_mu, v_T = logmap(tgt_mean_vector, cov_np_tgt,
    #                        src_mean_vector, cov_np)
    #
    # for t in [0.3, 0.5, 0.7, 1]:
    #     v_m_t = v_mu * t
    #     v_T_t = v_T * t
    #     q_x_mu, q_x_K = expmap(src_mean_vector, cov_np, v_m_t, v_T_t)
    #
    #     q_x_plot = q_x_mu.reshape((-1, 2))
    #
    #     plt.plot(q_x_plot[:, 0], q_x_plot[:, 1], c='orange', alpha=0.5)
    #
    #
    # plt.legend()
    # plt.show()