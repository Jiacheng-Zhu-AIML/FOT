"""
FOT
Functional Optimal Transport:
Mapping Estimation and Domain Adaptation for Functional Data
Jiacheng Zhu
jzhu4@andrew.cmu.edu
"""
from matplotlib import pyplot as plt
import numpy as np
import orthopy
import itertools


# Notice: Generate Mixtures of Sine functions, 1D functions
def Generate_Continuous_Sine_Mixture(mix_center_list, mix_var_list,
                          sine_scale_list, sine_scale_var_list,
                          sine_amp_list, sine_amp_var_list,
                          sine_shift_list, sine_shift_var_list,
                          x_list, traj_num, mix_type='normal'):
    # Notice: Generate center
    if mix_type not in ['normal', 'uniform']:
        print("mix_type mus be 'normal' or 'uniform'. Changed to 'normal'. ")
        return [], []
    center_index_list = np.random.choice(len(mix_center_list), traj_num)
    print('center_index_list =', center_index_list)

    func_mean_list = []
    for center_index in center_index_list:

        if mix_type == 'normal':
            func_mean = np.random.normal(loc=mix_center_list[center_index],
                                         scale=mix_var_list[center_index])
        elif mix_type == 'uniform':
            low_temp = mix_center_list[center_index] - mix_var_list[center_index]
            high_temp = mix_center_list[center_index] + mix_var_list[center_index]
            func_mean = np.random.uniform(low=low_temp, high=high_temp)
        func_mean_list.append(func_mean)

    y_list = []
    y_mean_list = []
    x_data_list = []
    for j, func_mean in enumerate(func_mean_list):
        i = center_index_list[j]
        scale = np.random.normal(loc=sine_scale_list[i],scale=sine_scale_var_list[i])
        amp = np.random.normal(loc=sine_amp_list[i],scale=sine_amp_var_list[i])
        shift = np.random.normal(loc=sine_shift_list[i],scale=sine_shift_var_list[i])
        y = amp * np.sin(scale * x_list[j] + shift) + func_mean
        y_list.append(y)
        y_mean_list.append(np.mean(y))
        x_data_list.append(x_list[j])

    y_sorted = [y for _, y in sorted(zip(y_mean_list, y_list), reverse=True)]
    x_sorted = [x for _, x in sorted(zip(y_mean_list, x_data_list), reverse=True)]
    return y_sorted, x_sorted


# Notice: Generate a list of (different) index
def Generate_index_set(curve_num,
                       x_start_mu, x_end_mu, start_end_var,
                       index_num_mu, index_num_var,
                       pert_var):

    x_list = []
    for i in range(curve_num):
        x_start = np.random.normal(loc=x_start_mu, scale=start_end_var)
        x_end = np.random.normal(loc=x_end_mu, scale=start_end_var)
        index_num = int(np.random.normal(loc=index_num_mu, scale=index_num_var))
        x_for_pert = np.linspace(x_start, x_end, index_num).reshape(-1, 1)
        x_for_sort = x_for_pert + np.random.normal(loc=0, scale=pert_var, size=(index_num, 1))
        x_for_sort.sort()
        # print("x =", x_for_sort)
        x_list.append(x_for_sort)

    return x_list


def plot_origin_domain_data_line(plt, x_list, data_list, marker, color, label,
                                 alpha=0.5, linewidth=5, markersize=12):
    for i, data in enumerate(data_list):
        if i == 0:
            plt.plot(x_list[i], data, marker=marker, c=color, label=label,
                     alpha=alpha, linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x_list[i], data, marker=marker, c=color, alpha=alpha,
                     linewidth=linewidth, markersize=markersize)


# Notice: the prototype of the Eigenfunction class
#   @ Learning Integral Representations of GP
class Eigenfunction_class():
    def __init__(self, poly_num, a, b, c, coe=2, **kwargs):
        self.poly_num = poly_num
        self.a = a
        self.b = b
        self.c = c
        self.coe = coe

    def evaluate_eigen_mat(self, x_np):
        """
        :param x_np: Should be (x_num, 1)   (number of index points, x dimension)
        :return:
        """
        assert x_np.shape[1] == 1

        x_for_h = x_np * np.sqrt(2 * self.c)

        exp_vec = np.exp( - (self.c - self.a) * np.square(x_np))

        # print("exp_vec.shape =", exp_vec.shape)

        self.poly_evaluator = orthopy.e1r2.Eval(x_for_h,
                          standardization="probabilists",  # "physicists", "probabilists"
                          scaling="normal")
        vec_list = []
        # c = 1
        for var in itertools.islice(self.poly_evaluator, self.poly_num):
            # print("var.shape =", var.shape)
            vec_list.append(np.multiply(var, exp_vec))

        vec_mat = np.asarray(vec_list).T
        vec_output = np.squeeze(vec_mat) / self.coe
        return vec_output

    def evaluate_mat_list(self, X_list):
        # Notice: Get the U and V list here !!!
        mat_list = []
        for i in range(len(X_list)):
            x_l = X_list[i]
            mat_list.append(self.evaluate_eigen_mat(x_l))

        return mat_list


# Notice: The Sinkhorn solver
def sinkhorn_plan(cost_matrix, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    # print("In skh alg")
    # print("cost_matrix.shape =", cost_matrix.shape)
    # print("r.shape =", r.shape)
    # print("c.shape =", c.shape)

    n, m = cost_matrix.shape
    P = np.exp(- lam * cost_matrix)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * cost_matrix)


# notice: Used to plot the origin data
def plot_origin_domain_data(plt, x, data_list, marker, color, label, alpha=0.5, s=42):
    for i, data in enumerate(data_list):
        if i == 0:
            plt.scatter(x, data, marker=marker, c=color, label=label, alpha=alpha, s=s)
        else:
            plt.scatter(x, data, marker=marker, c=color, alpha=alpha, s=s)


# notice: Used to plot the origin data as scatters
def plot_origin_domain_data_scatter(plt, x, data_list, marker, color, label, alpha=0.5, s=42):
    for i, data in enumerate(data_list):
        if i == 0:
            plt.scatter(x, data, marker=marker, c=color, label=label, alpha=alpha, s=s)
        else:
            plt.scatter(x, data, marker=marker, c=color, alpha=alpha, s=s)


def plot_origin_domain_data_line(plt, x_list, data_list, marker, color, label,
                                 alpha=0.5, linewidth=5, markersize=12):
    for i, data in enumerate(data_list):
        if i == 0:
            plt.plot(x_list[i], data, marker=marker, c=color, label=label,
                     alpha=alpha, linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x_list[i], data, marker=marker, c=color, alpha=alpha,
                     linewidth=linewidth, markersize=markersize)


# notice: Used to plot function lines
def plot_functions(plt, x, data_list, label_words, color, alpha, label_style='first', **kwargs):
    for i, data in enumerate(data_list):
        if label_style == 'first':
            if i == 0:
                plt.plot(x, data, color=color, label=label_words, alpha=alpha, **kwargs)
            else:
                plt.plot(x, data, color=color, alpha=alpha, **kwargs)


# Notice: empirical error... for quantitative comparison
def loss_l2_average(data_list_1, data_list_2):
    data_num_1 = len(data_list_1)
    data_num_2 = len(data_list_2)
    loss_mat = np.zeros((data_num_1, data_num_2))
    for i in range(data_num_1):
        for j in range(data_num_2):
            # Notice: Need to average over the axis?
            c = np.sum(np.square(data_list_1[i] - data_list_2[j]))
            loss_mat[i, j] = c
    return np.sum(loss_mat)


# Notice: empirical error... for quantitative comparison
def loss_weighted_l2_average(data_list_1, data_list_2, coupling):
    data_num_1 = len(data_list_1)
    data_num_2 = len(data_list_2)
    loss_mat = np.zeros((data_num_1, data_num_2))
    for i in range(data_num_1):
        for j in range(data_num_2):
            c = np.sum(np.square(data_list_1[i] - data_list_2[j]))
            loss_mat[i, j] = c
    # print('loss_mat =', loss_mat)
    return np.sum(np.multiply(loss_mat, coupling))


# Notice: Obtain the wasserstein distance with l2 ground metrix
def loss_l2_Wasserstein(data_list_1, data_list_2, lam=1.0/40, epsilon=1e-3):
    """
    :param data_list_1: A list of arrays
    :param data_list_2: A list of nd arrays
    :param lam: The coefficient of entropy
    :return: The W-distance: <Pi, C>_F
    """
    data_num_1 = len(data_list_1)
    data_num_2 = len(data_list_2)
    loss_mat = np.zeros((data_num_1, data_num_2))
    for i in range(data_num_1):
        for j in range(data_num_2):
            c = np.sum(np.square(data_list_1[i] - data_list_2[j]))
            loss_mat[i, j] = c
    # print('loss_mat =', loss_mat)
    # Notice: Then obtain the w-distance
    r_marginal = np.ones(data_num_1)
    c_marginal = np.ones(data_num_2)
    # print("loss_mat =", loss_mat)
    coupling, w_loss = sinkhorn_plan(loss_mat, r_marginal, c_marginal, lam, epsilon=epsilon)
    # print("w_loss =", w_loss)
    return np.sum(np.multiply(loss_mat, coupling))


# Notice: Generate Mixtures of Sine functions, 1D functions
def Generate_Sine_Mixture(mix_center_list, mix_var_list,
                          sine_scale_list, sine_scale_var_list,
                          sine_amp_list, sine_amp_var_list,
                          sine_shift_list, sine_shift_var_list,
                          x_list, traj_num, mix_type='normal'):
    # Notice: Generate center
    if mix_type not in ['normal', 'uniform']:
        print("mix_type mus be 'normal' or 'uniform'. Changed to 'normal'. ")
        return [], []
    center_index_list = np.random.choice(len(mix_center_list), traj_num)
    print('center_index_list =', center_index_list)

    func_mean_list = []
    for center_index in center_index_list:

        if mix_type == 'normal':
            func_mean = np.random.normal(loc=mix_center_list[center_index],
                                         scale=mix_var_list[center_index])
        elif mix_type == 'uniform':
            low_temp = mix_center_list[center_index] - mix_var_list[center_index]
            high_temp = mix_center_list[center_index] + mix_var_list[center_index]
            func_mean = np.random.uniform(low=low_temp, high=high_temp)
        func_mean_list.append(func_mean)

    print('func_mean =', func_mean_list)
    # sorted_func_mean_list = func_mean_list.sort()

    y_list = []
    y_mean_list = []
    x_data_list = []
    for j, func_mean in enumerate(func_mean_list):
        i = center_index_list[j]
        scale = np.random.normal(loc=sine_scale_list[i],scale=sine_scale_var_list[i])
        amp = np.random.normal(loc=sine_amp_list[i],scale=sine_amp_var_list[i])
        shift = np.random.normal(loc=sine_shift_list[i],scale=sine_shift_var_list[i])

        y = amp * np.sin(scale * x_list + shift) + func_mean
        y_list.append(y)
        y_mean_list.append(np.mean(y))
        x_data_list.append(x_list)

    y_sorted = [y for _, y in sorted(zip(y_mean_list, y_list),reverse=True)]

    return y_sorted, x_data_list


# Notice: Genrate multidimensional output functions
#   Just the very simple case, to test the algorithm
#   Note that, the entity here in the list represents every dimension
def Generate_MO_Sine(mix_center_list, mix_var_list,
                      sine_scale_list, sine_scale_var_list,
                      sine_amp_list, sine_amp_var_list,
                      sine_shift_list, sine_shift_var_list,
                      t_list, traj_num, mix_type='normal'):
    if mix_type not in ['normal', 'uniform']:
        print("mix_type mus be 'normal' or 'uniform'. Changed to 'normal'. ")
        return [], []

    data_len = t_list.shape[0]
    Y_dim = len(mix_center_list)
    Y_data_list = []
    X_data_list = []

    for traj in range(traj_num):
        data = np.zeros((data_len, Y_dim))

        # Notice: Sample the parameters of functions
        for d in range(Y_dim):
            if mix_type == "normal":
                center_temp = mix_center_list[d] + np.random.normal(loc=0, scale=mix_var_list[d])
            elif mix_type == "uniform":
                center_temp = mix_center_list[d] \
                              + np.random.uniform(low=-mix_var_list[d], high=mix_var_list[d])
            scale_temp = np.random.normal(loc=sine_scale_list[d], scale=sine_scale_var_list[d])
            amp_temp = np.random.normal(loc=sine_amp_list[d], scale=sine_amp_var_list[d])
            shift_temp = np.random.normal(loc=sine_shift_list[d], scale=sine_shift_var_list[d])

            data[:, d:d+1] = amp_temp * np.sin(scale_temp * t_list + shift_temp) + center_temp

        Y_data_list.append(data)
        X_data_list.append(t_list)

    return Y_data_list, X_data_list