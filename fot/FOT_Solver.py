
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
import time

# Notice: Self defined functions
# from General_Integral_GP_test import GP_model, data_domain_1, data_domain_2


class Continuous_FOT_optimization:
    def __init__(self, Y1_list, Y2_list,
                 X1_list, X2_list,
                 V_class, U_class,
                 l_num, k_num,
                 eigenfunc_num,
                 Y_dim=1):
        self.Y1_list = Y1_list
        self.Y2_list = Y2_list

        self.X1_list = X1_list
        self.X2_list = X2_list

        self.V_class = V_class
        self.U_class = U_class

        self.l_num = len(Y1_list)
        self.k_num = len(Y2_list)

        self.eignfunc_num = eigenfunc_num

        if self.l_num != l_num or self.k_num != k_num:
            print('Wrong! l_num, k_num!')

        _, self.Y_dim = Y1_list[0].shape
        # if self.data_len != data_len:
        #     print("Wrong! data_len")
        #     return
        if self.Y_dim != Y_dim:
            print("Wrong! Y_dim")
            return

        self.A = np.zeros((self.eignfunc_num, self.eignfunc_num))
        self.Pi = np.zeros((self.l_num, self.k_num))

        # Notice: Get the U and V list here !!!
        print("Get U and V list for each data!")
        self.U_list = []  # len = l_num
        self.V_list = []  # len = k_num
        for i in range(self.l_num):
            x_l = self.X1_list[i]
            self.U_list.append(self.U_class.evaluate_eigen_mat(x_l))
        for j in range(self.k_num):
            x_k = self.X2_list[j]
            self.V_list.append(self.V_class.evaluate_eigen_mat(x_k))

    # Notice: The cost function, l2 norm here, useless here
    def Cost_Function_norm(self, A, f_l, f_k, V, U):
        return np.sqrt(np.sum(np.square(V @ A @ U.T @ f_l - f_k)))

    # Todo: Modify to continuous case
    # Notice: A function to get the cost_matrix
    def Cost_Matrix(self, A):
        C_mat = np.zeros((self.l_num, self.k_num))
        for l in range(self.l_num):
            for k in range(self.k_num):
                # Notice: Not used in contnuous case..
                if self.multi_output:   #
                    F1_list_l_reshaped = self.F1_list[l].reshape((-1, 1))
                    F2_list_k_reshaped = self.F2_list[k].reshape((-1, 1))
                    C_mat[l, k] = np.sum(np.square(V @ A @ U.T @ F1_list_l_reshaped - F2_list_k_reshaped))
                else:
                    # TODO: Continuous case modify here!
                    U_x_l = self.U_list[l]
                    V_x_k = self.V_list[k]
                    Y_1_l = self.Y1_list[l]
                    Y_2_k = self.Y2_list[k]
                    C_mat[l, k] = np.sum(np.square(V_x_k @ A @ U_x_l.T @ Y_1_l - Y_2_k))
        return C_mat

    # Notice: The overall objective function to be minimized
    def Objective_Function(self):
        cost_mat = self.Cost_Matrix(A=self.A)
        objective_value = np.sum(np.multiply(cost_mat, self.Pi)) + np.sum(np.square(self.A))
        if self.entropy:
            objective_value += self.gamma_h * np.sum(np.multiply(self.Pi, np.log(self.Pi)))
        return objective_value

    # Notice: Initialize the variable amd the Lagrangian multipliers
    def Set_Initial_Variables(self, ini_A, ini_Pi,
                                ini_lbd_k, ini_lbd_l,
                                ini_lbd_i, s_mat
                                ):
        '''
        :param ini_A: (daya_len, data_len)
        :param ini_Pi: (l_num, k_num)
        :param ini_lbd_k: (k_num,)
        :param ini_lbd_l: (l_num,)
        :param ini_lbd_i: (l_num, k_num)
        :param s_mat: (l_num, k_num); slack variable for inequality
        :return:
        '''
        self.A = ini_A              # (daya_len, data_len)
        self.Pi = ini_Pi            # (l_num, k_num)
        self.lbd_k = ini_lbd_k      # (k_num,)
        self.lbd_l = ini_lbd_l      # (l_num,)
        self.lbd_i = ini_lbd_i      # (l_num, k_num)
        self.s_mat = s_mat          # (l_num, k_num); slack variable for inequality

    # Notice: Set the parameter of method of multipliers
    def Set_Parameters(self, rho_k, rho_l, rho_i,
                       gamma_A=1.0, gamma_h=1.0, gamma_c=1.0,
                       gamma_power=0, l_power=0.1):
        '''
        :param rho_k: (k_num,); hyper parameter of equality constraint for Pi
        :param rho_l: (l_num,); hyper parameter of equality constraint for Pi
        :param rho_i: scalar; hyper parameter of inequality constraint for Pi
        :param gamma_A: scalar; hyper parameter of A
        :param gamma_h: scalar; hyper parameter of entropy
        :param gamma_power: scalar; hyper parameter of power regularizer
        :param l_power: scalar, to regularize, let gamma_power > 0 and l < 1; or gamma_power <0 and l > 1;
        :return:
        '''
        self.rho_k = rho_k  # (k_num,); hyper parameter of equality constraint for Pi
        self.rho_l = rho_l  # (l_num,); hyper parameter of equality constraint for Pi
        self.rho_i = rho_i  # scalar; hyper parameter of inequality constraint for Pi
        self.gamma_h = gamma_h
        self.gamma_A = gamma_A
        self.gamma_c = gamma_c
        self.gamma_power = gamma_power
        self.l_power = l_power

    # Notice: Do the optimization
    def Optimize(self, lr_A, lr_Pi, tho, diagonal=False, max_iteration=50, entropy=False,
                        fix_Pi=False, inequality=False, multi_output=False):

        self.lr_A = lr_A
        self.lr_Pi = lr_Pi
        self.threshold = tho
        self.entropy = entropy
        self.fix_Pi = fix_Pi
        self.inequality = inequality
        self.multi_output = multi_output    # Notice: Important! It will affect the computing procedure

        # Notice: get the initial objective value
        objective_value = self.Objective_Function()

        # Notice: Iterate
        for i in range(max_iteration):
            print()
            print('Iteration Step: ', i)
            print("self.Objective_Function() =", self.Objective_Function())
            # Notice: Update the A matrix using gradient descent
            # TODO: Again, no multi-output case currently
            if self.multi_output:
                grad_A = np.zeros((self.data_len * self.Y_dim, self.data_len * self.Y_dim))
            else:
                grad_A = np.zeros((self.eignfunc_num, self.eignfunc_num))

            brutal_A_time_start = time.time()
            for l in range(self.l_num):
                for k in range(self.k_num):
                    # Notice: Compute the d C_lk / dA \in R{nxn}
                    # TODO: Again, no multi-output case currently
                    if self.multi_output:
                        raise Exception("Error! No multi-output in this case")
                        F1_list_l_reshaped = self.F1_list[l].reshape((-1, 1))
                        F2_list_k_reshaped = self.F2_list[k].reshape((-1, 1))
                        d_C_lk = (self.A @ self.U.T @ F1_list_l_reshaped - self.V.T @ F2_list_k_reshaped) \
                                 @ F1_list_l_reshaped.T @ self.U
                    else:
                        U_x_l = self.U_list[l]
                        V_x_k = self.V_list[k]
                        Y_1_l = self.Y1_list[l]
                        Y_2_k = self.Y2_list[k]
                        d_C_lk = (self.A @ U_x_l.T @ Y_1_l - V_x_k.T @ Y_2_k) \
                                 @ Y_1_l.T @ U_x_l
                    grad_A = grad_A + self.Pi[l, k] * d_C_lk * self.gamma_c

            grad_A += 2 * self.gamma_A * self.A
            # print("grad_A =", grad_A)
            brutal_A_time_end = time.time() - brutal_A_time_start

            # # Todo: Explore vectorized gradient of A under continuous case later
            # # Notice: 12/06/2020: Vectorized grad A
            # #   1. Obtain AUTF = A @ U.T @ F, where F \in R^{data_len, l_mun
            # vector_A_time_start = time.time()
            # F_mat = np.squeeze(np.asarray(self.F1_list)).T
            # # Notice: 2. VTG = V.T @ G, where G \in R^{data_len, k_num}
            # G_mat = np.squeeze(np.asarray(self.F2_list)).T
            # AUTF = self.A @ self.U.T @ F_mat
            # VTG = self.V.T @ G_mat
            # AUTF_tsr = np.expand_dims(AUTF.T, axis=1)
            # VTG_tsr = np.expand_dims(VTG.T, axis=0)
            # AUTF_tsr_tle = np.tile(AUTF_tsr, (1, self.k_num, 1))
            # VTG_tsr_tle = np.tile(VTG_tsr, (self.l_num, 1, 1))
            # diff_tsr = AUTF_tsr_tle - VTG_tsr_tle
            #
            # FT_tsr_tle = np.tile(F_mat.T[:, None, :], (1, self.k_num, 1))
            # diffF = np.einsum("lkn,lkm->lknm", diff_tsr, FT_tsr_tle)  # (A@U^T@f_1 - V^T@f_2)@f_1^T
            # U_tsr = self.U[None, None, :, :]
            # U_tsr_tle = np.tile(U_tsr, (self.l_num, self.k_num, 1, 1))
            # M_tsr = np.einsum("lkan,lknb->lkab", diffF, U_tsr_tle)
            # grad_A_vctr = np.einsum("lk, lknm->nm", self.Pi, M_tsr)
            # grad_A_vctr += 2 * self.gamma_A * self.A
            # vector_A_time_end = time.time() - vector_A_time_start
            #
            # print("brutal_A_time_end =", brutal_A_time_end)
            # print("vector_A_time_end =", vector_A_time_end)
            #
            # print("grad_A_vctr =")
            # print(grad_A_vctr)
            # print("grad_A =")
            # print(grad_A)

            # Notice: consider whether A is diagonal gradient
            if diagonal:
                grad_A = np.eye(self.eignfunc_num) * grad_A
            self.A = self.A - lr_A * grad_A

            # print('self.fix_Pi =', self.fix_Pi)

            if self.fix_Pi:
                continue
            else:
                # Notice: Update the Pi matrix
                # print('update Pi')
                last_cost = objective_value
                cost_difference = 100  # Set a large number
                C_matrix_temp = self.Cost_Matrix(A=self.A)
                # Notice: At this step, find the Pi that
                #  minimizes the current objective function
                #   Todo: modify it to be a <for iteration>
                # print('self.Pi =', self.Pi)
                while cost_difference > 0.01:

                    grad_Pi = np.zeros((self.l_num, self.k_num))
                    for l in range(self.l_num):
                        for k in range(self.k_num):
                            d_pi_lk = C_matrix_temp[l, k] + self.lbd_k[k] + self.lbd_l[l]
                            d_pi_lk += 1 * self.rho_k[k] * np.sum(self.Pi[:, k])
                            d_pi_lk += 1 * self.rho_l[l] * np.sum(self.Pi[l, :])
                            d_pi_lk += - 1 * (self.rho_k[k] + self.rho_l[l]) * self.Pi[l, k]
                            d_pi_lk += self.gamma_power * (self.Pi[l, k]**(self.l_power - 1))
                            if self.entropy:
                                d_pi_lk += self.gamma_h * (np.log(self.Pi[l, k]) + 1)
                            if self.inequality:
                                d_pi_lk += self.lbd_i[l, k] + self.rho_i * (self.Pi[l, k] - self.s_mat[l, k])
                            grad_Pi[l, k] = d_pi_lk
                    # print('grad_Pi =', grad_Pi)
                    # Notice: Now obtained the grad of Pi
                    Pi_temp = self.Pi - self.lr_Pi * grad_Pi
                    # # Notice: The boundary is applied here
                    if not self.inequality:
                        if (np.max(Pi_temp) >= 1.0) or (np.min(Pi_temp) <= 0):
                            break
                        else:
                            self.Pi = Pi_temp
                    else:
                        self.Pi = Pi_temp

                    # Notice: Terminate this update if found the
                    cost_temp = self.Objective_Function()
                    cost_difference = np.abs(last_cost - cost_temp)
                    last_cost = cost_temp

            # Notice: update the multipliers
            self.lbd_k = self.lbd_k + np.multiply(self.rho_k, np.sum(self.Pi, axis=0) - 1)
            self.lbd_l = self.lbd_l + np.multiply(self.rho_l, np.sum(self.Pi, axis=1) - 1)
            self.lbd_i = self.lbd_i + self.rho_i * (self.Pi - self.s_mat)

            # print("At this iteration, self.Pi =")
            # print(self.Pi)

        print('Optimization Ended')
        print("self.A = ")
        print(self.A)
        print("self.Pi =")
        print(self.Pi)
        print("self.Objective_Function()")
        print(self.Objective_Function())

        return self.A, self.Pi

    # Notice: Just some function
    def Somefunction(self):
        return


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
    def __init__(self, F1_list, F2_list, V, U, l_num, k_num, data_len, Y_dim=1):
        self.F1_list = F1_list
        self.F2_list = F2_list
        self.V = V
        self.U = U

        self.l_num = len(F1_list)
        self.k_num = len(F2_list)
        if self.l_num != l_num or self.k_num != k_num:
            print('Wrong! l_num, k_num!')

        self.data_len, self.Y_dim = F1_list[0].shape
        if self.data_len != data_len:
            print("Wrong! data_len")
            return
        if self.Y_dim != Y_dim:
            print("Wrong! Y_dim")
            return

        self.A = np.zeros((self.data_len, self.data_len))
        self.Pi = np.zeros((self.l_num, self.k_num))

    # Notice: The cost function, l2 norm here, useless here
    def Cost_Function_norm(self, A, f_l, f_k, V, U):
        return np.sqrt(np.sum(np.square(V @ A @ U.T @ f_l - f_k)))

    def Cost_Matrix(self, A, U, V):
        C_mat = np.zeros((self.l_num, self.k_num))
        for l in range(self.l_num):
            for k in range(self.k_num):
                if self.multi_output:
                    F1_list_l_reshaped = self.F1_list[l].reshape((-1, 1))
                    F2_list_k_reshaped = self.F2_list[k].reshape((-1, 1))
                    C_mat[l, k] = np.sum(np.square(V @ A @ U.T @ F1_list_l_reshaped - F2_list_k_reshaped))
                else:
                    C_mat[l, k] = np.sum(np.square(V @ A @ U.T @ self.F1_list[l] - self.F2_list[k]))
        return C_mat

    # Notice: The overall objective function to be minimized
    def Objective_Function(self):
        cost_mat = self.Cost_Matrix(A=self.A, U=self.U, V=self.V)
        objective_value = np.sum(np.multiply(cost_mat, self.Pi)) + np.sum(np.square(self.A))
        if self.entropy:
            objective_value += self.gamma_h * np.sum(np.multiply(self.Pi, np.log(self.Pi)))
        return objective_value

    # Notice: Initialize the variable amd the Lagrangian multipliers
    def Set_Initial_Variables(self, ini_A, ini_Pi,
                                ini_lbd_k, ini_lbd_l,
                                ini_lbd_i, s_mat
                                ):
        '''
        :param ini_A: (daya_len, data_len)
        :param ini_Pi: (l_num, k_num)
        :param ini_lbd_k: (k_num,)
        :param ini_lbd_l: (l_num,)
        :param ini_lbd_i: (l_num, k_num)
        :param s_mat: (l_num, k_num); slack variable for inequality
        :return:
        '''
        self.A = ini_A              # (daya_len, data_len)
        self.Pi = ini_Pi            # (l_num, k_num)
        self.lbd_k = ini_lbd_k      # (k_num,)
        self.lbd_l = ini_lbd_l      # (l_num,)
        self.lbd_i = ini_lbd_i      # (l_num, k_num)
        self.s_mat = s_mat          # (l_num, k_num); slack variable for inequality

    # Notice: Set the parameter of method of multipliers
    def Set_Parameters(self, rho_k, rho_l, rho_i,
                       gamma_A=1.0, gamma_h=1.0,
                       gamma_power=0, l_power=0.1):
        '''
        :param rho_k: (k_num,); hyper parameter of equality constraint for Pi
        :param rho_l: (l_num,); hyper parameter of equality constraint for Pi
        :param rho_i: scalar; hyper parameter of inequality constraint for Pi
        :param gamma_A: scalar; hyper parameter of A
        :param gamma_h: scalar; hyper parameter of entropy. the 1 / lambda in sinkhorn
        :param gamma_power: scalar; hyper parameter of power regularizer
        :param l_power: scalar, to regularize, let gamma_power > 0 and l < 1; or gamma_power <0 and l > 1;
        :return:
        '''
        self.rho_k = rho_k  # (k_num,); hyper parameter of equality constraint for Pi
        self.rho_l = rho_l  # (l_num,); hyper parameter of equality constraint for Pi
        self.rho_i = rho_i  # scalar; hyper parameter of inequality constraint for Pi
        self.gamma_h = gamma_h
        self.gamma_A = gamma_A
        self.gamma_power = gamma_power
        self.l_power = l_power

    # Notice: Do the optimization
    def Optimize(self, lr_A, lr_Pi, tho, diagonal=False, max_iteration=50, entropy=False,
                        fix_Pi=False, inequality=False, multi_output=False, sinkhorn=False):

        self.lr_A = lr_A
        self.lr_Pi = lr_Pi
        self.threshold = tho
        self.entropy = entropy
        self.fix_Pi = fix_Pi
        self.inequality = inequality
        self.multi_output = multi_output    # Notice: Important! It will affect the computing procedure
        self.sinkhorn = sinkhorn

        # Notice: get the initial objective value
        objective_value = self.Objective_Function()
        self._Pi_brutal_step_time_list = []
        self._Pi_sinkhorn_step_time_list = []

        # Notice: Iterate
        for i in range(max_iteration):
            print()
            print('Iteration Step: ', i)
            print("self.Objective_Function() =", self.Objective_Function())
            # Notice: Update the A matrix using gradient descent
            if self.multi_output:
                grad_A = np.zeros((self.data_len * self.Y_dim, self.data_len * self.Y_dim))
            else:
                grad_A = np.zeros((self.data_len, self.data_len))

            brutal_A_time_start = time.time()
            for l in range(self.l_num):
                for k in range(self.k_num):
                    # Notice: Compute the d C_lk / dA \in R{nxn}
                    if self.multi_output:
                        F1_list_l_reshaped = self.F1_list[l].reshape((-1, 1))
                        F2_list_k_reshaped = self.F2_list[k].reshape((-1, 1))
                        d_C_lk = (self.A @ self.U.T @ F1_list_l_reshaped - self.V.T @ F2_list_k_reshaped) \
                                 @ F1_list_l_reshaped.T @ self.U
                    else:
                        d_C_lk = (self.A @ self.U.T @ self.F1_list[l] - self.V.T @ self.F2_list[k]) \
                                 @ self.F1_list[l].T @ self.U
                    grad_A = grad_A + self.Pi[l, k] * d_C_lk

            grad_A += 2 * self.gamma_A * self.A
            brutal_A_time_end = time.time() - brutal_A_time_start

            # Notice: 12/06/2020: Vectorized grad A
            #   1. Obtain AUTF = A @ U.T @ F, where F \in R^{data_len, l_mun
            vector_A_time_start = time.time()
            F_mat = np.squeeze(np.asarray(self.F1_list)).T
            # Notice: 2. VTG = V.T @ G, where G \in R^{data_len, k_num}
            G_mat = np.squeeze(np.asarray(self.F2_list)).T
            AUTF = self.A @ self.U.T @ F_mat
            VTG = self.V.T @ G_mat
            AUTF_tsr = np.expand_dims(AUTF.T, axis=1)
            VTG_tsr = np.expand_dims(VTG.T, axis=0)
            AUTF_tsr_tle = np.tile(AUTF_tsr, (1, self.k_num, 1))
            VTG_tsr_tle = np.tile(VTG_tsr, (self.l_num, 1, 1))
            diff_tsr = AUTF_tsr_tle - VTG_tsr_tle

            FT_tsr_tle = np.tile(F_mat.T[:, None, :], (1, self.k_num, 1))
            diffF = np.einsum("lkn,lkm->lknm", diff_tsr, FT_tsr_tle)  # (A@U^T@f_1 - V^T@f_2)@f_1^T
            U_tsr = self.U[None, None, :, :]
            U_tsr_tle = np.tile(U_tsr, (self.l_num, self.k_num, 1, 1))
            M_tsr = np.einsum("lkan,lknb->lkab", diffF, U_tsr_tle)
            grad_A_vctr = np.einsum("lk, lknm->nm", self.Pi, M_tsr)
            grad_A_vctr += 2 * self.gamma_A * self.A
            vector_A_time_end = time.time() - vector_A_time_start

            print("brutal_A_time_end =", brutal_A_time_end)
            print("vector_A_time_end =", vector_A_time_end)

            # print("grad_A_vctr =")
            # print(grad_A_vctr)
            # print("grad_A =")
            # print(grad_A)

            # Notice: consider whether A is diagonal gradient
            if diagonal:
                grad_A = np.eye(self.data_len) * grad_A
            self.A = self.A - lr_A * grad_A

            # print('self.fix_Pi =', self.fix_Pi)

            if self.fix_Pi:
                continue
            elif self.sinkhorn:
                # Notice: Test the Sinkhorn algorithm here
                sinkhorn_time_start = time.time()
                cost_matrix = self.Cost_Matrix(A=self.A, U=self.U, V=self.V)
                # Notice:
                #   Get the marginal distribution.
                #   Also, modify here
                # u_l = np.ones((self.l_num)) / self.k_num
                # u_k = np.ones((self.k_num)) / self.l_num
                u_l = np.ones((self.l_num))
                u_k = np.ones((self.k_num))
                sinkhorn_Pi, _ = sinkhorn_plan(cost_matrix=cost_matrix,
                                               r=u_l, c=u_k, lam=1 / self.gamma_h,
                                               epsilon=0.001)
                self.sinkhorn_Pi = sinkhorn_Pi
                sinkhorn_time_took = time.time() - sinkhorn_time_start
                print("sinkhorn_time_took =", sinkhorn_time_took)
                self._Pi_sinkhorn_step_time_list.append(sinkhorn_time_took)
            else:
                brutal_time_start = time.time()
                # Notice: Update the Pi matrix
                # print('update Pi')
                last_cost = objective_value
                cost_difference = 100  # Set a large number
                C_matrix_temp = self.Cost_Matrix(A=self.A, U=self.U, V=self.V)
                # Notice: At this step, find the Pi that
                #  minimizes the current objective function
                #   Todo: modify it to be a <for iteration>
                # print('self.Pi =', self.Pi)
                while cost_difference > 0.001:

                    grad_Pi = np.zeros((self.l_num, self.k_num))
                    # Notice: Multithread/Multiprocess here?
                    for l in range(self.l_num):
                        for k in range(self.k_num):
                            d_pi_lk = C_matrix_temp[l, k] + self.lbd_k[k] + self.lbd_l[l]
                            d_pi_lk += 1 * self.rho_k[k] * np.sum(self.Pi[:, k])
                            d_pi_lk += 1 * self.rho_l[l] * np.sum(self.Pi[l, :])
                            # Notice: The following line is the difference
                            # d_pi_lk += - 1 * (self.rho_k[k]/self.k_num + self.rho_l[l]/self.l_num) * self.Pi[l, k]
                            d_pi_lk += - 1 * (self.rho_k[k] + self.rho_l[l]) * self.Pi[l, k]
                            d_pi_lk += self.gamma_power * (self.Pi[l, k]**(self.l_power - 1))
                            if self.entropy:
                                d_pi_lk += self.gamma_h * (np.log(self.Pi[l, k]) + 1)
                            if self.inequality:
                                d_pi_lk += self.lbd_i[l, k] + self.rho_i * (self.Pi[l, k] - self.s_mat[l, k])
                            grad_Pi[l, k] = d_pi_lk
                    # print('grad_Pi =', grad_Pi)
                    # Notice: Now obtained the grad of Pi
                    Pi_temp = self.Pi - self.lr_Pi * grad_Pi
                    # # Notice: The boundary is applied here
                    if not self.inequality:
                        # Notice: also the difference here
                        # print("np.max(Pi_temp) =", np.max(Pi_temp))
                        # print("1/(self.l_num * self.k_num)) =", 1/(self.l_num * self.k_num))
                        # if (np.max(Pi_temp) >= (1/(self.l_num * self.k_num))) or (np.min(Pi_temp) <= 0):
                        if (np.max(Pi_temp) >= 1.0) or (np.min(Pi_temp) <= 0):
                            break
                        else:
                            self.Pi = Pi_temp
                    else:
                        self.Pi = Pi_temp

                    # Notice: Terminate this update if found the
                    cost_temp = self.Objective_Function()
                    cost_difference = np.abs(last_cost - cost_temp)
                    # print("cost_difference =", cost_difference)
                    last_cost = cost_temp

                brutal_time_took = time.time() - brutal_time_start
                print("brutal_time_took =", brutal_time_took)
                self._Pi_brutal_step_time_list.append(brutal_time_took)

            # Notice: update the multipliers
            #   Here is the difference
            self.lbd_k = self.lbd_k + np.multiply(self.rho_k, np.sum(self.Pi, axis=0) - 1)
            self.lbd_l = self.lbd_l + np.multiply(self.rho_l, np.sum(self.Pi, axis=1) - 1)
            self.lbd_i = self.lbd_i + self.rho_i * (self.Pi - self.s_mat)

            print("At this iteration, self.Pi =")
            print(self.Pi)

        if self.sinkhorn:
            self.Pi = self.sinkhorn_Pi
        print('Optimization Ended')
        print("self.A = ")
        print(self.A)
        print("self.Pi =")
        print(self.Pi)
        print("self.Objective_Function()")
        print(self.Objective_Function())

        print("Sum brutal_time_took =", np.sum(self._Pi_brutal_step_time_list))
        print("Sum sinkhorn_time_took =", np.sum(self._Pi_sinkhorn_step_time_list))
        return self.A, self.Pi

    # Notice: Just some function
    def Somefunction(self):
        return


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


def plot_origin_domain_data_line(plt, x, data_list, marker, color, label,
                                 alpha=0.5, linewidth=5, markersize=12):
    for i, data in enumerate(data_list):
        if i == 0:
            plt.plot(x, data, marker=marker, c=color, label=label,
                     alpha=alpha, linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x, data, marker=marker, c=color, alpha=alpha,
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


if __name__ == '__main__':

    # Notice: Create the test dataset
    data_len = 10  # Notice, this should be consistent all through the process
    x = np.linspace(0, 10, data_len).reshape(-1, 1)

    y_12 = 6 * np.sin(0.2 * (x + 4)) - x / 10 - 3.4
    y_11 = 6 * np.sin(0.2 * (x + 4.5)) + x / 20 - 3.6
    l_num = 2
    F1_list = [y_11, y_12]

    y_21 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 3.6
    y_22 = 6 * np.sin(0.2 * (x + 1.7)) - x / 10 - 4.0
    k_num = 2
    F2_list = [y_21, y_22]

    # Notice: Plot it
    fig = plt.figure()

    plt.scatter(x, y_11, marker='+', c='tomato', label='y_11')
    plt.scatter(x, y_12, marker='+', c='fuchsia', label='y_12')

    plt.scatter(x, y_21, marker='s', c='aqua', label='y_21')
    plt.scatter(x, y_22, marker='s', c='teal', label='y_22')

    # Notice:
    #   Step 0:
    #   Process the data for training

    X1_train = np.concatenate([x, x], axis=0)
    Y1_train = np.concatenate([y_11, y_12], axis=0)

    X2_train = np.concatenate([x, x], axis=0)
    Y2_train = np.concatenate([y_21, y_22], axis=0)

    noise = 0.4

    # Notice:
    #   Step 1: Use GP regression to get K

    # Notice: Get the data for training GP_1

    X_test = np.linspace(0, 10, data_len).reshape(-1, 1)
    # print('X_test =', X_test)
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

    # Notice: Try the optimizer
    GFOT_optimizer = GFOT_optimization(F1_list=F1_list, F2_list=F2_list,
                                       V=V, U=U, l_num=l_num, k_num=k_num, data_len=data_len)

    # notice: Set initial values
    ini_A = np.eye(data_len)
    # ini_A = np.zeros((data_len, data_len))
    ini_Pi = 0.5 * np.ones((2, 2))
    # ini_Pi = np.eye(2)

    lbd_k = 0.1 * np.ones((k_num,))
    lbd_l = 0.1 * np.ones((l_num,))
    lbd_i = 0.1 * np.ones((l_num, k_num))

    s_mat = 0.1 * np.ones((l_num, k_num))

    GFOT_optimizer.Set_Initial_Variables(ini_A=ini_A, ini_Pi=ini_Pi,
                                         ini_lbd_k=lbd_k, ini_lbd_l=lbd_l,
                                         ini_lbd_i=lbd_i, s_mat=s_mat)

    # Notice: Set lagrangian parameters
    rho_k = 40 * np.ones((k_num,))
    rho_l = 40 * np.ones((l_num,))
    rho_i = 1
    gamma_h = 1e-2
    GFOT_optimizer.Set_Parameters(rho_k=rho_k, rho_l=rho_l, rho_i=rho_i, gamma_h=gamma_h)

    # Notice: Do the optimization
    lr_A = 1e-2
    lr_Pi = 0.0001
    ite_num = 50
    A_opt, Pi_opt = GFOT_optimizer.Optimize(lr_A=lr_A, lr_Pi=lr_Pi, tho=1e-5,
                                            diagonal=True, max_iteration=ite_num,
                                            entropy=False, fix_Pi=False,
                                            inequality=False)

    # Notice: Check the result and plot
    f_sharp_11 = V @ A_opt @ U.T @ y_11
    f_sharp_12 = V @ A_opt @ U.T @ y_12
    plt.plot(X_test, f_sharp_11, c='orange', label=str(ite_num) + ' push of y_11', alpha=0.3)
    plt.plot(X_test, f_sharp_12, c='orange', label=str(ite_num) + ' push of y_12', alpha=0.3)

    plt.legend()
    plt.show()


'''
#   The following is the draft of 
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
    initial_Pi = 0.5 * np.ones((2, 2))
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

    # Notice: The varaibles
    l_num = 2
    k_num = 2

    # Notice: The Pi
    ini_Pi = 0.5 * np.ones((2, 2))
    Pi = ini_Pi

    # print('ini_Pi =', ini_Pi)

    # Notice: The lbd
    lbd_k = 0.1 * np.ones((k_num, ))
    lbd_l = 0.1 * np.ones((l_num, ))

    # Notice: The augumentation variable
    rho_k = 40 * np.ones((k_num,))
    rho_l = 40 * np.ones((l_num,))

    # Notice:
    #   Start to compute the gradient
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
                for k in range(l_num):
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

'''













