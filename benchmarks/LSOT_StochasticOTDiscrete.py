'''
Used as one of the benchmark
'''
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as func
# from LSOT_StochasticOT import PyTorchStochasticOT


class PyTorchStochasticOT:
    def __init__(self, reg_type='entropy', reg_val=0.1, device_type='cpu', device_index=0):

        self.reg_type = reg_type
        self.reg_val = reg_val
        self.dtype = torch.float64
        self.device_type = device_type
        self.device_index = device_index

        self.barycentric_mapping = None

        if device_type == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:%d' % (self.device_index,))
        else:
            self.device = torch.device('cpu')


    def computeSquareEuclideanCostMatrix(self, Xs_batch, Xt_batch):
        return torch.reshape(torch.sum(torch.mul(Xs_batch, Xs_batch), dim=1), (-1, 1)) + torch.reshape(torch.sum(torch.mul(Xt_batch, Xt_batch), dim=1), (1, -1)) \
               - 2. * torch.matmul(Xs_batch, torch.transpose(Xt_batch, 0,1))


    def dual_OT_batch_loss(self, batch_size, u_batch, v_batch, Xs_batch, Xt_batch):

        C_batch = self.computeSquareEuclideanCostMatrix(Xs_batch=Xs_batch, Xt_batch=Xt_batch)

        if self.reg_type == 'entropy':
            loss_batch = torch.sum(u_batch)*batch_size + torch.sum(v_batch)*batch_size \
                         - self.reg_val*torch.sum(torch.exp((torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch) / self.reg_val))
        elif self.reg_type == 'l2':
            tmp = torch.max(torch.zeros(1, dtype=self.dtype, device=self.device), (torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch))
            loss_batch = torch.sum(u_batch)*batch_size + torch.sum(v_batch)*batch_size \
                         - (1./(4.*self.reg_val))*torch.sum(torch.mul(tmp, tmp))

        return -loss_batch


    def barycentric_model_batch_loss(self, u_batch, v_batch, Xs_batch, Xt_batch, fXs_batch):

        C_batch = self.computeSquareEuclideanCostMatrix(Xs_batch, Xt_batch)

        if self.reg_type == 'entropy':
            H = torch.exp((torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch) / self.reg_val)
        elif self.reg_type == 'l2':
            H = (1./(2.*self.reg_val))*torch.max(torch.zeros(1, dtype=self.dtype, device=self.device), (torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch))

        d_batch = self.computeSquareEuclideanCostMatrix(fXs_batch, Xt_batch)

        return torch.sum(torch.mul(d_batch, H))


    def evaluate_barycentric_mapping(self, xs):
        self.barycentric_mapping.eval()
        xs_tensor = torch.from_numpy(xs).to(device=self.device)
        return self.barycentric_mapping(xs_tensor).detach().cpu().numpy()


class PyTorchStochasticDiscreteOT(PyTorchStochasticOT):


    def __init__(self, xs=None, ws=None, xt=None, wt=None, reg_type='entropy', reg_val=0.1, device_type='cpu', device_index=0):

        PyTorchStochasticOT.__init__(self, reg_type=reg_type, reg_val=reg_val, device_type=device_type, device_index=device_index)

        self.ns = xs.shape[0]
        self.nt = xt.shape[0]
        self.d = xt.shape[1]

        self.Xs = torch.from_numpy(xs).to(device=self.device)
        self.ws = torch.from_numpy(ws).to(device=self.device)
        self.Xt = torch.from_numpy(xt).to(device=self.device)
        self.wt = torch.from_numpy(wt).to(device=self.device)

        self.u = torch.zeros(self.ns, dtype=self.dtype, requires_grad=True, device=self.device) # first dual variable
        self.v = torch.zeros(self.nt, dtype=self.dtype, requires_grad=True, device=self.device) # second dual variable


    def dual_OT_model(self, i_s, i_t):

        batch_size = i_s.shape[0]
        u_batch = torch.index_select(self.u, dim=0, index=i_s)
        v_batch = torch.index_select(self.v, dim=0, index=i_t)
        Xs_batch = torch.index_select(self.Xs, dim=0, index=i_s)
        Xt_batch = torch.index_select(self.Xt, dim=0, index=i_t)

        return self.dual_OT_batch_loss(batch_size=batch_size, u_batch=u_batch, v_batch=v_batch, Xs_batch=Xs_batch, Xt_batch=Xt_batch)


    def sampleFromIndependantCoupling(self, batch_size):

        i_s = torch.from_numpy(np.random.choice(self.ns, size=(batch_size,), replace=False, p=self.ws)).type(torch.LongTensor).to(device=self.device)
        i_t = torch.from_numpy(np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)).type(torch.LongTensor).to(device=self.device)

        return i_s, i_t


    def learn_OT_dual_variables(self, epochs=10, batch_size=100, optimizer=None, lr=0.01):

        trainable_params = [self.u, self.v]
        if not optimizer:
            optimizer = torch.optim.SGD(trainable_params, lr=lr)

        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])

        self.losses = []
        self.time = []
        history = {}

        tic = time.time()

        for e in range(epochs):

            batch_losses = np.zeros((batch_number_per_epoch,))

            for b in range(batch_number_per_epoch):

                i_s, i_t = self.sampleFromIndependantCoupling(batch_size)

                optimizer.zero_grad()
                loss_batch = self.dual_OT_model(i_s, i_t)
                loss_batch.backward()
                optimizer.step()

                batch_losses[b] = loss_batch.item()

            print('Epoch : {}, loss = {}'.format(e+1, np.sum(batch_losses)))

            self.losses.append(-np.sum(batch_losses)/(self.ns*self.nt))
            self.time.append(time.time()-tic)

        history['losses'] = self.losses
        history['time'] = self.time

        return history


    def compute_OT_MonteCarlo(self, epochs=10, batch_size=100): # before calling this, find the optimum dual variables with learn_OT_dual_variables
        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])
        OT_value = 0.
        for e in range(epochs):
            for b in range(batch_number_per_epoch):
                i_s, i_t = self.sampleFromIndependantCoupling(batch_size)
                OT_value += self.dual_OT_model(i_s, i_t).item()
        return -OT_value/epochs/(self.nt*self.nt)


    def barycentric_mapping_loss_model(self, neuralNet, i_s, i_t):

        self.u.requires_grad_(False)
        self.v.requires_grad_(False)

        u_batch = torch.index_select(self.u, dim=0, index=i_s)
        v_batch = torch.index_select(self.v, dim=0, index=i_t)
        Xs_batch = torch.index_select(self.Xs, dim=0, index=i_s)
        Xt_batch = torch.index_select(self.Xt, dim=0, index=i_t)

        fXs_batch = neuralNet(Xs_batch)

        return self.barycentric_model_batch_loss(u_batch, v_batch, Xs_batch, Xt_batch, fXs_batch)


    def learn_barycentric_mapping(self, neuralNet=None, epochs=10, batch_size=100, optimizer=None, lr=0.01):

        if not neuralNet:
            neuralNet = Net(input_d=self.d, output_d=self.d, device=self.device)

        self.barycentric_mapping = neuralNet.to(device=self.device)

        if not optimizer:
            optimizer = torch.optim.SGD(neuralNet.parameters(), lr=lr)

        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])

        barycentric_mapping_losses = []
        barycentric_mapping_time = []
        history = {}

        tic = time.time()

        for e in range(epochs):

            batch_losses = np.zeros((batch_number_per_epoch,))

            for b in range(batch_number_per_epoch):

                i_s = torch.from_numpy(np.random.choice(self.ns, size=(batch_size,), replace=False, p=self.ws)).type(torch.LongTensor).to(device=self.device)
                i_t = torch.from_numpy(np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)).type(torch.LongTensor).to(device=self.device)

                optimizer.zero_grad()

                loss_batch = self.barycentric_mapping_loss_model(neuralNet, i_s, i_t)
                loss_batch.backward()
                optimizer.step()

                batch_losses[b] = loss_batch.item()

            print('Epoch : {}, barycentric_mapping loss = {}'.format(e+1, np.sum(batch_losses)))

            barycentric_mapping_losses.append(np.sum(batch_losses)/(self.ns*self.nt))
            barycentric_mapping_time.append(time.time()-tic)

        history['losses'] = barycentric_mapping_losses
        history['time'] = barycentric_mapping_time

        return history



class Net(nn.Module):
    def __init__(self, input_d=2, output_d=2, device=None):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_d, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_d)
        self.device=device

    def forward(self, x):
        x = x.float()
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x).double()
        return x


