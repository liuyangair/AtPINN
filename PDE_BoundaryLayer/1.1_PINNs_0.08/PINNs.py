import sys
sys.path.insert(0, '../Utilities/')
import torch

import time
import matplotlib
import pickle
import random
import scipy.io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pyDOE import lhs
from collections import OrderedDict
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

random.seed(1234)
np.random.seed(1234)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DNN(torch.nn.Module):
    def __init__(self, layers, lb, ub):
        super(DNN, self).__init__()

        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out = self.layers(x)
        return out


# The physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, Collo, lp, up, lf, ur, uv_layers, lb, ub, eps):

        # Mat. properties 密度，粘性系数
        self.lb = lb
        self.ub = ub

        self.eps = eps
        self.x_cc = Collo[:, 0:2]

        # Collocation point
        self.t_ = torch.tensor(Collo[:, 0:1], requires_grad=True).float().to(device)
        self.x_ = torch.tensor(Collo[:, 1:2], requires_grad=True).float().to(device)

        self.t_lp = torch.tensor(lp[:, 0:1], requires_grad=True).float().to(device)
        self.x_lp = torch.tensor(lp[:, 1:2], requires_grad=True).float().to(device)

        self.t_up = torch.tensor(up[:, 0:1], requires_grad=True).float().to(device)
        self.x_up = torch.tensor(up[:, 1:2], requires_grad=True).float().to(device)

        self.t_lf = torch.tensor(lf[:, 0:1], requires_grad=True).float().to(device)
        self.x_lf = torch.tensor(lf[:, 1:2], requires_grad=True).float().to(device)

        self.t_ur = torch.tensor(ur[:, 0:1], requires_grad=True).float().to(device)
        self.x_ur = torch.tensor(ur[:, 1:2], requires_grad=True).float().to(device)

        # Define layers
        self.uv_layers = uv_layers

        # deep neural networks
        self.dnn = DNN(uv_layers, self.lb, self.ub).to(device)
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.1,
            max_iter=10000,
            max_eval=10000,
            history_size=200,
            tolerance_grad=1e-20,
            tolerance_change=1e-20
        )
        self.iter = 0
        self.iter_ = []
        self.loss_f = []
        self.loss_BC = []
        self.loss_save = []

    def net_u(self, t, x):
        psips = self.dnn(torch.cat([t, x], dim=1))
        u = self.dnn(psips)
        return u

    def net_f(self, t, x):
        u = self.net_u(t, x)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        e1 = self.eps*u_t - self.eps*self.eps*u_xx + u_x - 2*t

        return e1

    def loss_func_lgbs(self):
        self.optimizer_LBFGS.zero_grad()

        e1 = self.net_f(self.t_, self.x_)
        lossf = torch.mean(e1**2)

        pred_u_lf = self.net_u(self.t_lf, self.x_lf)
        pred_u_lp = self.net_u(self.t_lp, self.x_lp)
        pred_u_up = self.net_u(self.t_up, self.x_up)
        lossBC = torch.mean(pred_u_lf**2) + torch.mean(pred_u_lp**2) + torch.mean(pred_u_up**2)

        loss = lossf + lossBC

        self.loss_f.append(lossf.item())
        self.loss_BC.append(lossBC.item())
        self.loss_save.append(loss.item())

        loss.backward()
        self.iter += 1
        self.iter_.append(self.iter)

        if self.iter % 10 == 0:
            print('Iter %d, Loss: %.5e, loss_f: %.5e, loss_BC: %.5e' %
                  (self.iter, loss.item(), lossf.item(), lossBC.item()))

        if self.iter % 100 == 0:
            torch.save(self.dnn.state_dict(), './results/model_'+str(self.iter)+'.pth')
            self.predict_by_trained(self.x_cc, './results/model_'+str(self.iter)+'.pth')
            self.LossSave('Loss')

        return loss

    def LossSave(self, filename):
        data = np.array([self.iter_, self.loss_save, self.loss_f, self.loss_BC])
        data = np.transpose(data)
        df = pd.DataFrame(data, columns=['iter', 'Loss', 'Loss_f', 'loss_BC'])
        df.to_csv(filename + '.csv', index=False)

    def train_LBFGS(self):
        self.optimizer_LBFGS.step(self.loss_func_lgbs)
        torch.save(self.dnn.state_dict(), './results/model_LBFGS.pth')
        self.LossSave('Loss')

    def predict_by_trained(self, data, filename):
        self.dnn.load_state_dict(torch.load(filename))
        t = torch.tensor(data[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(data[:, 1:2], requires_grad=True).float().to(device)
        y_pred = self.net_u(t, x)
        y_pred = y_pred.detach().cpu().numpy()

        # 绘制等高线
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tick_params(labelsize=11)
        cf = ax.scatter(data[:, 0:1], data[:, 1:2], c=y_pred, alpha=1 - 0.1, edgecolors='none', cmap='rainbow', marker='o', s=int(5))
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        font = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'normal', 'size': 12}
        plt.xlabel('t', fontdict=font)
        plt.ylabel('x', fontdict=font)
        plt.axis('square')
        plt.savefig('./results/Figure_'+str(self.iter)+'.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 保存文件
        result = np.zeros([len(data), 3])
        result[:, 0] = data[:, 0]
        result[:, 1] = data[:, 1]
        result[:, 2] = y_pred[:, 0]
        df = pd.DataFrame(result, columns=['t', 'x', 'u'])
        df.to_csv('./results/results_'+str(self.iter)+'.csv', index=False, header=None)

        return y_pred

    def visualization(self, filename):
        self.dnn.load_state_dict(torch.load(filename))
        for k, v in self.dnn.named_parameters():
            name = k
            data = v.detach().cpu().numpy()
            df = pd.DataFrame(data)
            df.to_csv(name + '.csv', header=None, index=False)

    def predict_by_trained_1(self, data, filename):
        self.iter += 1
        self.dnn.load_state_dict(torch.load(filename))
        t = torch.tensor(data[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(data[:, 1:2], requires_grad=True).float().to(device)
        y_pred = self.net_u(t, x)
        y_pred = y_pred.detach().cpu().numpy()
        # 保存文件
        result = np.zeros([len(data), 3])
        result[:, 0] = data[:, 0]
        result[:, 1] = data[:, 1]
        result[:, 2] = y_pred[:, 0]
        df = pd.DataFrame(result, columns=['t', 'x', 'u'])
        df.to_csv('./results/curve_results_t=1.csv', index=False, header=None)

        return y_pred


if __name__ == "__main__":
    import os
    if not os.path.exists('results'):
        os.mkdir('results')

    # Network configuration
    uv_layers = [2] + 4*[30] + [1]

    # Domain bounds
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    lp = [0.0, 0.0] + [1.0, 0.0] * lhs(2, 101)
    lp[:, 0] = np.linspace(0.0, 1.0, 101)

    up = [0.0, 1.0] + [1.0, 0.0] * lhs(2, 101)
    up[:, 0] = np.linspace(0.0, 1.0, 101)

    lf = [0.0, 0.0] + [0.0, 1.0] * lhs(2, 201)
    lf[:, 1] = np.linspace(0.0, 1.0, 201)

    ur = [1.0, 0.0] + [0.0, 1.0] * lhs(2, 201)
    ur[:, 1] = np.linspace(0.0, 1.0, 201)

    a = np.linspace(0.0, 1.0, 101)
    b = np.linspace(0.0, 1.0, 201)
    Field = np.empty(shape=[101*201, 2])
    for i in range(101):
        for j in range(201):
            Field[i*201+j, 0] = b[j]
            Field[i*201+j, 1] = a[i]

    XY_c = np.concatenate((Field[:, 0:2], lp[:, 0:2], up[:, 0:2], lf[:, 0:2], ur[:, 0:2]), 0)

    eps = 0.08
    model = PhysicsInformedNN(XY_c, lp, up, lf, ur, uv_layers, lb, ub, eps)
    model.train_LBFGS()



