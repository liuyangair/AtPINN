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
    def __init__(self, Collo, BC, uv_layers, lb, ub, eps):

        # Mat. properties 密度，粘性系数
        self.lb = lb
        self.ub = ub

        self.eps = eps
        self.x_cc = Collo[:, 0:2]

        # Collocation point
        self.x_ = torch.tensor(Collo[:, 0:1], requires_grad=True).float().to(device)
        self.y_ = torch.tensor(Collo[:, 1:2], requires_grad=True).float().to(device)

        self.x_BC = torch.tensor(BC[:, 0:1], requires_grad=True).float().to(device)
        self.y_BC = torch.tensor(BC[:, 1:2], requires_grad=True).float().to(device)
        # Define layers
        self.uv_layers = uv_layers
        self.para = 1
        # deep neural networks
        self.dnn = DNN(uv_layers, self.lb, self.ub).to(device)

        self.eps_ = torch.tensor([0.0], requires_grad=True).to(device)
        self.eps_ = torch.nn.Parameter(self.eps_)
        self.dnn.register_parameter('eps_', self.eps_)

        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.5,
            max_iter=20000,
            max_eval=20000,
            history_size=200,
            tolerance_grad=1e-20,
            tolerance_change=1e-20
        )

        self.iter = 0
        self.iter_list = []
        self.loss_f_list = []
        self.loss_BC_list = []
        self.loss_list = []
        self.loss_eps_list = []
        self.eps_list = []

    def net_u(self, x, y):
        u = self.dnn(torch.cat([x, y], dim=1))
        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        e = u_xx + u_yy
        return e

    def loss_func_lgbs(self):
        self.optimizer_LBFGS.zero_grad()

        e = self.net_f(self.x_, self.y_)
        loss_f = torch.mean(e**2)
        u_BC_pred = self.net_u(self.x_BC, self.y_BC)

        eps1 = self.eps_ * 6 * 3.1415926
        eps2 = self.eps_ * 3 * 3.1415926

        self.u_BC = torch.sin(-eps1 * self.y_BC) * torch.exp(-eps2 * self.x_BC)
        loss_BC = torch.mean((self.u_BC - u_BC_pred) ** 2)
        loss_eps = torch.abs(self.eps_ - self.eps)
        loss_eps = (self.eps_ - self.eps)**2
        loss = loss_f + (loss_BC.item() / 0.00017) * loss_BC + loss_eps * self.para

        self.loss_f_list.append(loss_f.item())
        self.loss_BC_list.append(loss_BC.item())
        self.loss_list.append(loss.item())
        self.eps_list.append(self.eps_.item())
        self.loss_eps_list.append(loss_eps.item())
        loss.backward()
        self.iter += 1
        self.iter_list.append(self.iter)

        if self.iter % 10 == 0:
            print('Iter %d, eps: %.5e, Loss: %.5e, loss_f: %.5e, loss_BC: %.5e' %
                  (self.iter, self.eps_.item(), loss.item(), loss_f.item(), loss_BC.item()))

        if self.iter % 3000 == 0:
            torch.save(self.dnn.state_dict(), './results/model_'+str(self.iter)+'.pth')
            self.predict_by_trained(self.x_cc, './results/model_'+str(self.iter)+'.pth')
            self.LossSave('loss')

        return loss

    def LossSave(self, filename):
        data = np.array([self.iter_list, self.eps_list, self.loss_list, self.loss_f_list, self.loss_BC_list, self.loss_eps_list])
        data = np.transpose(data)
        df = pd.DataFrame(data, columns=['iter', 'eps', 'loss', 'loss_f', 'loss_BC', 'loss_eps'])
        df.to_csv(filename + '.csv', index=False)

    def train_LBFGS(self):
        self.optimizer_LBFGS.step(self.loss_func_lgbs)
        torch.save(self.dnn.state_dict(), './results/model_LBFGS.pth')
        torch.save(self.dnn.state_dict(), './results/model_'+str(self.iter)+'.pth')
        self.predict_by_trained(self.x_cc, './results/model_'+str(self.iter)+'.pth')
        self.LossSave('Loss'+str(self.iter))

    def predict_by_trained(self, data, filename):
        self.dnn.load_state_dict(torch.load(filename))
        x = torch.tensor(data[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(data[:, 1:2], requires_grad=True).float().to(device)
        u_pred = self.net_u(x, y)
        u_pred = u_pred.detach().cpu().numpy()

        # 绘制等高线
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tick_params(labelsize=11)
        cf = ax.scatter(data[:, 0:1], data[:, 1:2], c=u_pred, alpha=1 - 0.1, edgecolors='none', cmap='rainbow', marker='o', s=int(5))
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
        result[:, 2] = u_pred[:, 0]
        df = pd.DataFrame(result, columns=['x', 'y', 'u'])
        df.to_csv('./results/results_'+str(self.iter)+'.csv', index=False, header=None)

        return u_pred

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
        x = torch.tensor(data[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(data[:, 1:2], requires_grad=True).float().to(device)
        u_pred = self.net_u(x, y)
        u_pred = u_pred.detach().cpu().numpy()

        # 保存文件
        result = np.zeros([len(data), 3])
        result[:, 0] = data[:, 0]
        result[:, 1] = data[:, 1]
        result[:, 2] = u_pred[:, 0]
        df = pd.DataFrame(result, columns=['x', 'y', 'u'])
        df.to_csv('./results/curve_results_t=1.csv', index=False, header=None)

        return y_pred


if __name__ == "__main__":
    import os
    if not os.path.exists('results'):
        os.mkdir('results')

    # Network configuration
    uv_layers = [2] + 6*[75] + [1]

    # Domain bounds
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    lp = [0.0, 0.0] + [1.0, 0.0] * lhs(2, 301)
    lp[:, 0] = np.linspace(0.0, 1.0, 301)

    up = [0.0, 1.0] + [1.0, 0.0] * lhs(2, 301)
    up[:, 0] = np.linspace(0.0, 1.0, 301)

    lf = [0.0, 0.0] + [0.0, 1.0] * lhs(2, 301)
    lf[:, 1] = np.linspace(0.0, 1.0, 301)

    ur = [1.0, 0.0] + [0.0, 1.0] * lhs(2, 301)
    ur[:, 1] = np.linspace(0.0, 1.0, 301)

    BC = np.concatenate((lp[:, 0:2], up[:, 0:2], lf[:, 0:2], ur[:, 0:2]), 0)

    eps = 1*np.pi
    u_BC = np.sin(-6*eps*BC[:, 1:2]) * np.exp(-3*eps*BC[:, 0:1])

    plt.plot(BC[:, 0:1], u_BC)
    plt.show()
    plt.plot(BC[:, 1:2], u_BC)
    plt.show()

    a = np.linspace(0.0, 1.0, 151)
    b = np.linspace(0.0, 1.0, 151)
    Field = np.empty(shape=[151*151, 2])
    for i in range(151):
        for j in range(151):
            Field[i*151+j, 0] = b[j]
            Field[i*151+j, 1] = a[i]

    plt.scatter(Field[:, 0], Field[:, 1])
    plt.scatter(BC[:, 0], BC[:, 1])
    plt.show()

    XY_c = np.concatenate((Field[:, 0:2], BC[:, 0:2]), 0)

    model = PhysicsInformedNN(XY_c, BC, uv_layers, lb, ub, eps)
    model.train_LBFGS()




