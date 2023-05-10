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
    def __init__(self, Collo, Field, FAR_source, WALL_source, FAR_target, WALL_target, uv_layers, lb, ub, mu):

        self.lb = lb
        self.ub = ub

        self.mu = mu
        self.x_cc = Field[:, 0:2]

        # Collocation point
        self.x_ = torch.tensor(Collo[:, 0:1], requires_grad=True).float().to(device)
        self.y_ = torch.tensor(Collo[:, 1:2], requires_grad=True).float().to(device)

        self.x_FAR = torch.tensor(FAR_target[:, 0:1], requires_grad=True).float().to(device)
        self.y_FAR = torch.tensor(FAR_target[:, 1:2], requires_grad=True).float().to(device)
        self.p_FAR_target = torch.tensor(FAR_target[:, 2:3], requires_grad=True).float().to(device)
        self.u_FAR_target = torch.tensor(FAR_target[:, 3:4], requires_grad=True).float().to(device)
        self.v_FAR_target = torch.tensor(FAR_target[:, 4:5], requires_grad=True).float().to(device)

        self.p_FAR_source = torch.tensor(FAR_source[:, 2:3], requires_grad=True).float().to(device)
        self.u_FAR_source = torch.tensor(FAR_source[:, 3:4], requires_grad=True).float().to(device)
        self.v_FAR_source = torch.tensor(FAR_source[:, 4:5], requires_grad=True).float().to(device)

        self.x_WALL = torch.tensor(WALL_target[:, 0:1], requires_grad=True).float().to(device)
        self.y_WALL = torch.tensor(WALL_target[:, 1:2], requires_grad=True).float().to(device)
        self.p_WALL_target = torch.tensor(WALL_target[:, 2:3], requires_grad=True).float().to(device)
        self.p_WALL_source = torch.tensor(WALL_source[:, 2:3], requires_grad=True).float().to(device)

        # Define layers
        self.uv_layers = uv_layers

        # deep neural networks
        self.dnn = DNN(uv_layers, self.lb, self.ub).to(device)
        self.dnn.load_state_dict(torch.load('model_based.pth'))

        self.mu_ = torch.tensor([0.0], requires_grad=True).to(device)
        self.mu_ = torch.nn.Parameter(self.mu_)
        self.dnn.register_parameter('mu_', self.mu_)

        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1,
            max_iter=10000,
            max_eval=10000,
            history_size=300,
            tolerance_grad=1e-20,
            tolerance_change=1e-20
        )
        self.iter = 0
        self.iter_list = []
        self.mu_list = []
        self.loss_mu_list = []
        self.loss_f_list = []
        self.loss_FAR_list = []
        self.loss_WALL_list = []
        self.loss_list = []

    def net_u(self, x, y):
        psips = self.dnn(torch.cat([x, y], dim=1))
        p = psips[:, 0:1]
        u = psips[:, 1:2]
        v = psips[:, 2:3]
        return u, v, p

    def net_f(self, x, y, parameter):
        u, v, p = self.net_u(x, y)

        # Plane stress problem
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        e1 = u_x + v_y
        e2 = u * u_x + v * u_y + p_x - parameter * (u_xx + u_yy)
        e3 = u * v_x + v * v_y + p_y - parameter * (v_xx + v_yy)

        return e1, e2, e3

    def loss_func_lgbs(self):
        self.optimizer_LBFGS.zero_grad()

        mu_temp = self.mu_*(0.001-0.02) + 0.02

        p_WALL = self.mu_*(self.p_WALL_target - self.p_WALL_source) + self.p_WALL_source
        u_FAR = self.mu_*(self.u_FAR_target - self.u_FAR_source) + self.u_FAR_source
        v_FAR = self.mu_*(self.v_FAR_target - self.v_FAR_source) + self.v_FAR_source
        p_FAR = self.mu_*(self.p_FAR_target - self.p_FAR_source) + self.p_FAR_source

        e1, e2, e3 = self.net_f(self.x_, self.y_, mu_temp)
        loss_f = torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2)

        u_WALL_pred, v_WALL_pred, p_WALL_pred = self.net_u(self.x_WALL, self.y_WALL)
        loss_WALL = torch.mean(u_WALL_pred ** 2) + torch.mean(v_WALL_pred ** 2) + torch.mean((p_WALL - p_WALL_pred) ** 2)

        u_FAR_pred, v_FAR_pred, p_FAR_pred = self.net_u(self.x_FAR, self.y_FAR)
        loss_FAR = torch.mean((u_FAR-u_FAR_pred)**2)+torch.mean((v_FAR-v_FAR_pred)**2)+torch.mean((p_FAR-p_FAR_pred)**2)

        loss_mu = torch.abs(self.mu_ - self.mu)

        loss = loss_f * (loss_f.item()/0.0001) + loss_WALL + loss_FAR + loss_mu * 0.01

        self.loss_list.append(loss.item())
        self.mu_list.append(self.mu_.item())
        self.loss_f_list.append(loss_f.item())
        self.loss_WALL_list.append(loss_WALL.item())
        self.loss_FAR_list.append(loss_FAR.item())
        self.loss_mu_list.append(loss_mu.item())

        loss.backward()
        self.iter += 1
        self.iter_list.append(self.iter)

        if self.iter % 10 == 0:
            print('Iter %d, mu: %.5e, loss: %.5e, loss_f: %.5e, loss_FAR: %.5e, loss_WALL: %.5e' %
                  (self.iter, self.mu_.item(), loss.item(), loss_f.item(), loss_FAR.item(), loss_WALL.item()))

        if self.iter % 1000 == 0:
            torch.save(self.dnn.state_dict(), './results/model_'+str(self.iter)+'.pth')
            self.predict_by_trained(self.x_cc, './results/model_'+str(self.iter)+'.pth')
            self.Post_PINNs()

        return loss

    def LossSave(self, filename):
        data = np.array([self.iter_list, self.loss_list, self.mu_list, self.loss_mu_list, self.loss_f_list, self.loss_FAR_list, self.loss_WALL_list])
        data = np.transpose(data)
        df = pd.DataFrame(data, columns=['Iter', 'loss', 'mu', 'loss_mu', 'loss_f', 'loss_FAR', 'loss_WALL'])
        df.to_csv(filename + '.csv', index=False)

    def train_LBFGS(self):
        self.optimizer_LBFGS.step(self.loss_func_lgbs)
        torch.save(self.dnn.state_dict(), './results/model_LBFGS.pth')
        self.LossSave('Loss')

    def predict_by_trained(self, data, filename):
        self.dnn.load_state_dict(torch.load(filename))
        x = torch.tensor(data[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(data[:, 1:2], requires_grad=True).float().to(device)
        u_pred, v_pred, p_pred = self.net_u(x, y)

        u_pred = u_pred.detach().cpu().numpy()
        v_pred = v_pred.detach().cpu().numpy()
        p_pred = p_pred.detach().cpu().numpy()

        # 保存文件
        result = np.zeros([len(data), 5])
        result[:, 0] = data[:, 0]
        result[:, 1] = data[:, 1]
        result[:, 2] = p_pred[:, 0]
        result[:, 3] = u_pred[:, 0]
        result[:, 4] = v_pred[:, 0]

        return u_pred, v_pred, p_pred

    def visualization(self, filename):
        self.dnn.load_state_dict(torch.load(filename))
        for k, v in self.dnn.named_parameters():
            name = k
            data = v.detach().cpu().numpy()
            df = pd.DataFrame(data)
            df.to_csv(name + '.csv', header=None, index=False)


    def Post_PINNs(self):
        df = pd.read_csv('./results/results_'+str(self.iter)+'.csv', header=None)
        Field_result = df.values

        df = pd.read_csv('Mesh.csv', header=None)
        Mesh = df.values
        df = pd.read_csv('Field.csv', header=None)
        Field = df.values
        file = open('Field_result.dat', 'w')
        file.write('TITLE     = "results at NT=55, TAU=0.000000"\n')
        file.write('VARIABLES = "X"\n')
        file.write('"Y"\n')
        file.write('"P"\n')
        file.write('"U"\n')
        file.write('"V"\n')
        file.write('ZONE T="Zone_1"\n')
        file.write(' STRANDID=0, SOLUTIONTIME=0\n')
        file.write(' Nodes=23680, Elements=23305, ZONETYPE=FEQuadrilateral\n')
        file.write(' DATAPACKING=POINT\n')
        file.write(' DT=(SINGLE SINGLE SINGLE SINGLE SINGLE )\n')

        for i in range(len(Field_result)):
            file.write(str(Field_result[i, 0]))
            file.write('\t')
            file.write(str(Field_result[i, 1]))
            file.write('\t')
            file.write(str(np.sqrt(Field_result[i, 3]**2 + Field_result[i, 4]**2) - np.sqrt(Field[i, 3]**2 + Field[i, 4]**2)))
            file.write('\t')
            file.write(str(Field_result[i, 3]))
            file.write('\t')
            file.write(str(Field_result[i, 4]))
            file.write('\n')

        for i in range(len(Mesh)):
            file.write(str(Mesh[i, 0]))
            file.write('\t')
            file.write(str(Mesh[i, 1]))
            file.write('\t')
            file.write(str(Mesh[i, 2]))
            file.write('\t')
            file.write(str(Mesh[i, 3]))
            file.write('\n')

        df = pd.read_csv('./results/results_'+str(self.iter)+'.csv')
        data = df.values

        x_PINN = data[:, 0]
        y_PINN = data[:, 1]
        p_PINN = data[:, 2]
        u_PINN = data[:, 3]
        v_PINN = data[:, 4]

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(5, 6))
        fig.subplots_adjust(hspace=0.2, wspace=0.6)

        cf = ax[0, 0].scatter(x_PINN, y_PINN, c=p_PINN, alpha=1 - 0.1, edgecolors='none', cmap='rainbow', marker='o',
                              s=int(2))
        ax[0, 0].axis('square')
        ax[0, 0].set_xlim([-0.5, 1.5])
        ax[0, 0].set_ylim([-0.5, 0.5])
        fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)

        cf = ax[1, 0].scatter(x_PINN, y_PINN, c=u_PINN, alpha=0.7 - 0.1, edgecolors='none', cmap='rainbow', marker='o',
                              s=int(2))
        ax[1, 0].axis('square')
        ax[1, 0].set_xlim([-0.5, 1.5])
        ax[1, 0].set_ylim([-0.5, 0.5])
        fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)

        cf = ax[2, 0].scatter(x_PINN, y_PINN, c=v_PINN, alpha=1 - 0.1, edgecolors='none', cmap='rainbow', marker='o',
                              s=int(2))
        ax[2, 0].axis('square')
        ax[2, 0].set_xlim([-0.5, 1.5])
        ax[2, 0].set_ylim([-0.5, 0.5])
        fig.colorbar(cf, ax=ax[2, 0], fraction=0.046, pad=0.04)

        df = pd.read_csv('Field.csv', header=None)
        data = df.values
        x_PINN = data[:, 0]
        y_PINN = data[:, 1]
        p_PINN = data[:, 2]
        u_PINN = data[:, 3]
        v_PINN = data[:, 4]

        cf = ax[0, 1].scatter(x_PINN, y_PINN, c=p_PINN, alpha=1 - 0.1, edgecolors='none', cmap='rainbow', marker='o', s=int(2))
        ax[0, 1].axis('square')
        ax[0, 1].set_xlim([-0.5, 1.5])
        ax[0, 1].set_ylim([-0.5, 0.5])
        fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)

        cf = ax[1, 1].scatter(x_PINN, y_PINN, c=u_PINN, alpha=0.7 - 0.1, edgecolors='none', cmap='rainbow', marker='o',
                              s=int(2))
        ax[1, 1].axis('square')
        ax[1, 1].set_xlim([-0.5, 1.5])
        ax[1, 1].set_ylim([-0.5, 0.5])
        fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)

        cf = ax[2, 1].scatter(x_PINN, y_PINN, c=v_PINN, alpha=1 - 0.1, edgecolors='none', cmap='rainbow', marker='o',
                              s=int(2))
        ax[2, 1].axis('square')
        ax[2, 1].set_xlim([-0.5, 1.5])
        ax[2, 1].set_ylim([-0.5, 0.5])
        fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)
        plt.savefig('./results/Figure_'+str(self.iter)+'.png', dpi=300, bbox_inches='tight')

        plt.close()

    def post_processing(self):
        import shutil

        for i in range(10000):
            self.iter += 1
            if self.iter % 1000 == 0:
                self.predict_by_trained(self.x_cc, './results/model_'+str(self.iter)+'.pth')
                self.Post_PINNs()
                shutil.copyfile('Field_result.dat', 'Field_result_'+str(self.iter)+'.dat')


if __name__ == "__main__":
    import os
    if not os.path.exists('results'):
        os.mkdir('results')

    # Network configuration
    uv_layers = [2] + 4*[100] + [3]

    # Domain bounds
    lb = np.array([-0.15, -0.1])
    ub = np.array([1.1, 0.1])

    # 划定区域
    df_origin = pd.read_csv('Field.csv', header=None)
    df_origin.columns = ['x', 'y', 'p', 'u', 'v']
    df = df_origin[(df_origin['x'] <= 1.5) & (df_origin['x'] >= -0.5) & (df_origin['y'] <= 0.5) & (df_origin['y'] >= -0.5)]
    Field = df.values
    print(Field.shape)

    # 多一些离散点
    Field_2 = [-0.5, -0.5] + [2, 1] * lhs(2, 5000)
    df1 = pd.DataFrame(Field_2, columns=['x', 'y'])
    df2 = df1[((df1['x'] < 0) | (df1['x'] >= 1.1)) | ((df1['y'] <= -0.1) | (df1['y'] >= 0.1))]
    data2 = df2.values

    # 读取边界数据
    df = pd.read_csv('left.csv', header=None)
    left = df.values

    df = pd.read_csv('low.csv', header=None)
    low = df.values

    df = pd.read_csv('right.csv', header=None)
    right = df.values

    df = pd.read_csv('up.csv', header=None)
    up = df.values

    df = pd.read_csv('WALL.csv', header=None)
    WALL_target = df.values

    FAR_target = np.concatenate((left, low, right, up), 0)

    df = pd.read_csv('left_source.csv', header=None)
    left = df.values

    df = pd.read_csv('low_source.csv', header=None)
    low = df.values

    df = pd.read_csv('right_source.csv', header=None)
    right = df.values

    df = pd.read_csv('up_source.csv', header=None)
    up = df.values

    df = pd.read_csv('WALL_source.csv', header=None)
    WALL_source = df.values

    FAR_source = np.concatenate((left, low, right, up), 0)

    eps = 1

    data3 = [-0.5, -0.25] + [0.5, 0.5] * lhs(2, 2000)
    XY_c = np.concatenate((left[:, 0:2], low[:, 0:2], right[:, 0:2], up[:, 0:2], Field[:, 0:2], data2[:, 0:2], data3[:, 0:2]), 0)

    plt.scatter(XY_c[:, 0:1], XY_c[:, 1:2])
    plt.show()

    df = pd.read_csv('Field.csv', header=None)
    Field = df.values
    model = PhysicsInformedNN(XY_c, Field, FAR_source, WALL_source, FAR_target, WALL_target, uv_layers, lb, ub, eps)
    model.train_LBFGS()

    df = pd.read_csv('Field_matlab.csv', header=None)
    data = df.values
    print(len(data))
    u_pred, v_pred, p_pred = model.predict_by_trained(data, './results/model_10000.pth')
    result = np.zeros([len(data), 5])
    result[:, 0] = data[:, 0]
    result[:, 1] = data[:, 1]
    result[:, 2] = p_pred[:, 0]
    result[:, 3] = u_pred[:, 0]
    result[:, 4] = v_pred[:, 0]
    df = pd.DataFrame(result, columns=['x', 'y', 'p', 'u', 'v'])
    df.to_csv('Field_matlab_result.csv', index=False, header=None)

