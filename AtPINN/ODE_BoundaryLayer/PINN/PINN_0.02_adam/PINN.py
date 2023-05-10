import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

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

        self.lb = lb
        self.ub = ub

        self.eps = eps
        self.x_cc = Collo[:, 0:1]

        # Collocation point
        self.x_c = torch.tensor(Collo[:, 0:1], requires_grad=True).float().to(device)
        self.x_BC = torch.tensor(BC[:, 0:1]).float().to(device)
        self.u_BC = torch.tensor(BC[:, 1:2]).float().to(device)

        # Define layers
        self.uv_layers = uv_layers

        # deep neural networks
        self.dnn = DNN(uv_layers, self.lb, self.ub).to(device)

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=0.0001)

        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_bc_list = []

    def net_u(self, x):
        y = self.dnn(x)
        return y

    def net_f(self, x):
        y = self.net_u(x)
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
        y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x), retain_graph=True, create_graph=True)[0]
        e1 = self.eps*y_xx + y_x + y

        return e1

    def train_adam(self):
        for self.iter in range(10000):
            self.optimizer.step()
            self.optimizer.zero_grad()

            e1 = self.net_f(self.x_c)
            loss_f = torch.mean(e1 ** 2)

            u_bc_prediction = self.net_u(self.x_BC)
            loss_bc = torch.mean((self.u_BC - u_bc_prediction) ** 2)

            loss = loss_f + loss_bc

            self.loss_f_list.append(loss_f.item())
            self.loss_bc_list.append(loss_bc.item())
            self.loss_list.append(loss.item())

            loss.backward()
            self.iter += 1
            self.iter_list.append(self.iter)

            if self.iter % 1000 == 0:
                print('Iter %d, Loss: %.5e, Loss_f: %.5e, Loss_BC: %.5e' %
                      (self.iter, loss.item(), loss_f.item(), loss_bc.item()))

        torch.save(self.dnn.state_dict(), 'model_adam.pth')
        torch.save(self.dnn.state_dict(), 'model_based.pth')

        self.loss_save('loss')

    def loss_save(self, filename):
        loss_data = np.array([self.iter_list, self.loss_list, self.loss_f_list, self.loss_bc_list])
        loss_data = np.transpose(loss_data)
        df_loss_data = pd.DataFrame(loss_data, columns=['iter', 'loss', 'loss_f', 'loss_bc'])
        df_loss_data.to_csv('./results/' + filename + '.csv', index=False)

    def predict_by_trained(self, x, filename):
        self.dnn.load_state_dict(torch.load(filename))
        x_ = torch.tensor(x, requires_grad=True).float().to(device)
        y_prediction_temp = self.dnn(x_)
        y_prediction_temp = y_prediction_temp.detach().cpu().numpy()

        df_result = pd.DataFrame(np.concatenate((x, y_prediction_temp), 1), columns=['x', 'y_prediction'])
        df_result.to_csv('./results/result_' + str(self.iter) + '.csv', index=False, header=False)

        plt.subplots(figsize=(3.3, 3.3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tick_params(labelsize=11)
        plt.plot(x, y_prediction_temp)
        font = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'normal', 'size': 12}
        plt.xlabel('x', fontdict=font)
        plt.ylabel('y', fontdict=font)
        plt.savefig('./results/Figure_'+str(self.iter)+'.png', dpi=300, bbox_inches='tight')
        plt.close()

        return y_prediction_temp

    def visualization(self, filename):
        self.dnn.load_state_dict(torch.load(filename))
        for k, v in self.dnn.named_parameters():
            name = k
            df_weight = pd.DataFrame(v.detach().cpu().numpy())
            df_weight.to_csv(name + '.csv', header=False, index=False)


if __name__ == "__main__":
    import os
    if not os.path.exists('results'):
        os.mkdir('results')

    # Network configuration
    uv_layers = [1] + 4*[30] + [1]

    a = 0
    b = 1
    # Domain bounds
    lb = np.array([a])
    ub = np.array([b])

    x = np.linspace(a, b, 1001)
    x = x.reshape([1001, 1])

    eps = 0.02
    BC = np.array([[a, 0], [b, 1]])

    model = PhysicsInformedNN(x, BC, uv_layers, lb, ub, eps)

    model.train_adam()

    y_prediction = model.predict_by_trained(x, r'model_adam.pth')
    data = np.concatenate((x, y_prediction), 1)
    df = pd.DataFrame(data, columns=['x', 'y_prediction'])
    df.to_csv('result_PINN_0.02.csv', index=False, header=False)
