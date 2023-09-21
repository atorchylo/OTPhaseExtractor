import torch
from torch import nn
from torch import optim
import numpy as np
from ot import stochastic
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import FlattenOptimalTransport2d

class Potential(nn.Module):
    def __init__(self, num_layers = 5, hidden_size_0 = 32):
        super(Potential, self).__init__()
        self.linear_list = nn.ModuleList([])
        self.linear_list.append(nn.Linear(2, hidden_size_0))
        for i in range(1, num_layers-1):
            self.linear_list.append(nn.Linear(hidden_size_0 * 2**(i-1), hidden_size_0 * 2**i))
        self.last_linear = nn.Linear(hidden_size_0 * 2**i, 1)
        self.relu = nn.ReLU() 

    def forward(self, x):
        for linear_layer in self.linear_list:
            x = linear_layer(x)
            x = self.relu(x)
        x = self.last_linear(x)
        return x.ravel()

# ContinuousOptimalTransport inherets few methods from FlattenOptimalTransport2d
class ContinuousOptimalTransport2d(FlattenOptimalTransport2d):
    def __init__(
        self, 
        potential_model = Potential, 
        model_parameters = {}, 
        device=torch.device('cpu'),
        reg = 0.2, 
        lr = .0005,
        batch_size = 512,
        n_iter = 300,
        print_every = 50,
        plot_loss = True,
    ):
        """Initializes ContinuousOptimalTransport model"""
        self.potential_model = potential_model
        self.model_parameters = model_parameters
        self.device = device
        self.reg = reg
        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.plot_loss = plot_loss
        self.print_every = print_every

    def __call__(self, input, target, xs, ys):
        # create input
        input = self.normalize(input)
        target = self.normalize(target)

        # define models
        u_potential = self.potential_model(**self.model_parameters).to(self.device)
        v_potential = self.potential_model(**self.model_parameters).to(self.device)
        print(u_potential)

        # infer the map
        optimizer = optim.Adam(list(u_potential.parameters()) + list(v_potential.parameters()), lr=self.lr)
        self.train(input, target, u_potential, v_potential, xs, ys, optimizer)
        Fx, Fy = self.get_first_moments(u_potential, v_potential, xs, ys)

        # computer phase
        dx, dy = np.abs(xs[0,0] - xs[0,1]), np.abs(ys[0,0] - ys[1,0])
        curl = self.get_curl(Fx, Fy, dx, dy)
        phase = self.get_phase(Fx, Fy, dx, dy)
        return Fx, Fy, phase, curl

    @staticmethod
    def normalize(x):
        """Normalizes intensity of an image to 1"""
        return x / x.sum()
    
    def random_sample(self, intensity, xs, ys, nsamples):
        # sample points
        intensity_flat = intensity.flatten()
        sample_index = np.random.choice(a=intensity_flat.size, size=nsamples, p=intensity_flat)
        y_samples, x_samples = np.unravel_index(sample_index, intensity.shape)

        # add random subpixel noise
        y_samples = y_samples + np.random.rand(y_samples.shape[0]) 
        x_samples = x_samples + np.random.rand(x_samples.shape[0])
        
        # convert to coordinates
        dx, dy = np.abs(xs[0,0] - xs[0,1]), np.abs(ys[0,0] - ys[1,0])
        y_samples = y_samples * dy + ys.min()
        x_samples = x_samples * dx + xs.min()
        return np.array([x_samples, y_samples]).T
    
    def train(self, input, target, u_potential, v_potential, xs, ys, optimizer):
        losses = []
        for i in tqdm(range(self.n_iter)):
            # generate samples
            source_points = self.random_sample(input, xs, ys, self.batch_size)
            target_points = self.random_sample(target, xs, ys, self.batch_size)
            
            # push to device
            source_points = torch.tensor(source_points, dtype=torch.float)
            target_points = torch.tensor(target_points, dtype=torch.float)

            # minus because we maximize te dual loss
            loss = -stochastic.loss_dual_entropic(
                u_potential(source_points), 
                v_potential(target_points), 
                source_points, 
                target_points, 
                reg=self.reg
            )
            losses.append(loss.item())

            if i % self.print_every == 0:
                print("Iter: {:3d}, loss={}".format(i, losses[-1]))

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if self.plot_loss:
            plt.figure(figsize=(10, 7))
            plt.plot(losses)
            plt.grid()
            plt.title('Dual objective (negative)')
            plt.xlabel("Iterations")

    def get_first_moments(self, u_potential, v_potential, xs, ys):
        # create coordinates -- inputs to potnetial neural networks
        cooridnates = np.concatenate((xs.ravel()[:, None], ys.ravel()[:, None]), axis=1)
        cooridnates = torch.tensor(cooridnates, dtype=torch.float).to(self.device)
        xs_tensor = torch.tensor(xs, dtype=torch.float).to(self.device)
        ys_tensor = torch.tensor(ys, dtype=torch.float).to(self.device)

        # iterate over coordinates, computing moments
        batch_size = 1024
        Xmoments = []
        Ymoments = []
        with torch.no_grad():
            u_potential.eval()
            v_potential.eval()
            for i in tqdm(range(0, len(cooridnates), batch_size)):
                Gg_plans = stochastic.plan_dual_entropic(
                    u_potential(cooridnates[i:i+batch_size]), 
                    v_potential(cooridnates), 
                    cooridnates[i:i+batch_size], 
                    cooridnates, 
                    reg=self.reg
                )
                Gg_plans = Gg_plans/Gg_plans.sum(axis=1).unsqueeze(1)
                Gg_plans = Gg_plans.reshape(batch_size, xs.shape[0], xs.shape[1])
                x_mom = (Gg_plans * xs_tensor.unsqueeze(0)).sum(axis=[1, 2])
                y_mom = (Gg_plans * ys_tensor.unsqueeze(0)).sum(axis=[1, 2])
                Xmoments.extend(x_mom.detach().numpy())
                Ymoments.extend(y_mom.detach().numpy())
        Fx = np.array(Xmoments).reshape(xs.shape)
        Fy = np.array(Ymoments).reshape(xs.shape)
        return Fx, Fy