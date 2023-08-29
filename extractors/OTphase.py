import ot
import ot.plot
import matplotlib.pyplot as plt
import numpy as np

class OptimalTransport1d:
    def __init__(self, method='sinkhorn', reg=0.01, plot_plan=False):
        self.method = method
        self.reg = reg
        self.plot_plan = plot_plan
    
    def __call__(self, input, target, r):
        # normalize input distributions
        input = self.normalize(input)
        target = self.normalize(target)

        assert input.shape == target.shape, \
        'input and target arrays must have the same shape'
        
        assert len(input.shape) == 1, \
        'input and target arrays must be 1d'

        # compute OT
        cost_matrix = self.get_cost_matrix(r)
        G0 = self.get_transport_plan(
            input, target, cost_matrix, 
            self.method, self.reg, self.plot_plan
        )

        # extract phase derivative
        dphi_dr = self.get_phase_derivative(G0, r)
        return dphi_dr


    @staticmethod
    def normalize(x):
        """Normalizes intensity of an image to 1"""
        return x / x.sum()

    @staticmethod
    def get_cost_matrix(r):
        """Creates a cost matrix for a flattened array"""
        M = ot.dist(r.reshape((-1, 1)), r.reshape((-1, 1)))
        M /= M.max()
        return M
    
    @staticmethod
    def get_transport_plan(input, target, cost_matrix, method, reg, plot_plan):
        """Computes transport plan"""
        if method == 'sinkhorn':
            G0 = ot.sinkhorn(input, target, cost_matrix, reg, numItermax=1000000, numThreads='max')
        elif method == 'emd': 
            G0 = ot.emd(input, target, cost_matrix, numItermax=1000000, numThreads='max')
        else:
            raise ValueError("method must be one of {'sinkhorn', 'emd'}")
        if plot_plan:
            plt.figure(figsize=(10, 10))
            ot.plot.plot1D_mat(input, target, G0, 'OT matrix G0')
        return G0

    @staticmethod
    def get_phase_derivative(G0, r):
        G0_norm = G0 / G0.sum(axis=1)[:, np.newaxis]
        dphi_dr = G0_norm @ r
        return dphi_dr


class FlattenOptimalTransport2d(OptimalTransport1d):
    def __init__(self, method='sinkhorn', reg=0.01, plot_plan=False):
        self.method = method
        self.reg = reg
        self.plot_plan = plot_plan

    def __call__(self, input_intensity, target_intensity, xs, ys):
        # normalize input intensities
        input_intensity = self.normalize(input_intensity)
        target_intensity = self.normalize(target_intensity)

        assert input_intensity.shape == target_intensity.shape, \
        'input and target arrays must have the same shape'

        # flatten images
        input_flat = input_intensity.flatten()
        target_flat = target_intensity.flatten()

        # compute transport plan
        cost_matrix = self.get_cost_matrix(xs, ys)
        G0 = self.get_transport_plan(
            input_flat, target_flat, cost_matrix, 
            self.method, self.reg, self.plot_plan
        )

        # extract phase from transport plan
        Fx, Fy = self.get_first_moments(G0, xs, ys)
        dx, dy = np.abs(xs[0,0] - xs[0,1]), np.abs(ys[0,0] - ys[1,0])
        curl = self.get_curl(Fx, Fy, dx, dy)
        phase = self.get_phase(Fx, Fy, dx, dy)
        return Fx, Fy, phase, curl
    
    
    @staticmethod
    def get_cost_matrix(xs, ys):
        """Creates a cost matrix for a flattened array"""
        coords = np.array([ys, xs]).transpose() 
        coords_flat = coords.reshape(-1, 2)
        M = ot.dist(coords_flat, metric='sqeuclidean') 
        M /= M.max()
        return M
    
    @staticmethod
    def get_first_moments(G0, xs, ys):
        """Extracts first moments out of transport plan"""
        # normalizes each row of the transport plan
        G0_norm = G0 / G0.sum(axis=1)[:, np.newaxis]
        
        # flatten coordinates
        xs_flatten = xs.flatten()
        ys_flatten = ys.flatten()

        # compute first moments
        x_moment = G0_norm @ xs_flatten
        y_moment = G0_norm @ ys_flatten

        # reshape into a 2d map
        x_moment = x_moment.reshape(xs.shape)
        y_moment = y_moment.reshape(xs.shape)
        
        return x_moment, y_moment
    

    @staticmethod
    def get_phase(Fx, Fy, dx, dy):
        """Assuming extracted field (Fx, Fy) has 0 curl. 
        We can parametrize the path from (0, 0) to (x, y) 
        by splitting it into (0, 0) -> (x, 0) -> (x, y). 
        This gives us phase up to a constant offset."""
        
        # extract integrants
        Fx_at_x_0 = np.tile(Fx[0,:], (Fx.shape[0], 1))
        Fy_at_x_y = Fy
        
        # integrates from (0, 0) to (x, 0)
        X_path = np.cumsum(Fx_at_x_0, axis=1) * dx
        
        # integrates from (x, 0) to (x, y)  
        Y_path = np.cumsum(Fy_at_x_y, axis=0) * dy
        
        return X_path + Y_path
    

    @staticmethod
    def get_curl(Fx, Fy, dx, dy):
        """Computes curl of a vector field"""
        dFydx = np.diff(Fy, axis=1)[:-1,:]/dx
        dFxdy = np.diff(Fx, axis=0)[:,:-1]/dy
        curl = dFydx - dFxdy 
        return curl
    
