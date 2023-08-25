import ot
import ot.plot
import matplotlib.pyplot as plt
import numpy as np

class FlattenPhaseExtractor:
    def __init__(self, method='sinkhorn', reg=0.01, plot_plan=False):
        self.method = method
        self.reg = reg
        self.plot_plan = plot_plan

    def __call__(self, input_intensity, target_intensity):
        # normalize input intensities
        input_intensity = self.normalize(input_intensity)
        target_intensity = self.normalize(target_intensity)
        
        assert input_intensity.shape == target_intensity.shape, \
        'input and target arrays must have the same shape'

        # compute transport plan
        Ny, Nx = input_intensity.shape
        xs, ys = self.get_coordinate_system(Nx, Ny)
        cost_M = self.get_cost_matrix(xs, ys)
        G0 = self.get_transport_plan(
            input_intensity, target_intensity, cost_M, 
            self.method, self.reg, self.plot_plan
        )

        # extract phase from transport plan
        Fx, Fy = self.get_first_moments(G0, xs, ys)
        curl = self.get_curl(Fx, Fy)
        phase = self.get_phase(Fx, Fy)
        return Fx, Fy, phase, curl
    
    @staticmethod
    def normalize(intensity):
        """Normalizes intensity of an image to 1"""
        return intensity / intensity.sum()

    @staticmethod
    def get_coordinate_system(Nx, Ny):
        """Creates a coordinate system"""
        xs, ys = np.meshgrid(
            np.arange(-Nx//2, Nx//2, dtype='float64'), 
            np.arange(-Ny//2, Ny//2, dtype='float64')
        )
        return xs, ys
    
    @staticmethod
    def get_cost_matrix(xs, ys):
        """Creates a cost matrix for a flattened array"""
        coords = np.array([ys, xs]).transpose() 
        coords_flat = coords.reshape(-1, 2)
        M = ot.dist(coords_flat, metric='sqeuclidean') 
        M /= M.max()
        return M
    
    @staticmethod
    def get_transport_plan(input_intensity, target_intensity, cost_matrix, method, reg, plot_plan):
        """Computes transport plan"""
        input_flatten = input_intensity.flatten()
        target_flatten = target_intensity.flatten()
        if method == 'sinkhorn':
            G0 = ot.sinkhorn(input_flatten, target_flatten, cost_matrix, reg)
        elif method == 'emd': 
            G0 = ot.emd(input_flatten, target_flatten, cost_matrix)
        else:
            raise ValueError("method must be one of {'sinkhorn', 'emd'}")
        if plot_plan:
            plt.figure(figsize=(10, 10))
            ot.plot.plot1D_mat(input_flatten, target_flatten, G0, 'OT matrix G0')
        return G0
    
    @staticmethod
    def get_first_moments(G0, xs, ys):
        """Extracts first moments out of transport plan"""
        # normalizes each row of the transport plan
        G0_norm = G0 / G0.sum(axis=1)[:, np.newaxis]
        
        # flatten coordinates
        xs_flatten = xs.flatten()
        ys_flatten = ys.flatten()

        # duplicate for each row of the transport plan
        # TODO change np.repat -> np.tile
        x_matrix = np.repeat(xs_flatten[np.newaxis, :], G0_norm.shape[0], axis=0)
        y_matrix = np.repeat(ys_flatten[np.newaxis, :], G0_norm.shape[0], axis=0)

        # compute first moments
        x_moment = np.sum(G0_norm * x_matrix, axis=1)
        y_moment = np.sum(G0_norm * y_matrix, axis=1)

        # reshape into a 2d map
        x_moment = x_moment.reshape(xs.shape)
        y_moment = y_moment.reshape(xs.shape)
        
        return x_moment, y_moment

    @staticmethod
    def get_curl(Fx, Fy):
        """Computes curl of a vector field"""
        dFydx = np.diff(Fy, axis=1)[:-1,:]
        dFxdy = np.diff(Fx, axis=0)[:,:-1]
        curl = dFydx - dFxdy 
        return curl
    
    @staticmethod
    def get_phase(Fx, Fy, x0=0, y0=0):
        """Assuming extracted field (Fx, Fy) has 0 curl. 
        We can parametrize the path from (x0, y0) to (x, y) 
        by splitting it into (x0, y0) -> (x, y0) -> (x, y). 
        This gives us phase up to a constant offset."""
        
        # extract integrants
        Fx_at_x_y0 = np.tile(Fx[y0,:], (Fx.shape[0], 1))
        Fy_at_x_y = Fy
        
        # integrates from (0, y0) to (x, y0)
        X_path = np.cumsum(Fx_at_x_y0, axis=1)
        
        # subtracts integral from (0, y0) to (x0, y0)
        X_path -= np.tile(X_path[:,x0][:, np.newaxis], (1, Fx.shape[1]))
        
        # integrates from (x, 0) to (x, y)    
        Y_path = np.cumsum(Fy_at_x_y, axis=0)

        # subtracts integral from (x, 0) to (x, y0)
        Y_path -= np.tile(Y_path[y0,:], (Fy.shape[0], 1))
        
        return X_path + Y_path