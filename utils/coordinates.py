
import numpy as np

class RadialCoordinates:
    """
    1d coordinate system that has the same spacing in position 
    domain and fourier domain. [0, R], where R = np.sqrt(n)
    """
    def __init__(self, n):
        self.n = n
        self.dr = 1/np.sqrt(n)
        self.r = np.linspace(0, n * self.dr, n, endpoint=False)


class CenteredGridCoordinates:
    """
    2d coordinate system that has the same spacing in position 
    domain and fourier domain. [-X, X] x [-Y, Y], 
    where X = np.sqrt(nx)
    and Y = np.sqrt(ny)
    """
    def __init__(self, nx, ny):
        """ 
        Initializes SymmetricCenteredGrid class

        nx - number of x samples
        ny - number of y samples
        """
        self.nx = nx
        self.ny = ny
        self.dx = 1/np.sqrt(nx)
        self.dy = 1/np.sqrt(ny)

        # compute x coordinates
        if nx % 2 == 0:
            self.x = np.linspace(-(nx//2) * self.dx, nx//2 * self.dx, nx, endpoint=False)
        else:
            self.x = np.linspace(-(nx//2) * self.dx, nx//2 * self.dx, nx, endpoint=True)

        # compute y coordinates
        if ny % 2 == 0:
            self.y = np.linspace(-(ny//2) * self.dy, ny//2 * self.dy, ny, endpoint=False)
        else:
            self.y = np.linspace(-(ny//2) * self.dy, ny//2 * self.dy, ny, endpoint=True)
    
    def get_mesh(self):
        """
        Returns mesh grid using coordinates

        xs - x coordinates across the image; shape is [ny, nx]
        xs - y coordinates across the image; shape is [ny, nx]
        """
        xs, ys = np.meshgrid(self.x, self.y)
        return xs, ys
    
