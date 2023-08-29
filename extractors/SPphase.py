import numpy as np

class SpotlightSolution1d:
    def __init__(self):
        pass
    
    def __call__(self, a, b, r):
        """
        Works only for 1d distributions

        Extracts derivative of the phase using spotlight integration method.
        This method is a numerical solution to Monge-Ampere differential eq.

        a - input 1d array
        b - target 1d array
        r - coordinates for both arrays
        """
        a = a/a.sum()
        b = b/b.sum()
        p_r = np.cumsum(a)
        P_r = np.cumsum(b)
        # numerically invert function
        P_inv = lambda u: r[np.absolute(P_r-u).argmin()]
        # computes P^-1(p(r))
        dphi_dr = np.array([P_inv(u) for u in p_r])
        return dphi_dr