import numpy as np
from utils.fourier import FFT2, IFFT2

class GerchbergSaxton2d: 
    def __init__(self, num_iter):
        self.num_iter = num_iter

    def __call__(self, source, target, phase=None):
        """
        Performs GS algorithm on source and target amplitude

        source - input amplitude 
        targetr - target amplitude
        phase - initial guess for the phase
        """
        phase = np.angle(IFFT2(target)) if phase is None else phase
        for i in range(self.num_iter):
            source_field = np.abs(source) * np.exp(1j * phase)
            phase = np.angle(FFT2(source_field))
            target_field = np.abs(target) * np.exp(1j * phase)
            phase = np.angle(IFFT2(target_field))
        return phase


