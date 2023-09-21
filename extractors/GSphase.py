import numpy as np
from utils.fourier import FFT2, IFFT2
from tqdm import tqdm

class GerchbergSaxton2d: 
    def __init__(self, num_iter):
        self.num_iter = num_iter

    def __call__(self, input, target, phase=None):
        """
        Performs GS algorithm on source and target amplitude

        input - input intensity 
        target - target intensity
        phase - initial guess for the phase
        """
        # convert to amplitude
        input_amplitude = np.sqrt(input/input.sum())
        target_amplitude = np.sqrt(target/target.sum())

        # initial phase guess
        if phase is None:
            phase = np.angle(IFFT2(target_amplitude))
        
        # optimize
        for i in tqdm(range(self.num_iter)):
            source_field = np.abs(input_amplitude) * np.exp(1j * phase)
            phase = np.angle(FFT2(source_field))
            target_field = np.abs(target_amplitude) * np.exp(1j * phase)
            phase = np.angle(IFFT2(target_field))
        return phase


