import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.fftpack import fftshift, ifftshift


def FFT2(position_data):
    """
    Computes a 2d fast fourier transform of centered input data.
    
    fourrier_data - centered fourier transform of data.
    """
    fourrier_data = fftshift(fft2(ifftshift(position_data)))
    return fourrier_data

def IFFT2(fourrier_data):
    """
    Computes a 2d inverse fast fourier transform of the centered input data.
    
    position_data - centered position data of the given fourier signal.
    """
    position_data = fftshift(ifft2(ifftshift(fourrier_data)))
    return position_data

def fourier_propagation(input_intensity, phase):
    """
    Computes fourier image for a given intensity and phase 
    """
    input_amplitude = np.sqrt(input_intensity / input_intensity.sum())
    output_amplitude = np.abs(FFT2(input_amplitude * np.exp(1j * phase)))
    output_intensity = output_amplitude ** 2
    output_intensity = output_intensity / (input_amplitude.shape[0] * input_amplitude.shape[1])
    return output_intensity