# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:34:10 2016

Module with signal transformations. Usually used to transform bigger pieces
of signal (>1s). Some methods might use feature extractions.

@author: Anderson Brito da Silva, Jan Cimbalnik
"""

import numpy as np
import scipy.signal as sig

def compute_hilbert_envelope(signal):
    '''
    Calcule the Hilbert envelope

    Parameters:
    ----------
        signal - numpy array
    '''
    return np.abs(sig.hilbert(sig.detrend(signal)))

def compute_hilbert_energy(signal):
    '''
    Calcule the Hilbert energy

    Parameters:
    ----------
        signal - numpy array
    '''
    return np.abs(sig.hilbert(sig.detrend(signal)))**2

def compute_teager_energy(signal):
    '''
    Calcule the Teager energy

    Parameters:
    ----------
        signal - numpy array
    '''
    sqr = np.power(signal[1:-1],2)
    odd = signal[:-2]
    even = signal[2:]
    energy = sqr-odd*even
    energy = np.append(energy[0],energy)
    energy = np.append(energy,energy[-1])
    return energy

def compute_rms(signal, window_size = 6):
    '''
    Calcule the Root Mean Square (RMS) energy

    Parameters:
    ----------
        signal - numpy array
        window_size - number of the points of the window
    '''
    aux = np.power(signal,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(aux, window, 'same'))


def compute_stenergy(signal, window_size = 6):
    '''
    Calcule Short Time energy -
    Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721-31.

    Parameters:
    ----------
        signal - numpy array
        window_size - number of the points of the window
    '''
    aux = np.power(signal,2)
    window = np.ones(window_size)/float(window_size)
    return np.convolve(aux, window, 'same')


def compute_line_lenght(signal, window_size = 6):
    '''
    Calcule Short time line leght -
    Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721-31.

    Parameters:
    ----------
        signal - numpy array
        window_size - number of the points of the window
    '''
    aux = np.abs(np.diff(signal))
    window = np.ones(window_size)/float(window_size)
    data =  np.convolve(aux, window, 'same')
    data = np.append(data,data[-1])
    return data
    
def compute_stockwell_transform(signal, fs, min_freq, max_freq, f_fs = 1,
                                factor = 1):
    """
    Calculates Stockwell transform - 
    Localization of the Complex Spectrum: The S Transform
    IEEE Transactions on Signal Processing, vol. 44., number 4,
    April 1996, pages 998-1001.
    
    Parameters:
    -----------
    signal - numpy array\n
    min_freq - minimum frequency of ST\n
    max_freq - maximum frequency of ST\n
    fs - sampling frequency of the signal\n
    f_fs - is the frequency-sampling interval you desire in the ST result\n
    
    Returns:
    --------
    st -a complex matrix containing the Stockwell transform. \n
        The rows of STOutput are the frequencies and the columns are the time
        values ie each column is the "local spectrum" for that point in time\n
    t - a vector containing the sampled times\n
    f - a vector containing the sampled frequencies\n
    """
    
    # Calculate the sampled time and frequency values
    # from the two sampling rates
    
    samp_rate = 1 / fs
  
    t = np.linspace(0, len(signal)-1,len(signal)) * samp_rate
    spe_nelements = np.ceil((max_freq - min_freq+1)/f_fs)
    f = (min_freq + np.linspace(0,spe_nelements-1,spe_nelements)*f_fs)/(samp_rate*len(signal))

    # Compute the length of the data
    n = len(signal)
    
    # Compute FFT's
    vector_fft = np.fft.fft(signal)
    vector_fft = np.concatenate((vector_fft, vector_fft))

    # Preallocate output matrix
    st = np.zeros((int(np.ceil((max_freq - min_freq+1)/f_fs)), n),
                  dtype=complex)
    
    # Start computing the S_transform
    if min_freq == 0:
        st[0,:] = np.mean(signal)*np.ones(n) #///
    else:
        st[0,:] = np.fft.ifft(vector_fft[min_freq:min_freq+n]*\
                              _g_window(n, min_freq, factor))
    
    for i in np.linspace(f_fs, (max_freq - min_freq), (max_freq - min_freq)):
        st[int((i-1)/f_fs+1),:] = np.fft.ifft(vector_fft[int(min_freq+i):\
                                                         int(min_freq+i+n)]\
                                            * _g_window(n, min_freq+i, factor))
    
        
    return st, t, f
        

def _g_window(length, freq, factor):
    """
    Function to compute the Gaussion window for 
    function compute_stockwell_transform.
    
    length-the length of the Gaussian window
    freq-the frequency at which to evaluate the window.
    factor- the window-width factor
    
    -----Outputs Returned--------------------------
    
    gauss-The Gaussian window
    """

    vector = np.zeros((2, length))
    vector[0,:] = np.linspace(0, length - 1, length)
    vector[1,:] = np.linspace(-length, -1, length)
    vector = np.square(vector)
    vector = vector*(-factor*2*np.square(np.pi) / np.square(freq))
    gauss = np.sum(np.exp(vector), axis=0)  # Gaussian window
    
    return gauss
