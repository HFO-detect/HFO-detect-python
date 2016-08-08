# -*- coding: utf-8 -*-
"""
Library of functions for simulated data creation

Created on 07/04/2016
@author: Anderson Brito da Silva & Jan Cimbalnik
"""
import numpy as np
import scipy.signal as sig
from scipy.stats import norm

# %% Noise types

def pinknoise(N):
    """
    Create a pink noise (1/f) with N points.
    
    Parameters:
    ----------
    N - Number of samples to be returned
    """
    M = N
    # ensure that the N is even
    if N % 2:     
        N += 1
        
    x = np.random.randn(N)  # generate a white noise
    X = np.fft.fft(x) #FFT
    
    # prepare a vector for 1/f multiplication
    nPts = int(N/2 + 1)
    n = range(1,nPts+1)
    n = np.sqrt(n)
    
    #multiplicate the left half of the spectrum
    X[range(nPts)] = X[range(nPts)]/n
    
    #prepare a right half of the spectrum - a copy of the left one
    X[range(nPts,N)] = np.real(X[range(int(N/2-1),0,-1)]) - 1j*np.imag(X[range(int(N/2-1),0,-1)])
  
    y = np.fft.ifft(X)  #IFFT
    
    y = np.real(y)
    # normalising
    y -= np.mean(y)
    y /= np.sqrt(np.mean(y**2))
    # returning size of N
    if M % 2 == 1:
        y = y[:-1]
    return y

def brownnoise(N):
    """
    Create a brown noise (1/f²) with N points.
    
    Parameters:
    ----------
    N - Number of samples to be returned
    """
    M = N
    # ensure that the N is even
    if N % 2:
        N += 1
        
    x = np.random.randn(N) # generate a white noise
    
    X = np.fft.fft(x) #FFT
    
    # prepare a vector for 1/f² multiplication
    nPts = int(N/2 + 1)
    n = range(1,nPts+1)
   
    #multiplicate the left half of the spectrum
    X[range(nPts)] = X[range(nPts)]/n
    #prepare a right half of the spectrum - a copy of the left one
    X[range(nPts,N)] = np.real(X[range(int(N/2-1),0,-1)]) - 1j*np.imag(X[range(int(N/2-1),0,-1)])
    
    y = np.fft.ifft(X) #IFFT
    
    y = np.real(y)
    # normalising    
    y -= np.mean(y)
    y /= np.sqrt(np.mean(y**2))
    # returning size of N
    if M % 2 == 1:
        y = y[:-1]
    return y
    
# %% Artifacts
    
def delta(srate = 5000, decay_dur = None):
    """
    Delta function with exponential decay.
    
    Parameters:
    -----------
    decay_dur - decay duration before returning to 0\n
    
    Returns:
    --------
    delta - numpy array\n
    """
    
    if decay_dur is None:
        decay_dur = np.random.random()
        
    decay_N = int(srate * decay_dur)
    return_value = 0.001  # This is the value where decay finishes
    decay_factor = np.log(return_value)/-decay_N
    t = np.linspace(0,decay_N,decay_N, endpoint=False)
    decay = np.exp(-t * decay_factor)
    
    delta = np.concatenate([[0],decay])
    
    return delta
    
def line_noise(srate = 5000, freq = 50, numcycles = None):
    """
    Line noise artifact.
    
    Parameters:
    -----------
    srate = 5000 - sampling frequency\n
    freq = 50 (Default) - line noise frequency\n
    ncycles - number of cycles\n
    
    Returns:
    --------
    line_noise - numpy array\n
    """
    
    if numcycles is None:
        numcycles = np.random.randint(3,50)
    
    dur_samps = int((numcycles / freq) * srate)
    x = np.arange(dur_samps)
    y = np.sin(2 * np.pi * freq * x / srate)
    
    return y
    
def artifact_spike(srate = 5000, dur = None):
    """
    Artifact like spike (sharp, not gaussian)
    
    Parameters:
    -----------
    srate = 5000 - sampling frequency\n
    dur - duration of the event\n
    
    Returns:
    --------
    artifact_spike - numpy array\n
    """
    
    if dur is None:
        dur = round(np.random.random()/10,3)
    
    N = int(srate * dur)
    if not N % 2: # Check if the number is odd - we want to have proper spike
        N -= 1
    y = np.zeros(N)
    y[:int(N/2)+1] = np.linspace(0,1,int(N/2)+1)
    y[-int(N/2):] = np.linspace(1,0,int(N/2)+1)[1:]
      
    return y
    
# %% HFO    

def _wavelet(numcycles,f,srate):
    """
    Create a wavelet
    
    Parameters:
    ----------
    numcycles - number of cycles (gaussian window)\n
    f - central frequency\n
    srate - signal sample rate\n
        
    Returns:
    ----------
    wave - numpy array with waveform.\n
    time - numpy array with the time vector.\n
    """
    N = float(srate*numcycles)/(f) # number of points
    time = np.linspace((-numcycles/2)/float(f),(numcycles/2)/float(f),N) # time vector
    std = numcycles/(2*np.pi*f) # standard deviation
    wave = np.exp(2*1j*np.pi*f*time)*np.exp(-(time**2)/(2*(std**2))) # waveform
    return wave,time
    
def hfo(srate = 5000, f=None, numcycles = None):
    """
    Create an HFO.
    
    Parameters:
    ----------
    srate = 5000 (Defaults) - sampling rate\n
    f = None (Default) - Create a random HFO with central frequency between 60-600 Hz.\n
    numcycles = None (Default) - Create a random HFO with numcycles between 9 - 14.\n
    
    Returns:
    ----------
    wave - numpy array with waveform.\n
    time - numpy array with the time vector.\n
    """
    if numcycles is None:
        numcycles = np.random.randint(9,15)
    if f is None:
        f = np.random.randint(60,600)
    wave,time = _wavelet(numcycles,f,srate)
    return np.real(wave), time

# %% Spike
    
def spike(srate = 5000, dur = None):
    """
    Create a simple gausian spike.
    
    Parameters:
    -----------
    srate = 5000 (Default) - sampling rate.\n
    dur - spike duration (sec)\n
    
    Returns:
    --------
    spike - numpy array.
    """
    if dur is None:
        dur = round(np.random.random()*0.5,2)
    
    x = np.linspace(-4, 4, int(srate * dur)) # 4 stds
    spike_dist = norm.pdf(x, loc=0, scale=1)  # Create gaussian shape
    spike = spike_dist * 1/max(spike_dist) # Normalize so that the peak is at 1
    
    return spike

#def wavelet_spike(srate = 2000, f=None, numcycles = None):
#    '''
#    Create a wavelet spike. 
#    
#    Parameters:
#    ----------
#        f = None (Default) - Create a random Spike with central frequency between 60-600 Hz.
#        numcycles = None (Default) - Create a random Spike with numcycles between 1 - 2.
#        
#    Returns:
#    ----------
#        wave - numpy array with waveform.
#        time - numpy array with the time vector.
#    '''
#    if numcycles is None:
#        numcycles = np.random.randint(1,3)
#    if f is None:
#        f = np.random.randint(60,600)
#    wave,time = wavelet(numcycles,f,srate)
#    return -np.real(wave),time

# %% Combinations - just convenience functions
