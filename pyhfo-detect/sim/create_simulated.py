# -*- coding: utf-8 -*-
"""
Library of functions for simulated data creation

Created on 07/04/2016
@author: Anderson Brito da Silva
"""
import numpy as np
import scipy.signal as sig

def pinknoise(N):
    '''
    Create a pink noise (1/f) with N points.
    
    Parameters:
    ----------
        N - Number of samples to be returned
    '''
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
    X[range(nPts,N)] = np.real(X[range(N/2-1,0,-1)]) - 1j*np.imag(X[range(N/2-1,0,-1)])
  
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
    '''
    Create a brown noise (1/f²) with N points.
    
    Parameters:
    ----------
        N - Number of samples to be returned
    '''
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
    X[range(nPts,N)] = np.real(X[range(N/2-1,0,-1)]) - 1j*np.imag(X[range(N/2-1,0,-1)])
    
    y = np.fft.ifft(X) #IFFT
    
    y = np.real(y)
    # normalising    
    y -= np.mean(y)
    y /= np.sqrt(np.mean(y**2))
    # returning size of N
    if M % 2 == 1:
        y = y[:-1]
    return y
    
def wavelet(numcycles,f,srate):
    '''
    Create a wavelet
    
    Parameters:
    ----------
        numcycles - number of cycles (gaussian window)
        f - central frequency
        srate - signal sample rate
        
    Returns:
    ----------
        wave - numpy array with waveform.
        time - numpy array with the time vector.
    '''
    N = float(2*srate*numcycles)/(f) # number of points
    time = np.linspace(-numcycles/float(f),numcycles/float(f),N) # time vector
    std = numcycles/(2*np.pi*f) # standard desviation
    wave = np.exp(2*1j*np.pi*f*time)*np.exp(-(time**2)/(2*(std**2))) # waveform
    return wave,time
    
def hfo(srate = 2000, f=None, numcycles = None):
    '''
    Create a HFO.
    
    Parameters:
    ----------
        f = None (Default) - Create a random HFO with central frequency between 60-600 Hz.
        numcycles = None (Default) - Create a random HFO with numcycles between 9 - 14.
    
    Returns:
    ----------
        wave - numpy array with waveform.
        time - numpy array with the time vector.
    '''
    if numcycles is None:
        numcycles = np.random.randint(9,15)
    if f is None:
        f = np.random.randint(60,600)
    wave,time = wavelet(numcycles,f,srate)
    return np.real(wave), time

def wavelet_spike(srate = 2000, f=None, numcycles = None):
    '''
    Create a wavelet spike. 
    
    Parameters:
    ----------
        f = None (Default) - Create a random Spike with central frequency between 60-600 Hz.
        numcycles = None (Default) - Create a random Spike with numcycles between 1 - 2.
        
    Returns:
    ----------
        wave - numpy array with waveform.
        time - numpy array with the time vector.
    '''
    if numcycles is None:
        numcycles = np.random.randint(1,3)
    if f is None:
        f = np.random.randint(60,600)
    wave,time = wavelet(numcycles,f,srate)
    return -np.real(wave),time