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
    Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721–31.

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
    Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721–31.

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