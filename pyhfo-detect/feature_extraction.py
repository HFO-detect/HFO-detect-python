# -*- coding: utf-8 -*-
"""
Library of functions for feature extraction

Module for feature extraxtions. Usually short pieces of signal such as HFOs
themselves or windows of signal.

Created on 07/04/2016
@author: Anderson Brito da Silva
"""

import numpy as np

def extract_teager_energy(signal):
    '''
    Extract the Teager energy

    Parameters:
    ----------
        signal - numpy array
    '''
    sqr = np.power(signal[1:-1],2)
    odd = signal[:-2]
    even = signal[2:] # This triplicates the signal not sure about memory here.
    energy = sqr-odd*even
    energy = np.append(energy[0],energy)
    energy = np.append(energy,energy[-1])
    return energy

def extract_rms(signal, window_size = 6):
    '''
    Extract the Root Mean Square (RMS) energy

    Parameters:
    ----------
        signal - numpy array

    Returns:
    -------
        rms - float
    '''

    return np.sqrt(np.mean(np.power(signal,2)))


def extract_stenergy(signal):
    '''
    Extract Short Time energy -
    Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721–31.

    Parameters:
    ----------
        signal - numpy array

    Returns:
    -------
        stenergy - float
    '''

    return np.mean(np.power(signal,2))


def extract_line_lenght(signal):
    '''
    Extract Short time line leght -
    Gardner er al, 2006.  Clinical Neurophysiology: 118 (5): 1134–43.
    Note:
    ----
        There is a slight difference between extract LL and compute LL.

    Parameters:
    ----------
        signal - numpy array

    Returns:
    -------
        stenergy - float
    '''

    return np.sum(np.abs(np.diff(signal)))