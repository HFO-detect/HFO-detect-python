# -*- coding: utf-8 -*-
"""
Module for threshold calculation.

Created on 07/04/2016
@author: Anderson Brito da Silva
"""

import numpy as np

def th_std(signal,ths):
    '''
    Calcule threshold by Standar Desviations above the mean

    Parameters:
    ----------
        signal - numpy array
        ths - number of SD above the mean

    Returns:
    ----------
        ths_value - value of the threshold
    '''
    ths_value = np.mean(signal) + ths*np.std(signal)
    return ths_value

def th_tukey(signal,ths):
    '''
    Calcule threshold by Tukey method.

    Parameters:
    ----------
        signal - numpy array
        ths - number of interquartile interval above the 75th percentile

    Returns:
    ----------
        ths_value - value of the threshold
    '''
    ths_value = np.percentile(signal,75) + ths*(np.percentile(signal,75)-np.percentile(signal,25))
    return ths_value

def th_percentile(signal,ths):
    '''
    Calcule threshold by Percentile

    Parameters:
    ----------
        signal - numpy array
        ths - percentile

    Returns:
    ----------
        ths_value - value of the threshold
    '''
    ths_value = np.percentile(signal,ths)
    return ths_value

def th_quian(signal,ths):
    '''
    Calcule threshold by Quian
    Quian Quiroga, R. 2004. Neural Computation 16: 1661–87.

    Parameters:
    ----------
        signal - numpy array
        ths - number of estimated noise SD above the mean

    Returns:
    ----------
        ths_value - value of the threshold
    '''
    ths_value = ths * np.median(np.abs(signal)) / 0.6745
    return ths_value

