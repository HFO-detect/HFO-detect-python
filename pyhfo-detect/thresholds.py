# -*- coding: utf-8 -*-
"""
Library of functions for feature extraction

Created on 07/04/2016
@author: Anderson Brito da Silva
"""

def std(signal,ths):
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

def tukey(signal,ths):
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

def percentile(signal,ths):
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

def quian(signal,ths):
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
    
