#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:18:40 2016

Morphology detector provided by Sergey Burnos.
Reprogrammed to python by Jan Cimbalnik..

If you use this detector please cite!



@author: jan_cimbalnik
"""


from scipy.signal import butter, filtfilt, hilbert
import numpy as np

from ..signal_transform import *
from ..feature_extraction import *
from ..thresholds import *
from ..io.data_operations import *

# TODO - finish function help

# %% Morphology detector

def morphology_detect(data, fs, low_fc, high_fc,
                      bl_mu = .9, cdf_rms = .95, cdf_filt = .99, bs_dur = 30,
                      dur_th = .99, time_th = .02, max_noise_uv = 10.,
                      max_amp = 30., max_gap = .02, min_N_osc = 6):
    """
    Morphology detector by Sergey Burnos et al.\n
    Please cite!\n
    \n
    Burnos et al., 2016 Clinical Neurophysiology\n
    \n
    
    Parameters:
    -----------
    data(1-d numpy array) - raw data\n
    fs(int) - sampling frequency\n
    low_fc(float) - low cut-off frequency\n
    high_fc(float) - high cut-off frequency\n
    bl_mu(float) - level for maximum entropy\n
    cdf_rms(float) - percentile of detected baselines, incr in trheshold\n
    cdf_filt(float) - percentile of detected baselines, incr in trheshold\n
    bs_dur(float) - number of seconds for baseline detection\n
    dur_th(float) - 
    
    max_amp(float) - max event amplitude - artifact rejection\n
    max_gap(float) - interval for joining events
    min_N_osc(float) - minimum number of oscillations
    
    Returns:
    --------
    df_out(pandas.DataFrame) - output dataframe with detections\n
    """
    
    # Create output dataframe 
    
    df_out = create_output_df()
    
    # Filter the signal
    # FIXME - Sergey uses custom filter. Ask him about this.
    
    b, a = butter (3,[low_fc/(fs/2), high_fc/(fs/2)], 'bandpass')
    filt_data = filtfilt(b, a, data)
    
    # Define some additional parameters
    time_th = np.ceil(time_th * fs)
    smooth_window = 1 / high_fc  # RMS smoothing window
    bl_border = .02  # Ignore bordeers because of ST
    bl_mindist = 10*p.fs/1e3  # Min distance interval from baseline
    
    # Create signal envelope
    env = smooth(np.abs(hilbert(filt_data)),smooth_window * fs)
    
# %% Helper functions for morphology detect

#TODO - check that this functions corresponds with matlab smooth function.

def smooth(data,N):
    """
    Function to smooth the signal by running mean.
    
    Parameters:
    -----------
    data(array) - data to be smoothed.\n
    N(int) - sliding window size\n
    
    Returns:
    --------
    smoothed_data(array)\n
    """
    
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def baseline_threshold(data, filt_data, env,
                       bs_dur, bl_mu, bl_border, bl_mindist, maxNoiseuV,
                       fs, low_fc, high_fc):
    """
    distinguish background activity from spikes-HFO-artifacts
    according to Stockwell entrophy
    ref: wavelet entrophy: a new tool for analysis...
    Osvaldo a. Rosso et al, journal of neuroscience methods, 2000
    
    Parameters:
    -----------
    data(array) - original data\n
    filt_data(array) - filtered data\n
    env(array) - amplitude envelope\n
    
    Returns:
    --------
    
    """

    # Parameters
    indHighEntr=np.array([])
    S=np.zeros(fs)
    
    # Check duration
    
    if bs_dur > len(data) / fs:
        bs_dur = np.floor(len(data) / fs)
        
    for sec in range(len(np.floor(len(data) / fs))):
        
        signal = data[sec * fs : (sec + 1) * fs]
                      
        # S transform
        ST_data = stockwell_transform(signal, low_fc+1, high_fc, 1/fs, 1)
        stda = np.square(np.abs(ST_data))
        
        # Stockwell entropy
        std_total = np.sum(stda)  # Total energy
        std_rel = stda / std_total
        
        # Total entropy
        S = []
        for ifr in std_rel:
            S.append(-sum(ifr) * np.log(ifr))
        
        Smax=np.log(np.size(stda, 1)) # maximum entrophy = log(f_ST), /mu in mni,
        
        ### Threshold and baseline ###
        
        thr=bl_mu*Smax # threshold at mu*Smax, in mni BLmu=0.67
        indAboveThr=(S>thr) # find pt with high entrophy
        
        if indAboveThr:
            # FIXME - could we eliminate one if?
            # dont take border points because of stockwell transf
            indAboveThr=indAboveThr[~indAboveThr<(fs * bl_border)]
            indAboveThr=indAboveThr[~indAboveThr>(fs * (1-bl_border))]

            if indAboveThr:

                # check for the length
                indAboveThrN = indAboveThr[1:]
                indBrake = np.where(indAboveThrN - indAboveThrN[:-1] > 1)
    
                # check if it starts already above or the last point is abover the threshold
    
                if indAboveThr[0] == fs * bl_border: #??? This should not happen!!!
                    indBrake = indBrake[1:]
                if indAboveThr[-1] == fs * (1-bl_border): #??? Neither should this happen!!!
                    indBrake = indBrake[:-1]
                
                if np.size(indBrake) == 0:
                    indBrake = len(indAboveThr)
                    
            for iper in range(indBrake[:-1]):
                j = np.arange(indBrake[iper]+1,indBrake[iper+1])
                if len(j) >= bl_mindist:
                    indAboveThr[j] = indAboveThr[j] + (sec-1) *fs
                    if sum(abs(filt_data[indAboveThr[j]])) <= maxNoiseuV:  # % check that filtered signal is below max Noise level
                        indHighEntr = np.concatenate([indHighEntr,indAboveThr[j]])
                    
                    
    ### check one more time if the lentgh of baseline is too small ###
    if len(indHighEntr) < 2*fs:
        print('Baseline length < 2 sec, calculating for 5 min ')
        
        for sec in range(len(np.floor(len(data) / fs))):
        
            signal = data[sec * fs : (sec + 1) * fs]
                          
            # S transform
            ST_data = stockwell_transform(signal, low_fc+1, high_fc, 1/fs, 1)
            stda = np.square(np.abs(ST_data))
            
            # Stockwell entropy
            std_total = np.sum(stda)  # Total energy
            std_rel = stda / std_total
            
            # Total entropy
            S = []
            for ifr in std_rel:
                S.append(-sum(ifr) * np.log(ifr))
            
            Smax=np.log(np.size(stda, 1)) # maximum entrophy = log(f_ST), /mu in mni,
            
            ### Threshold and baseline ###
            
            thr=bl_mu*Smax # threshold at mu*Smax, in mni BLmu=0.67
            indAboveThr=(S>thr) # find pt with high entrophy
            
            if indAboveThr:
                # FIXME - could we eliminate one if?
                # dont take border points because of stockwell transf
                indAboveThr=indAboveThr[~indAboveThr<(fs * bl_border)]
                indAboveThr=indAboveThr[~indAboveThr>(fs * (1-bl_border))]
    
                if indAboveThr:
    
                    # check for the length
                    indAboveThrN = indAboveThr[1:]
                    indBrake = np.where(indAboveThrN - indAboveThrN[:-1] > 1)
        
                    # check if it starts already above or the last point is abover the threshold
        
                    if indAboveThr[0] == fs * bl_border: #??? This should not happen!!!
                        indBrake = indBrake[1:]
                    if indAboveThr[-1] == fs * (1-bl_border): #??? Neither should this happen!!!
                        indBrake = indBrake[:-1]
                    
                    if np.size(indBrake) == 0:
                        indBrake = len(indAboveThr)
                        
                for iper in range(indBrake[:-1]):
                    j = np.arange(indBrake[iper]+1,indBrake[iper+1])
                    if len(j) >= bl_mindist:
                        indAboveThr[j] = indAboveThr[j] + (sec-1) *fs
                        if sum(abs(filt_data[indAboveThr[j]])) <= maxNoiseuV:  # % check that filtered signal is below max Noise level
                            indHighEntr = np.concatenate([indHighEntr,indAboveThr[j]])
                            
    print('For '+str(npfloor(len(data)/fs))+' sec, baseline length '+str(len(indHighEntr)/fs)+' sec')