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

def morphology_detect(data, fs, low_fc, high_fc, mark,
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
    dur_th(float) - duraiton threshold
    
    max_amp(float) - max event amplitude - artifact rejection\n
    max_gap(float) - interval for joining events
    min_N_osc(float) - minimum number of oscillations
    
    Returns:
    --------
    df_out(pandas.DataFrame) - output dataframe with detections\n
    """
    
    # Create output dataframe 
    
    df_out = create_output_df(fields=['peak','peak_amp'])
    
    # Define some additional parameters -could go into parameters
    time_th = np.ceil(time_th * fs)
    smooth_window = 1 / high_fc  # RMS smoothing window
    bl_border = .02  # Ignore bordeers because of ST
    bl_mindist = 10*p.fs/1e3  # Min distance interval from baseline  
    
    if  mark == 'Ripple':
        max_amp_filt = 30
    elif mark == 'FastRipple':
        max_amp_filt = 20
    
    
    # 1) Filtering
    # FIXME - Sergey uses custom filter. Ask him about this.
    
    b, a = butter (3,[low_fc/(fs/2), high_fc/(fs/2)], 'bandpass')
    filt_data = filtfilt(b, a, data)
    
    # 2) Envelope 
    env = smooth(np.abs(hilbert(filt_data)),smooth_window * fs)
    
    # 3) threshold
    thr, thr_filt, indHighEntr = baseline_threshold(data, filt_data, env,
                                                   bs_dur, bl_mu, bl_border,
                                                   bl_mindist, max_noise_uv,
                                                   fs, low_fc, high_fc)
                                  
    # Display warning if baseline is too short
    if len(indHighEntr) < 2*fs:
        print('!!!!Short baseline!!!!')
        
    # 4) Stage 1 - detection of EoIs
        
    env[0] = env[-1] = 0
    pred_env = np.zeros(len(env))
    pred_env[1:] = env[:-1]
    pred_env[0] = pred_env[1]


    t1 = np.where(pred_env < (thr * dur_th) & env >= (thr * dur_thr))
    t2 = np.where(pred_env > (thr * dur_th) & env <= (thr * dur_thr))

    trig = np.where(pred_env < thr & env >= thr)
    trig_end = np.where(pred_enc >= thr & env < thr)

    det_cnt = 0
    
    for trigs in zip(trig, trig_end):
        if trigs[1] - trigs[0] >= time_th:
            
            k = np.where(t1 <= trigs[0] & t2 >= trigs[0])
            
            if t1[k] > 0:
                det_start = t1[k]
            else:
                det_start = 0
                
            if t2[k] <= len(env):
                det_stop = t2[k]
            else:
                det_stop = len(env)
                
            peak_amp = np.max(env[t1[k]:t2[k]])
            peak_ind = np.argmax(env[t1[k]:t2[k]])
            
            if peak_amp > max_amp_filt:
                continue
            
            df_out.loc[det_cnt] = [det_start, det_stop, peak_ind, peak_amp]
            det_cnt += 1
            
    if det_cnt:
        df_out = check_oscillations(df_out, filt_data, thr_filt, min_N_osc) #FIXME - should this be a general function?
        df_out = join_detections(df_out) #FIXME - this should be general function
    
# %% Helper functions for morphology detect

#TODO - check that this functions corresponds with matlab smooth function.

def check_oscillations(df, data, thr_filt, min_N_osc):
    """
    Function to reject oscillations that have less than 8 peaks.
    """
    
    rejected_idcs = []
    for row in df.iterrows():
        # Detrend data
        to_detrend = 0
        
        # Get eeg interval
        interval_eeg = data[row[1].event_start:row[1].event_stop] - to_detrend
        
        # Compute absolute values for oscillations interval
        abs_eeg = np.abs(interval_eeg)
        
        # Look for zeros
        zero_vec = np.argmax(interval_eeg[:-1] * interval_eeg[1:] < 0)
        N_zeros = len(zero_vec)
        
        N_max_counter = np.zeros(N_zeros)
        
        if N_zeros:
            for zi, z in enumerate(zero_vec[:-1]):
                cross_start = z
                cross_stop = zero_vec[zi+1]
                d_max = np.max(abs_eeg[cross_start:cross_stop])
                
                if d_max > thr_filt:
                    N_max_counter[zi] = 1
                else:
                    N_max_counter[zi] = 0
                    
        N_max_counter = np.concatenate([[0],N_max_counter,[0]])
        
        # NOTE: inversed logic from the original code - is more readable this way
        if not True in (np.diff(np.where(N_max_counter==0)) > min_N_osc):
            rejected_idcs.append(row[0])
            
    # Get rid of detections that did not have enough oscillations
    df = df.loc[~df.index.isin(rejected_idcs)]
        
    return df

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

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
                       cdf_level_filt,
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
    
    baseline = env[indHighEntr]
    
    if len(indHighEntr):
        xs, ys = ecdf(filt_data(indHighEntr))
        thr_cdf = xs[ys>cdf_level_filt]
    else:
        thr_cdf = 1000
        
    # show thresholds
    print('ThrEnv = '+str(thr)+', ThrFiltSig = '+str(thr_cdf))
    
    return thr, thr_cdf, indHighEntr