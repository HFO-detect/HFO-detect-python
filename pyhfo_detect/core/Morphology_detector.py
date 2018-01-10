#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:18:40 2016

Morphology detector provided by Sergey Burnos.
Reprogrammed to python by Jan Cimbalnik..

If you use this detector please cite!



@author: jan_cimbalnik
"""
import pickle,os

from scipy.signal import filtfilt, hilbert
import numpy as np

from ..signal_transform import compute_stockwell_transform
from ..io.data_operations import create_output_df, correct_boundary_dtype

# %% Presets - Sergey's custom filter
mod_dir = os.path.split(__file__)[0]
filter_coefs = pickle.load(open(mod_dir+'/Morphology_detector.pkl','rb'))

# %% Morphology detector

def morphology_detect(data, fs, low_fc, high_fc,
                      bl_mu = .9, cdf_rms = .95, cdf_filt = .99, bs_dur = 30,
                      dur_th = .99, time_th = .02, max_noise_uv = 10.,
                      max_amp = 30., max_gap = .02, min_N_osc = 6):
    """
    Morphology detector by Sergey Burnos et al.\n
    Please cite!\n
    \n
    [1] BURNOS, Sergey, Birgit FRAUSCHER, Rina ZELMANN, Claire HAEGELEN, \n
    Johannes SARNTHEIN and Jean GOTMAN. The morphology of high frequency \n
    oscillations (HFO) does not improve delineating the epileptogenic zone. \n
    Clinical Neurophysiology [online]. 2016, pp. 1–9. ISSN 13882457. \n
    Available at: doi:10.1016/j.clinph.2016.01.002
    \n
    
    Parameters:
    -----------
    data(1-d numpy array) - raw data\n
    fs(int) - sampling frequency\n
    low_fc(float) - low cut-off frequency, allows only 80, 250\n
    high_fc(float) - high cut-off frequency, allows onlt 250, 500\n
    bl_mu(float) - level for maximum entropy\n
    cdf_rms(float) - percentile of detected baselines, incr in trheshold\n
    cdf_filt(float) - percentile of detected baselines, incr in trheshold\n
    bs_dur(float) - number of seconds for baseline detection\n
    dur_th(float) - duraiton threshold
    
    max_amp(float) - max event amplitude - artifact rejection (default 30 uV)\n
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
    bl_mindist = 10*fs/1e3  # Min distance interval from baseline  
    
    
    # 1) Filtering
    if low_fc == 80:
        b = filter_coefs['Rb']
        a = filter_coefs['Ra']
    elif low_fc == 250:
        b = filter_coefs['FRb']
        a = filter_coefs['FRa']
    filt_data = filtfilt(b, a, data)
    
    # 2) Envelope 
    smooth_N = int(smooth_window * fs)
    env = np.abs(hilbert(filt_data))
    env = np.convolve(env, np.ones((smooth_N,))/smooth_N, mode='same')
    
    # 3) threshold
    thr, thr_filt, indHighEntr = baseline_threshold(data, filt_data, env,
                                                   bs_dur, bl_mu, bl_border,
                                                   bl_mindist, max_noise_uv,
                                                   cdf_rms, cdf_filt,
                                                   fs, low_fc, high_fc)
    
                                  
    # Display warning if baseline is too short
    if len(indHighEntr) < 2*fs:
        print('!!!!Short baseline!!!!')
        
    # 4) Stage 1 - detection of EoIs
        
    env[0] = env[-1] = 0
    pred_env = np.zeros(len(env))
    pred_env[1:] = env[:-1]
    pred_env[0] = pred_env[1]


    t1 = np.where((pred_env < (thr * dur_th)) & (env >= (thr * dur_th)))[0]
    t2 = np.where((pred_env > (thr * dur_th)) & (env <= (thr * dur_th)))[0]
    
    trig = np.where((pred_env < thr) & (env >= thr))[0]
    trig_end = np.where((pred_env >= thr) & (env < thr))[0]

    det_cnt = 0

    for trigs in zip(trig, trig_end):
        if trigs[1] - trigs[0] >= time_th:
            
            k = np.where((t1 <= trigs[0]) & (t2 >= trigs[0]))[0][0]
            
            if t1[k] > 0:
                det_start = t1[k]
            else:
                det_start = 0
                
            if t2[k] < len(env):
                det_stop = t2[k]
            else:
                det_stop = len(env)-1
                
            peak_amp = np.max(env[t1[k]:t2[k]])
            peak_ind = np.argmax(env[t1[k]:t2[k]]) + t1[k] + 1
            
            if peak_amp > max_amp:
                continue
            
            df_out.loc[det_cnt] = [det_start, det_stop, peak_ind, peak_amp]
            det_cnt += 1
            
    if det_cnt:
        
        # 5) check sufficient number of oscillations
        df_out = check_oscillations(df_out, filt_data, thr_filt, min_N_osc)
        
        # 6) Merge detections
        df_out = join_detections(df_out, max_gap, fs) #FIXME - this should be general function
        
    correct_boundary_dtype(df_out)
    return df_out
    
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
        interval_eeg = data[int(row[1].event_start):int(row[1].event_stop)+1] - to_detrend
        
        # Compute absolute values for oscillations interval
        abs_eeg = np.abs(interval_eeg)
        
        # Look for zeros
        zero_vec = np.where(interval_eeg[:-1] * interval_eeg[1:] < 0)[0]
        N_zeros = len(zero_vec)
        
        N_max_counter = np.zeros(N_zeros-1)
        
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
        if not np.any((np.diff(np.where(N_max_counter==0)[0]) > min_N_osc)):
            rejected_idcs.append(row[0])
            
    # Get rid of detections that did not have enough oscillations
    df = df.loc[~df.index.isin(rejected_idcs)]
    df.reset_index(drop=True, inplace=True) 
        
    return df

def join_detections(df, max_gap, fs):
    """
    Function to join detections.
    """
    
    N_det_c = 0    
    max_gap = max_gap * fs
    
    joined_df = df[0:1]
    joined_df.reset_index(drop=True, inplace=True) 
    
    for row in df[1:].iterrows():
        # Join detection
        if row[1].event_start > joined_df.loc[N_det_c,'event_stop']:
            n_diff = row[1].event_start - joined_df.loc[N_det_c,'event_stop']
            
            if n_diff < max_gap:
                joined_df.loc[N_det_c,'event_stop'] = row[1].event_stop
                
                if joined_df.loc[N_det_c,'peak_amp'] < row[1].peak_amp:
                    joined_df.loc[N_det_c,'peak_amp'] = row[1].peak_amp
                    joined_df.loc[N_det_c,'peak'] = row[1].peak
                
            else:
                N_det_c += 1
                joined_df.loc[N_det_c,:] = df.loc[row[0],:]    
            
    return joined_df
            
            
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def baseline_threshold(data, filt_data, env,
                       bs_dur, bl_mu, bl_border, bl_mindist, max_noise_uv,
                       cdf_level_rms, cdf_level_filt,
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

    indHighEntr=np.array([],dtype=np.int64)
    S=np.zeros(int(fs))
    
    # Check duration
    
    if bs_dur > len(data) / fs:
        bs_dur = np.floor(len(data) / fs)
        
    for sec in range(bs_dur):
        
        signal = data[int(sec * fs) : int((sec + 1) * fs)]
                      
        # S transform
        ST_data = compute_stockwell_transform(signal, fs, low_fc+1, high_fc, 1)[0]
        stda = np.square(np.abs(ST_data))
        
        # Stockwell entropy
        std_total = np.sum(stda,0)  # Total energy
        std_rel = stda / std_total
        
        # Total entropy
        S = []
        for ifr in range(np.size(std_rel,1)):
            S.append(-sum(std_rel[:,ifr] * np.log(std_rel[:,ifr])))
        
        Smax=np.log(np.size(stda, 0)) # maximum entrophy = log(f_ST), /mu in mni,
        
        ### Threshold and baseline ###
        
        thr=bl_mu*Smax # threshold at mu*Smax, in mni BLmu=0.67
        indAboveThr=np.where(S>thr)[0] # find pt with high entrophy
        indAboveThr += 1 # Had to do this because matlab is stupid!
        
        if len(indAboveThr):
            # dont take border points because of stockwell transf
            indAboveThr=indAboveThr[~(indAboveThr<(fs * bl_border))]
            indAboveThr=indAboveThr[~(indAboveThr>(fs * (1-bl_border)))]
    
            if len(indAboveThr):
    
                # check for the length
                indAboveThrN = indAboveThr[1:]
                indBrake = np.where(indAboveThrN - indAboveThr[:-1] > 1)[0]
    
                # check if it starts already above or the last point is above the threshold
    
                if indAboveThr[0] == fs * bl_border:
                    indBrake = np.concatenate(([0],indBrake))
                if indAboveThr[-1] == fs * (1-bl_border):
                    indBrake = np.concatenate((indBrake,[len(indAboveThr)-1]))
                
                if np.size(indBrake) == 0:
                    indBrake = len(indAboveThr)
                    
            for iper in range(len(indBrake[:-1])):
                j = np.arange(indBrake[iper]+1,indBrake[iper+1]+1)
                if len(j) >= bl_mindist:
                    indAboveThr[j] = indAboveThr[j] + ((sec*fs)-1)
                    if not sum(abs(filt_data[indAboveThr[j]]) > max_noise_uv):  # % check that filtered signal is below max Noise level
                        indHighEntr = np.concatenate([indHighEntr,indAboveThr[j]])
                          
    print('For '+str(bs_dur)+' sec, baseline length '+str(len(indHighEntr)/fs)+' sec')
                        
    ### check one more time if the lentgh of baseline is too small ###
    if len(indHighEntr) < 2*fs:
        print('Baseline length < 2 sec, calculating for 5 min ')
        
        for sec in range(bs_dur,int(np.floor(len(data) / fs))):
        
            signal = data[int(sec * fs) : int((sec + 1) * fs)]
                      
            # S transform
            ST_data = compute_stockwell_transform(signal, fs, low_fc+1, high_fc, 1)[0]
            stda = np.square(np.abs(ST_data))
            
            # Stockwell entropy
            std_total = np.sum(stda,0)  # Total energy
            std_rel = stda / std_total
            
            # Total entropy
            S = []
            for ifr in range(np.size(std_rel,1)):
                S.append(-sum(std_rel[:,ifr] * np.log(std_rel[:,ifr])))
            
            Smax=np.log(np.size(stda, 0)) # maximum entrophy = log(f_ST), /mu in mni,
            
            ### Threshold and baseline ###
            
            thr=bl_mu*Smax # threshold at mu*Smax, in mni BLmu=0.67
            indAboveThr=np.where(S>thr)[0] # find pt with high entrophy
            indAboveThr += 1 # Had to do this because matlab is stupid!
            
            if len(indAboveThr):
                # dont take border points because of stockwell transf
                indAboveThr=indAboveThr[~(indAboveThr<(fs * bl_border))]
                indAboveThr=indAboveThr[~(indAboveThr>(fs * (1-bl_border)))]
        
                if len(indAboveThr):
        
                    # check for the length
                    indAboveThrN = indAboveThr[1:]
                    indBrake = np.where(indAboveThrN - indAboveThr[:-1] > 1)[0]
        
                    # check if it starts already above or the last point is above the threshold
        
                    if indAboveThr[0] == fs * bl_border:
                        indBrake = np.concatenate(([0],indBrake))
                    if indAboveThr[-1] == fs * (1-bl_border):
                        indBrake = np.concatenate((indBrake,[len(indAboveThr)-1]))
                    
                    if np.size(indBrake) == 0:
                        indBrake = len(indAboveThr)
                        
                for iper in range(len(indBrake[:-1])):
                    j = np.arange(indBrake[iper]+1,indBrake[iper+1]+1)
                    if len(j) >= bl_mindist:
                        indAboveThr[j] = indAboveThr[j] + ((sec*fs)-1)
                        if not sum(abs(filt_data[indAboveThr[j]]) > max_noise_uv):  # % check that filtered signal is below max Noise level
                            indHighEntr = np.concatenate([indHighEntr,indAboveThr[j]])
                                                    
        print('For '+str(np.floor(len(data)/fs))+' sec, baseline length '+str(len(indHighEntr)/fs)+' sec')
    
    if len(indHighEntr):
        xs, ys = ecdf(env[indHighEntr])
        thr_cdf = xs[[ys>cdf_level_rms]][0]
    else:
        thr_cdf = 1000
        
    thr = thr_cdf
    
    if len(indHighEntr):
        xs, ys = ecdf(filt_data[indHighEntr])
        thr_cdf = xs[[ys>cdf_level_filt]][0]
    else:
        thr_cdf = 1000
        
    thr_filt = thr_cdf
        
    # show thresholds
    print('ThrEnv = '+str(thr)+', ThrFiltSig = '+str(thr_filt))
    
    return thr, thr_filt, indHighEntr