# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:41:30 2016

Module with algorithm cores. Detection output goes to pandas dataframe.
Obilgatory fields of the dataframe are:
event_start
event_stop
pat_id
channel_name
low_fc
high_fc

@author: Jan Cimbalnik
"""

from scipy.signal import butter, filtfilt
import numpy as np

from .signal_transform import *
from .feature_extraction import *
from .thresholds import *
from .io.data_operations import *


# %% Line-length detector

def line_length_detect(data, fs, low_fc, high_fc, threshold, window_size,
                       window_overlap):
    """
    Line-length detection algorithm. (CITATIONS)
    
    Parameters:
    -----------
    data(1-d numpy array) - raw data\n
    fs(int) - sampling frequency\n
    low_fc(float) - low cut-off frequency\n
    high_fc(float) - high cut-off frequency\n
    window_size(float) - sliding window size in secs\n
    window_overlap(float) - fraction of the window overlap (0 to 1)\n
    
    Returns:
    --------
    df_out(pandas.DataFrame) - output dataframe with detections\n
    """    
    
    # Calculate window values for easier operation
    samp_win_size = int(window_size*fs) # Window size in samples
    samp_win_inc = int(samp_win_size*window_overlap) # Window increment in samples
    
    # Create output dataframe 
    
    df_out = create_output_df()
    
    # Filter the signal
    
    b, a = butter (3,[low_fc/(fs/2), high_fc/(fs/2)], 'bandpass')
    filt_data = filtfilt(b, a, data)
    
    # Transform the signal - one sample window shift
    
    #LL = compute_line_lenght(filt_data, window_size*fs)
    
    # Alternative approach - overlapping window
    
    win_start = 0
    win_stop = window_size*fs
    LL = []
    while win_start < len(filt_data):
        if win_stop > len(filt_data):
            win_stop = len(filt_data)
            
        LL.append(extract_line_lenght(filt_data[win_start:win_stop]))
        
        win_start += samp_win_inc
        win_stop += samp_win_inc
    
    # Create threshold    
    det_th = th_std(LL,threshold)
    
    # Detect
    LL_idx=0
    df_idx=0
    while LL_idx < len(LL):
        if LL[LL_idx] >= det_th:
            event_start = LL_idx * samp_win_inc
            while LL[LL_idx] >= det_th:
                LL_idx += 1
            event_stop = (LL_idx * samp_win_inc)+samp_win_size

            # Optional feature calculations can go here

            # Write into dataframe
            df_out.loc[df_idx] = [event_start, event_stop]
            df_idx += 1

            LL_idx+=1
        else:
            LL_idx+=1
    
    return df_out