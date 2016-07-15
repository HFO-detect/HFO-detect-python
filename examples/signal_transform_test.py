#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:16:36 2016

@author: bmestudent
"""

import pyhfo_detect

import os

from pyhfo_detect.io import data_feeder

from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt

#Edf file
parent_dir = os.path.realpath('..')
#file_path = parent_dir+'/test_data/ieeg_sample.edf'
file_path = parent_dir + '/test_data/Easrec_ic_exp-036_150429-1346.d'
file_path = '/mnt/BME_shared/raw_data/SEEG/seeg-032-141008/Easrec_ic_exp-032_141008-0929.d'

# Get the data
data,fs = data_feeder(file_path, 0, 50000, "B'1")

# %% Data
low_fc = 80
high_fc = 600
treshold = 1
window_size = 0.1
window_overlap = 0.25

# Calculate window values for easier operation
samp_win_size = int(window_size*fs) # Window size in samples
samp_win_inc = int(samp_win_size*window_overlap) # Window increment in samples
   
# Filter the signal
    
b, a = butter (3,[low_fc/(fs/2), high_fc/(fs/2)], 'bandpass')
filt_data = filtfilt(b, a, data)
    
# Transform the signal - one sample window shift
    
Line_length = pyhfo_detect.signal_transform.compute_line_lenght(filt_data, window_size*fs)
Hilbert_energy = pyhfo_detect.signal_transform.compute_hilbert_energy(filt_data)
Hilbert_envelope = pyhfo_detect.signal_transform.compute_hilbert_envelope(filt_data)
Stenergy = pyhfo_detect.signal_transform.compute_stenergy(filt_data, window_size * fs)
Teager_energy = pyhfo_detect.signal_transform.compute_teager_energy(filt_data)
Rms = pyhfo_detect.signal_transform.compute_rms(filt_data, window_size*fs)

# Plot
plt.figure(1)
plt.subplot(311)
plt.plot(Line_length)
plt.title('Line length')
plt.subplot(312)
plt.plot(Stenergy)
plt.title('Stenergy')
plt.subplot(313)
plt.plot(Rms)
plt.title('Rms')
 
plt.figure(2)
plt.subplot(311)
plt.plot(Hilbert_energy)
plt.title('Hilbert energy')
plt.subplot(312)
plt.plot(Hilbert_envelope)
plt.title('Hilbert envelope')
plt.subplot(313)
plt.plot(Teager_energy)
plt.title('Teager energy')
 
plt.figure(3)
plt.subplot(211)
plt.plot(data)
plt.title('Original data')
plt.subplot(212)
plt.plot(filt_data)
plt.title('Filtred data') 