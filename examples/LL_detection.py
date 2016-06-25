# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:18:10 2016

Script load file, detect and dump to pandas dataframe

@author: jan_cimbalnik
"""

import os

from pyhfo_detect.io import data_feeder, add_metadata
from pyhfo_detect.core import ll_detect

import matplotlib.pyplot as plt
import numpy as np

#Edf file
parent_dir = os.path.realpath('..')
file_path = parent_dir+'/test_data/ieeg_sample.edf'

# Get the data
data,fs = data_feeder(file_path, 0, 50000, "B'1")

# Presets - metadata - suggested 
met_dat = {'channel_name':"B'1", 'pat_id':'12'}


# %% We have data call the core of the algorithm and get detections
LL_df = ll_detect(data, fs, 80, 600, 1, 0.1, 0.25)

# %% Optional conversion to uUTC time or to absolute samples in the recording

# %% Adding metadata
LL_df = add_metadata(LL_df,met_dat)

# %% Optional rearange columns
LL_df = LL_df.loc[:,['pat_id','channel_name','event_start','event_stop']]

# %% Plot the detections in signal
plt.plot(data)
for row in LL_df.iterrows():
    det_size = row[1].event_stop-row[1].event_start
    plt.plot(np.linspace(row[1].event_start,row[1].event_stop,det_size,endpoint=False),
             data[row[1].event_start:row[1].event_stop],
             'r-')



# %% Optional insert into database

# %% Optional machine learning