# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:18:10 2016

Script load file, detect and dump to pandas dataframe

@author: jan_cimbalnik
"""

from pyhfo_detect.io import data_feeder, add_metadata
from pyhfo_detect import line_length_detect
import matplotlib.pyplot as plt
import numpy as np

file_path = '/home/jan_cimbalnik/Dropbox/Easrec-1404171032.d'

# Get the data
data,fs = data_feeder(file_path, 0, 50000, "O'1")
data = data[:,0]

# Presets - metadata - suggested 
met_dat = {'channel_name':'O1', 'pat_id':'12'}


# %% We have data call the core of the algorithm and get detections
LL_df = line_length_detect(data, fs, 80, 600, 1, 0.1, 0.25)

# %% Optional conversion to uUTC time

# %% We have the dataframe, lets add metadata
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


# %% Insert into database

# %% Machine learning