# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:18:10 2016

Example script to detect HFO with pyhfo-detect

Script load file, detect and dump to pandas dataframe

@author: jan_cimbalnik
"""

import os, requests, tempfile, pickle

from pyhfo_detect.io import add_metadata
from pyhfo_detect.core import (ll_detect, rms_detect, morphology_detect,
                               cs_detect_beta)

import matplotlib.pyplot as plt
import numpy as np

# %% Auxiliary
def download_file(url, local_file):
    r = requests.get(url, stream=True)
    with open(local_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return

# %% Get the data

# Download the data file
file_link = "https://raw.github.com/HFO-detect/HFO-detect-python/master/example_data/seeg.pkl"
local_file = tempfile.gettempdir()+'/hfo_detect_example.pkl'

download_file(file_link, local_file)

# Read the data
data_dict = pickle.load(open(local_file,'rb'))

data_arr = data_dict['data']
channels = data_dict['channels']
fsamps = data_dict['fsamp']

# Remove the file
os.remove(local_file)



# %% Presets - metadata - suggested 
met_dat = {'channel_name':"B'1", 'pat_id':'Nobody'}

data = data_arr[channels.index(met_dat['channel_name'])]
fs = fsamps[channels.index(met_dat['channel_name'])]

# %% We have data call the core of the algorithm and get detections
LL_df = ll_detect(data, fs, 80, 600, 1, 0.1, 0.25)
RMS_df = rms_detect(data, fs, 80, 600, 1, 0.1, 0.25)
Mor_df = morphology_detect(data, fs, 80, 600)
CS_df = cs_detect_beta(data, fs, 80, 600, 0.1)


# The dataframe now containes starts / stops of detections

# %% Optional conversion to uUTC time or to absolute samples in the recording

# %% Adding metadata
LL_df = add_metadata(LL_df,met_dat)
RMS_df = add_metadata(RMS_df,met_dat)
Mor_df = add_metadata(Mor_df,met_dat)
CS_df = add_metadata(CS_df,met_dat)


# %% Optional rearange columns
LL_df = LL_df.loc[:,['pat_id','channel_name','event_start','event_stop']]
RMS_df = RMS_df.loc[:,['pat_id','channel_name','event_start','event_stop']]
Mor_df = Mor_df.loc[:,['pat_id','channel_name','event_start','event_stop']]
CS_df = CS_df.loc[:,['pat_id','channel_name','event_start','event_stop']]


det_dfs = [LL_df, RMS_df, Mor_df, CS_df]

# %% Plot the detections in signal

f, axes_arr = plt.subplots(len(det_dfs), sharex = True)

for a,df in zip(axes_arr, det_dfs):
    a.plot(data)
    for row in df.iterrows():
        det_size = row[1].event_stop-row[1].event_start
        a.plot(np.linspace(row[1].event_start,row[1].event_stop,det_size,endpoint=False),
             data[row[1].event_start:row[1].event_stop],
             'r-')



# %% Optional insert into database

# %% Optional machine learning