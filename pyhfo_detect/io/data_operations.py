# -*- coding: utf-8 -*-
# Copyright (c) 2016, HFO-detect Development Team.


from . import D_file_read
import pyedflib
import pandas as pd
import numpy as np


def data_feeder(file_path, start_samp, stop_samp, channel_name):
    """
    Function that specifies file opener based on extension and returns data.

    Parameters:
    -----------
    file_path(str) - path to data file\n
    start_samp(int) - start sample\n
    stop_samp(int) - stop sample\n
    channel_name(str) - requested channel\n

    Returns:
    --------
    data - requested data\n
    fs - sampling frequency\n
    """

    # Parse the file name to get extension
    ext = file_path[-file_path[::-1].find('.'):]

    # Decide which opener to use and get the data
    if ext == 'd':
        sheader, xheader = D_file_read.read_d_header(file_path)
        fs = sheader['fsamp']
        ch_idx = xheader['channel_names'].index(channel_name)
        data = D_file_read.read_d_data(sheader, [ch_idx],
                                       start_samp, stop_samp)

    elif ext in ['edf','bdf']:
        if ext == 'edf':
            f = pyedflib.EdfReader(file_path,3)
        elif ext == 'bdf':
            f = pyedflib.EdfReader(file_path,4)
        ch_idx = [x.decode() for x in f.getSignalLabels()].index(channel_name)
        fs = f.getSampleFrequency(ch_idx)
        data = f.readSignal(ch_idx)

    return data, fs


def create_output_df(fields=[],dtypes=None):
    """
    Function to create a custom pandas dataframe depending on the algorithm
    needs. Fields: event_start,event_stop preset.
    
    Parameters:
    -----------
    fields(list) - additional fields for the dataframe\n
    dtypes(dict) - dictionary with dtypes for specified fields (optional)\n
    
    Returns:
    --------
    dataframe - pandas dataframe for deteciton insertion\n
    """
    
    # Preset dtypes
    dtype_dict = {'event_start':np.int64, 'event_stop':np.int64} 
                  
    if dtypes is not None:
        dtype_dict.update(dtypes)
    
    out_df = pd.DataFrame(columns=['event_start', 'event_stop']+fields)
                                   
    for col in out_df.columns:
        if col in dtype_dict.keys():
            out_df[col] = out_df[col].astype(dtype_dict[col])
            
    return out_df
            
def add_metadata(df,metadata):
    """
    Convenience function to add metadata to the output dataframe.
    
    Parameters:
    -----------
    df(pandas.DataFrame) - dataframe with original data\n
    metadata(dict) - dictionary with column_name:value\n
    
    Returns:
    --------
    new_df(pandas.DataFrame) - updated dataframe
    """
    
    for key in metadata.keys():
        df[key] = metadata[key]
        
    return df
        