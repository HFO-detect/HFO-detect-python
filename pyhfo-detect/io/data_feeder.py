# -*- coding: utf-8 -*-
# Copyright (c) 2016, HFO-detect Development Team.


from . import D_file_read
import pyedflib


def data_feeder(file_path, start_samp, stop_samp, channel_name):
    """
    Function that specifies file opener based on extension and returns data.

    Parameters:
    -----------
    file_path - path to data file\n
    start_samp - start sample\n
    stop_samp - stop sample\n
    channel_name - requested channel\n

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
        ch_idx = xheader['channel_name'].index(channel_name)
        data = D_file_read.read_d_header(sheader, [ch_idx],
                                         start_samp, stop_samp)

    elif ext in ['edf','bdf']:
        f = pyedflib.EdfReader(file_path)
        ch_idx = f.getSignalLabels().index(channel_name)
        fs = f.getSampleFrequency(ch_idx)
        data = f.readSignal(ch_idx)

    return data, fs
