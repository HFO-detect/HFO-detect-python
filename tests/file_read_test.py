# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:12:16 2016

Jan Cimbalnik
"""

import pyedflib,os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname+'/../test_data/')

file_name='./test.edf'
file_name = '/home/jan_cimbalnik/Dropbox/HFO_detectors_project/HFO-detect-python/test_data/test.edf'
f = pyedflib.EdfReader(file_name)
ch_idx = f.getSignalLabels().index(b'DC01')
sig = f.readSignal(0)

