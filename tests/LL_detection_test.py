# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:18:10 2016

Script load file, detect and dump to pandas dataframe

@author: jan_cimbalnik
"""

from pyhfo_detect.io import data_feeder

file_path = '/home/jan_cimbalnik/Dropbox/Easrec-1404171032.d'


data = data_feeder(file_path, 0, 5000, "O'1")