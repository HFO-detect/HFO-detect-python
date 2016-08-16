#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:51:22 2016

Precision - recall test

Ing.,Mgr. (MSc.) Jan Cimbálník
Biomedical engineering
International Clinical Research Center
St. Anne's University Hospital in Brno
Czech Republic
&
Mayo systems electrophysiology lab
Mayo Clinic
200 1st St SW
Rochester, MN
United States
"""

from pyhfo_detect.evaluation import precision_recall
import pandas as pd

# Create artificial dataframes

gs_ss = [[1,5],[8,10],[15,24]]
dd_ss = [[2,6],[11,14],[23,28]]

# This should get 2x TP, 1 FN, 1 FP - precision 0.6, ercall 0.6

gs_df = pd.DataFrame(columns=('ev_start','ev_stop'),data=gs_ss)
dd_df = pd.DataFrame(columns=('ev_start','ev_stop'),data=dd_ss)

p,r = precision_recall.calculate_precision_recall(gs_df, dd_df, ['ev_start','ev_stop'])