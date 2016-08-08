#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:10:26 2016

General functions used by evaluation

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

import pandas as pd

def match_detections(gs_df, dd_df, bound_names, freq_name = None):
    """
    Matches gold standard detections with detector detections.
    
    Parameters:
    -----------
    gs_df - gold standard detections (pandas DataFrame)\n
    dd_df - detector detections (pandas DataFrame)\n
    bound_names - names of event start stop [start_name, stop_name] (list)\n
    freq_name - name of frequency column (str)\n
    
    Returns:
    --------
    match_df - dataframe with matched indeces (pandas DataFrame)\n
    """

    match_df = pd.DataFrame(columns = ('gs_index', 'dd_index'))
    match_df_idx = 0
    for row_gs in gs_df.iterrows():
        matched_idcs = []
        gs = [row_gs[1][bound_names[0]],row_gs[1][bound_names[1]]]
        for row_dd in dd_df.iterrows():  # We can create a subset here?! - would speed things up
            dd = [row_dd[1][bound_names[0]],row_dd[1][bound_names[1]]]
            if detection_overlap_check(gs, dd):
                matched_idcs.append(row_dd[0])
                
        if len(matched_idcs) == 0:
            match_df.loc[match_df_idx] = [row_gs[0], None]                
        elif len(matched_idcs) == 1:
            match_df.loc[match_df_idx] = [row_gs[0], row_dd[0]]
        else:  
            if freq_name:  # In rare event of multiple overlaps - get the closest in frequency domain
                dd_idx = (abs(dd_df.loc[matched_idcs,freq_name]-row_gs[freq_name])).idxmin()
                match_df.loc[match_df_idx] = [row_gs[0], dd_idx]
            else:  # Get the detection with closest event start - less precision than frequency
                dd_idx = (abs(dd_df.loc[matched_idcs,bound_names[0]]-row_gs[bound_names[0]])).idxmin()
                match_df.loc[match_df_idx] = [row_gs[0], dd_idx]

        match_df_idx += 1
        
    return match_df
    
def detection_overlap_check(gs,dd):
    """
    Evaluates if two detections overlap

    Paramters:
    ----------
    gs - gold standard detection [start,stop] (list)\n
    dd - detector detection [start,stop] (list)\n

    Returns:
    --------
    overlap - boolean\n

    """

    overlap = False

    if (dd[1] >= gs[0]) and (dd[1] <= gs[1]): # dd stop in gs
        overlap = True

    if (dd[0] >= gs[0]) and (dd[0] <= gs[1]): # dd start in gs
        overlap = True

    if (dd[0] <= gs[0]) and (dd[1] >= gs[1]): # gs inside dd
        overlap = True

    return overlap