#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:43:32 2016

Module for feature precision evaluation.

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

from .util import match_detections

from scipy.stats import ttest_1samp

def eval_feature_differences(diff_df, bn):
    """
    Function to evaluate feature differences between known values an estimated\n
    values.
    
    Parameters:
    -----------
    diff_df - dataframe produced by get_feature_differences
    bn - names of event start stop [start_name, stop_name] (list)\n
    
    Returns:
    --------
    stat_dict - dictionary with p values of 1 sample ttests for each feature
    """

    # Get the feature diffs
    feature_diff_keys = diff_df.columns.difference(bn)

    # Run statistical test on
    stat_dict = {}
    for f_key in feature_diff_keys:
        res = ttest_1samp(diff_df.loc[:, f_key].values, 0)[1]
        stat_dict[f_key] = res

    return stat_dict
    
def get_feature_differences(gs_df, dd_df, bn, feature_names):
    """
    Function to get feature differences between known and estimated values.
    
    Parameters:
    -----------
    gs_df - dataframe of events with known features (pandas Dataframe)\n
    dd_df - dataframe of events with estimated efatures (pandas DataFrame)\n
    bn - names of event start stop [start_name, stop_name] (list)\n
    feature_names - dictionary with features as keys and column names as values\n
    
    Returns:
    --------
    match_df - dataframe with indexes of matched detections\n
    N_missed - number of missed artificial detections\n
    """
    
    # Match the detections first
    if 'frequency' in feature_names.keys():
        match_df = match_detections(gs_df, dd_df, bn, feature_names['frequency'])
    else:
        match_df = match_detections(gs_df, dd_df, bn)
        
    # Get count of missed detections and pop them from the df
    N_missed = len(match_df.loc[match_df.dd_index.isnull()])
    match_df = match_df.loc[~match_df.dd_index.isnull()]
        
    # Run through matched detections and compare the features
    for feature in feature_names.keys():
        for match_row in match_df.iterrows():
            gs_feat = gs_df.loc[match_row[1].gs_index, feature_names[feature]]
            dd_feat = dd_df.loc[match_row[1].dd_index, feature_names[feature]]
            match_df.loc[match_row[0],feature+'_diff'] = gs_feat - dd_feat
            
    return match_df, N_missed
    