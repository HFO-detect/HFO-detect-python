#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:02:16 2016

Calculation of precision, recall (sensitivity).

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

from .general import detection_overlap_check

"""
NOTE: we could use scikit-learn for this but that would require additional
modules to be installed. Something to consider for the future. Would simplify
things a bit. If we incorporate clustering and machine learning we should
switch to this.
"""

def create_precision_recall_curve(gs_df, dd_df, bn, threshold):
    """
    Function to create precision recall curve.
    
    Parameters:
    -----------
    gs_df - gold standard detections\n
    dd_df - automatically detected detections\n
    bn - names of event start stop [start_name, stop_name] (list)\n
    threshold - name of the threshold field for 
    
    Returns:
    --------
    precision - list of precision points\n
    recall - list of recall points\n
    """
    
    # Initiate lists
    precision = []
    recall = []

    # Thresholds
    ths = list(dd_df[threshold].unique())
    ths.sort()

    # Run through thresholds
    for th in ths:
        p, r = calculate_precision_recall(gs_df,
                                          dd_df[dd_df == threshold],
                                          bn)
        precision.append(p)
        recall.append(r)
        
    return precision, recall
    

def calculate_precision_recall(gs_df, dd_df, bn):
    """
    Function to calculate precision and recall values.
    
    Parameters:
    -----------
    gs_df - gold standard detections\n
    dd_df - automatically detected detections\n
    bn - names of event start stop [start_name, stop_name] (list)\n
    
    Returns:
    --------
    precision - precision of the detection set\n
    recall - recall(sensitivity) of the detection set\n
    """
    
    # Create column for matching
    dd_df['match'] = False
    gs_df['match'] = False
    
    # Initiate true positive
    TP = 0
    
    # Start running through gold standards
    for gs_row in gs_df.iterrows():
        gs_det = [gs_row[1][bn[0]], gs_row[1][bn[1]]]
        
        det_flag = False
        for dd_row in dd_df.iterrows():
            dd_det = [dd_row[1][bn[0]], dd_row[1][bn[1]]]

            if detection_overlap_check(gs_det, dd_det):
                det_flag = True
                break
            
        # Mark the detections
        if det_flag:
            TP += 1
            dd_df.loc[dd_row[0], 'match'] = True
            gs_df.loc[gs_row[0], 'match'] = True
            
    # We ge number of unmatched detections
    FN = len(gs_df[gs_df['match'] == False])
    FP = len(dd_df[dd_df['match'] == False])
    
    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return precision, recall

            