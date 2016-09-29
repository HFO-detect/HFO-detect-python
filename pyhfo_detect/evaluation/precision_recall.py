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

def create_precision_recall_curve(gs_df, dd_df, bn, threshold,
                                  sec_unit = None, sec_margin = 1):
    """
    Function to create precision recall curve.
    
    Parameters:
    -----------
    gs_df - gold standard detections\n
    dd_df - automatically detected detections\n
    bn - names of event start stop [start_name, stop_name] (list)\n
    threshold - name of the threshold field for evaluation
    sec_unit - number representing one second of signal - this can\n
    significantly imporove the speed of this operation\n
    sec_margin - margin for creating subsets of compared data - should be set\n
    according to the legnth of compared events (1s for HFO should be enough)
    
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
        print('Processing threshold '+str(th))
        if sec_unit:
            p, r = calculate_precision_recall(gs_df,
                                              dd_df[dd_df[threshold] == th],
                                              bn,
                                              sec_unit,
                                              sec_margin)
        else:
            p, r = calculate_precision_recall(gs_df,
                                              dd_df[dd_df[threshold] == th],
                                              bn)
            
        precision.append(p)
        recall.append(r)
        
    return precision, recall
    

def calculate_precision_recall(gs_df, dd_df, bn, 
                               sec_unit = None, sec_margin = 1):
    """
    Function to calculate precision and recall values.
    
    Parameters:
    -----------
    gs_df - gold standard detections\n
    dd_df - automatically detected detections\n
    bn - names of event start stop [start_name, stop_name] (list)\n
    sec_unit - number representing one second of signal - this can\n
    significantly imporove the speed of this operation\n
    sec_margin - margin for creating subsets of compared data - should be set\n
    according to the legnth of compared events (1s for HFO should be enough)
    
    Returns:
    --------
    precision - precision of the detection set\n
    recall - recall(sensitivity) of the detection set\n
    """
    
    # Create column for matching
    dd_df.loc[:,'match'] = False
    gs_df.loc[:,'match'] = False
    
    # Initiate true positive
    TP = 0
    
    # Start running through gold standards
    for gs_row in gs_df.iterrows():
        gs_det = [gs_row[1][bn[0]], gs_row[1][bn[1]]]
        
        det_flag = False
        if sec_unit:
            for dd_row in dd_df[(dd_df[bn[0]] < gs_det[0]+sec_unit*sec_margin) &
                                (dd_df[bn[0]] > gs_det[0]-sec_unit*sec_margin)].iterrows():
                dd_det = [dd_row[1][bn[0]], dd_row[1][bn[1]]]
    
                if detection_overlap_check(gs_det, dd_det):
                    det_flag = True
                    break
        else:      
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

            