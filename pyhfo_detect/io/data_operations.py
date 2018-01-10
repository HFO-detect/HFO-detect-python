# -*- coding: utf-8 -*-
# Copyright (c) 2016, HFO-detect Development Team.

import pandas as pd

def create_output_df(fields=[]):
    """
    Function to create a custom pandas dataframe depending on the algorithm
    needs. Fields: event_start,event_stop preset.
    
    Parameters:
    -----------
    fields(list) - additional fields for the dataframe\n
    
    Returns:
    --------
    dataframe - pandas dataframe for deteciton insertion\n
    """
    
    out_df = pd.DataFrame(columns=['event_start', 'event_stop']+fields)

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

def correct_boundary_dtype(df):
    """
    Corrects data type of event_start and event_stop columns
    """
    df['event_start'] = df['event_start'].astype('int64')
    df['event_stop'] = df['event_stop'].astype('int64')
        
    return