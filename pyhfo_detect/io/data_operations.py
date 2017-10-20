# -*- coding: utf-8 -*-
# Copyright (c) 2016, HFO-detect Development Team.

import pandas as pd
import numpy as np

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
        