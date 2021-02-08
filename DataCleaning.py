# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:08:36 2021

@author: Mani
"""

def clean_data(df):
    """to clean the data pass the dataframe itself """
    
    # Drop columns which have all NaN values
    c=df.columns[df.isnull().all()]
    df.drop(c, inplace=True, axis=1)
    
    # Drop Columns which have more than 90% NAs
    df.dropna(axis=1, thresh=int(0.1 * df.shape[0]),inplace=True)
    # Drop rows with missing values greater than 50%
    df = df[df.isnull().sum(axis=1) <=(df.shape[1] * 0.5) ]
    
    return df