# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:09:29 2021

@author: Mani
"""


import pandas as pd
from sklearn.impute import KNNImputer
def fill_numeric_data(df,neighbors = 2):
    """ provide dataframe and neighbors , by default it is 2 """
    imputer = KNNImputer(n_neighbors=neighbors, weights="uniform")
    cols = df.columns
    filled_array = imputer.fit_transform(df)
    df_filled = pd.DataFrame(filled_array, columns = cols)
    return df_filled