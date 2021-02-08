# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:16:10 2021

@author: Mani
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer

def normalize_num_data(df_num):
    pt=PowerTransformer()
    df_num_pt=pt.fit_transform(df_num)
    df_num_pt=pd.DataFrame(df_num_pt)
    df_num_pt.columns=df_num.columns
    return df_num_pt
def encode_cat_data(df_cat):    
    df_cat=df_cat.drop('Response',axis=1)
    df_cat_dum=pd.get_dummies(df_cat,columns=['Gender','Driving_License','Previously_Insured','Vehicle_Damage'],drop_first=True)
    v_age = {'> 2 Years':0, '< 1 Year':1, '1-2 Year':2}
    df_cat_dum['Vehicle_Age'] = df_cat['Vehicle_Age'].map(lambda x : v_age[x])
    df_cat_dum.rename(columns={'Gender_Male':'Gender', 'Driving_License_1':'Driving_License','Previously_Insured_1':'Previously_Insured', 'Vehicle_Damage_Yes':'Vehicle_Damage'},inplace=True)
    return df_cat_dum