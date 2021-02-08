# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:24:36 2021

@author: Mani
"""

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC



def balance_data(X,y):
    y=y.astype('int64')
    xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,stratify=y)
    smotenc = SMOTENC([0,1,2,3,4,5])
    X_oversample,y_oversample = smotenc.fit_resample(xtrain,ytrain)
    
    return X_oversample,y_oversample,xtest,ytest
