# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 01:05:32 2021

@author: Mani
"""


from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix , plot_roc_curve , f1_score
from sklearn.metrics import precision_recall_fscore_support as score
import pickle

        
from sklearn.ensemble import GradientBoostingClassifier

def gred_boost(X_oversample,y_oversample,xtest,ytest):
    GBoost=GradientBoostingClassifier(n_estimators=100)
    GBoost.fit(X_oversample,y_oversample)
    pickle.dump(GBoost, open('gboost_model.pkl','wb'))
    