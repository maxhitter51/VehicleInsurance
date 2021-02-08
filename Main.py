# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:08:02 2021

@author: Mani
"""

import pandas as pd
from DataCleaning import clean_data
from Normalization import normalize_num_data,encode_cat_data
from FillData import fill_numeric_data
from StatisticalTest import stat_test
from DataImbalance import balance_data
from NewModel import gred_boost
import pickle

df  = pd.read_csv("train.csv")
df=df.drop("id",axis=1)


df['Driving_License']=df['Driving_License'].astype('object')
df['Previously_Insured']=df['Previously_Insured'].astype('object')
df['Response']=df['Response'].astype('object')

clean_data(df)

df_num=df.select_dtypes(exclude='object')
df_cat=df.select_dtypes(include='object')

fill_numeric_data(df_num)

stat_test(df,df_num)


df_num=df_num.drop(['Vintage','Policy_Sales_Channel','Region_Code'],axis=1)




normalize_num_data(df_num).head()

encode_cat_data(df_cat).head()


X=pd.concat([encode_cat_data(df_cat),normalize_num_data(df_num)],axis=1)
y=df['Response']

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['feature']=X.columns
vif.sort_values('VIF',ascending=False)

y.value_counts()

X_oversample,y_oversample,xtest,ytest = balance_data(X,y)
gred_boost(X_oversample,y_oversample,xtest,ytest)

model1 = pickle.load(open('gboost_model.pkl','rb'))
ypred=model1.predict(xtest[1:2])
print("ypred",ypred)
   

