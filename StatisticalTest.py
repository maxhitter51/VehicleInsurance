# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:14:10 2021

@author: Mani
"""

from scipy.stats import stats


def stat_test(df,df_num):
    for i in df_num.columns:
        df_1=df[df['Response']==1][i]
        df_0=df[df['Response']==0][i]
        tsats,pval=stats.ttest_ind(df_1,df_0)
        tstas,pval=stats.mannwhitneyu(df_1,df_0)