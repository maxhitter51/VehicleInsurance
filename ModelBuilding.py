# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:20:20 2021

@author: Mani
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix , plot_roc_curve
import matplotlib.pyplot as plt


def model_test(X_oversample,y_oversample,xtest,ytest,choice = 1):
    if choice == 1:
        from sklearn.linear_model import LogisticRegression
        #from sklearn.model_selection import KFold,cross_val_score
        log=LogisticRegression()
        log.fit(X_oversample,y_oversample)
        ypred=log.predict(xtest)
        #CLASSIFICATION REPORT
        print(classification_report(ytest,ypred))
        #ROC CURVE
        plot_roc_curve(log , xtest , ytest)
        plt.show()
        
        #BIAS AND VARIANCE
        #kf = KFold(shuffle=True , n_splits=5 , random_state=7)
        #score = cross_val_score(log , X , y , cv=kf , scoring='roc_auc')
        #bias1 = np.mean(1-score)
        #variance1 = np.std(score , ddof=1)
        #print(bias1 , variance1)
        
    elif choice == 2:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import KFold,cross_val_score
        NB = GaussianNB()
        NB.fit(X_oversample,y_oversample)
        ypred=NB.predict(xtest)
        #CLASSIFICATION REPORT
        print(classification_report(ytest,ypred))
        #ROC CURVE
        plot_roc_curve(NB , xtest , ytest)
        plt.show()
        
    elif choice == 3:
        from sklearn.neighbors import KNeighborsClassifier
        KNN = KNeighborsClassifier()
        KNN.fit(X_oversample,y_oversample)
        ypred=KNN.predict(xtest)
        print(classification_report(ytest,ypred))
        plot_roc_curve(KNN , xtest , ytest)
        plt.show()
        
    elif choice == 4:
        from sklearn.ensemble import GradientBoostingClassifier
        GBoost=GradientBoostingClassifier(n_estimators=100)
        GBoost.fit(X_oversample,y_oversample)
        ypred=GBoost.predict(xtest)
        print(classification_report(ytest,ypred))
        plot_roc_curve(GBoost , xtest , ytest)
        plt.show()

    elif choice ==5:
        from xgboost import XGBClassifier
        XGB = XGBClassifier()
        XGB
        XGB.fit(X_oversample,y_oversample,eval_metric='auc')
        XGB
        ypred = XGB.predict(xtest)
        print(classification_report(ytest,ypred))
        plot_roc_curve(XGB , xtest , ytest)
        plt.show()