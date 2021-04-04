import os

import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn import (metrics, linear_model, preprocessing)
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform,randint

import xgboost as xgb 
from xgboost import plot_importance


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def read_data(path): 
    '''
    csv 타입의 데이터 불러오는 함수
    '''
    df = pd.read_csv(path, na_values='NA') 
    print ("features:") 
    print (list(df.columns)) 
    print ("row and column number") 
    print (df.shape) 
    print ("data / feature types:") 
    print (df.dtypes) 
    df_num = df.select_dtypes(include='number')
    df_cat = df.select_dtypes(include='object')
    print ("missing values:") 
    print (df.isnull().sum()) 
    return [df, df_num, df_cat] 


def main():
    path = os.getcwd()
    data_path = os.path.join(path,'xgboost_study/02_xgboost_basic/data/sales_buy.csv')

    dflist = read_data(data_path)

    df = dflist[0]
    predictor = list(df.columns)

    for s in ['customerid', 'buy']:
        predictor.remove(s)
        
    print (predictor)

    X = df[predictor]
    y = df.buy

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)


    # model = xgb.XGBClassifier() 
    model = xgb.XGBRegressor() 
    model.fit(X_train, y_train) 

    # y_pred = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test) 

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)  
    roc_auc = metrics.auc(fpr, tpr)  
    print ("auc is", roc_auc)


if __name__ == "__main__":
    main()