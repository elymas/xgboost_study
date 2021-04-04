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


def data_partition(df, features, target, seed): 
    '''
    train/test 데이터 분리해주는 함수
    '''
    X = df[features]
    Y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = seed)      
    return [X, Y, X_train, X_test, y_train, y_test] 


def tree_model(X_train, y_train, criterionv, random_statev, max_depthv, min_samples_leafv): 
    '''
    모델 학습(생성) 함수
    '''
    clf = DecisionTreeClassifier(
        criterion = criterionv,
        random_state = random_statev,
        max_depth = max_depthv,
        min_samples_leaf = min_samples_leafv
    ) 
    clf.fit(X_train, y_train) 
    return clf 


def tree_prediction(X_test, clf): 
    '''
    모델 예측 함수
    '''
    y_pred = clf.predict(X_test) 
    print(y_pred) 
    return y_pred 


def get_accuracy(y_test, y_pred): 
    '''
    모델 성능지표 리포팅 함수
    '''
    print("Confusion Matrix: ", 
    confusion_matrix(y_test, y_pred)) 

    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 

    print("Report : ", 
    classification_report(y_test, y_pred)) 


def plot_confusion(labels, y_test, pred):
    '''
    confusion matrix 시각화 함수
    '''  
    cm = confusion_matrix(y_test, pred, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    path = os.getcwd()
    data_path = os.path.join(path,'xgboost_study/01_decision_tree/data/BankNote_Authentication.csv')
    dflist = read_data(data_path)
    df = dflist[0]
    df.rename(columns = {'class':'target'}, inplace = True)
    df.head()
    df.target.mean()

    features = list(df.columns)
    features.remove('target')

    X, Y, X_train, X_test, y_train, y_test = data_partition(df, features, 'target', 12)

    tree_gini = tree_model(X_train, y_train, 'gini', 25, 3, 4)
    tree_entropy = tree_model(X_train, y_train, 'entropy', 67, 3, 4)

    gini_predict= tree_prediction(X_test, tree_gini) 
    get_accuracy(y_test, gini_predict) 
    # plot_confusion([0,1], y_test, gini_predict)
    
    entropy_predict= tree_prediction(X_test, tree_entropy) 
    get_accuracy(y_test, entropy_predict)     
    # plot_confusion([0,1], y_test, entropy_predict)  

    # tree plot
    tree.plot_tree(tree_gini)
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    tree.plot_tree(tree_gini,feature_names = features,class_names= ['0', '1'],filled = True)
    savefig_path = os.path.join(path,'xgboost_study/01_decision_tree/gini_tree.png')
    fig.savefig(savefig_path)

    tree.plot_tree(tree_entropy)
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    tree.plot_tree(tree_entropy,feature_names = features,class_names= ['0', '1'],filled = True)
    savefig_path = os.path.join(path,'xgboost_study/01_decision_tree/entropy_tree.png')
    fig.savefig(savefig_path)

    # AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_test, gini_predict)  
    roc_auc = metrics.auc(fpr, tpr)  
    print ("auc: roc prob -- auc", roc_auc)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, entropy_predict)  
    roc_auc = metrics.auc(fpr, tpr)  
    print ("auc: roc prob -- auc", roc_auc)


if __name__ == "__main__":
    main()