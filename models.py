# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:46:11 2021

@author: Sunny
"""

import pandas as pd
import numpy as np
import xgboost
from sklearn.ensemble import RandomForestRegressor

## Module 4 - Model Building
#Random Forest regressor

def execute_random_forest(training_data,testing_data,lag_cols, *args, **kwargs):
    col_names = list(training_data.columns)
    excluded_features= ['SKU','Sales','ISO_week']
    lag_cols=list(lag_cols)
    excluded_features=excluded_features + lag_cols
    feature_list=list(set(col_names)-set(excluded_features))
    testing_data1=testing_data[feature_list]
    training_data1=training_data[feature_list]
    x_train=training_data1
    y_train = training_data.loc[:,"Sales"]
    rf = RandomForestRegressor(bootstrap=True, criterion='mse', #criterion='mse'
            max_depth=5, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=4, min_weight_fraction_leaf=0.0, # min_samples_split=2
            n_estimators=300, n_jobs=2, oob_score=False, random_state=0, # n_estimators=300
            verbose=0, warm_start=False)
    rf.fit(x_train, y_train)
    y_test_predicted=rf.predict(testing_data1)
    #training_data['Predicted_rf']=y_train_predicted
    testing_data['Predicted_rf']=y_test_predicted.round(2)
    testing_data['Predicted_rf']=np.where(testing_data['Predicted_rf']<0,0,testing_data['Predicted_rf']) 
    result = testing_data[['Predicted_rf']]
    return result


#Xgb regressor
def execute_xgboost(training_data,testing_data, *args, **kwargs):

    col_names = list(training_data.columns)
    excluded_features= ['SKU','Sales','ISO_week']
    
    feature_list=list(set(col_names)-set(excluded_features))
    testing_data1=testing_data[feature_list]
    training_data1=training_data[feature_list]
    x_train=training_data1
    y_train = training_data.loc[:,"Sales"]

    #x_train,x_test,y_train,y_test=train_test_split(training_data[feature_list],y_train,test_size=0.2,random_state = 0)
    #data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)
    import xgboost as xgb
    #xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
     #           max_depth = 5,subsample=0.6,verbose=0,seed=123,reg_lambda=0.45,reg_alpha=0.75,gamma=0, n_estimators = 10, min_child_weight=1.5)
    xg_reg=xgb.XGBRegressor(objective ='reg:squarederror',missing=-999,seed=123)
    xg_reg.fit(x_train, y_train)

    #preds = xg_reg.predict(testing_data1)#
    #rf.fit(x_train, y_train)

    #y_train_predicted=xg_reg.predict(training_data1)
    y_test_predicted=xg_reg.predict(testing_data1)
    
    #training_data['Predicted_xgb']=y_train_predicted
    testing_data['Predicted_xgb']=y_test_predicted.round(2)
    testing_data['Predicted_xgb']=np.where(testing_data['Predicted_xgb']<0,0,testing_data['Predicted_xgb'])
    
       
    result = testing_data[['Predicted_xgb']]

    return result


# Assign weightage to each feature
# ================================
# def random_forest_importance_matrix(X, *args, **kwargs):    
#     #X = training_data
#     X = X[X.columns.drop(list(X.filter(regex='Lag_')))]
#     feature_list = list(X.columns)
#     feature_list.remove("Actuals")
#     features = np.array(X.drop(['Actuals'], axis = 1))
#     target = np.array(X['Actuals'])
#     rf = RandomForestRegressor(bootstrap=True, criterion='mse',
#             max_depth=5, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=None, min_samples_leaf=2,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=200, n_jobs=2, oob_score=False, random_state=0, ##400 or 300, depth
#             verbose=0, warm_start=False)
    
#     rf.fit(features, target)
    
#     # Get numerical feature importances
#     importances = list(rf.feature_importances_)
#     # List of tuples with variable and importance
#     feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#     # Sort the feature importances by most important first
#     feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#     df = pd.DataFrame(feature_importances, columns =['Variable', 'Importance']) 
#     return(df)
