# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 22:22:50 2021

@author: Priyanka.Dawn
"""

#Importing required libraries
import pandas as pd
import numpy as np
# import os
from Codes import models




#Define test and train
def test_train_split(proc_SKU_data, splitPercent=70, *args, **kwargs):
    
    points = np.round(len(proc_SKU_data['Sales'])*(splitPercent)/100).astype(int) 
    train_data = proc_SKU_data.loc[:points]
    test_data = proc_SKU_data.loc[points:]
    return train_data,test_data





def executeModels(outDir, proc_SKU_data, splitPercent=70, *args, **kwargs):
    #Final_fcast=pd.DataFrame()

    #Running random forest regressor for each SKU
    
    # Seasonal dummies
    dt_dummy=proc_SKU_data[['SEASON']]
    
    if(dt_dummy.shape[0]>0):
        dt_fit_dummy=pd.DataFrame()    
        dt_fit_dummy=pd.get_dummies(dt_dummy,drop_first=True)
    dt_fit_dummy['ISO_week']=proc_SKU_data['ISO_week']
    df_season=dt_fit_dummy
    
    # Adding it to merged data
    proc_SKU_data=proc_SKU_data.merge(df_season,how='left',on='ISO_week')
    
    del proc_SKU_data['SEASON'] 
    
    for i in range(1,5):
        proc_SKU_data['Lag_'+str(i)]=proc_SKU_data.Sales.shift(i)        
    
    #For the lag columns fill -999
    proc_SKU_data=proc_SKU_data.fillna(-999)
    
    
    #the train , test and prediction df
    # training_data=pd.DataFrame()
    # testing_data=pd.DataFrame()
    # predicted_lag4=pd.DataFrame()
    
    lag_columns = ([col for col in proc_SKU_data if col.startswith('Lag_')])
    
    train_df,test_df=test_train_split(proc_SKU_data)
    
    result_RF = pd.DataFrame()
    
    result_RF = models.predModels.execute_random_forest(train_df,test_df,lag_columns)
    result_xgb = models.predModels.execute_xgboost(train_df,test_df)
    
    forecst_weeks=test_df['ISO_week'].unique()
    result_RF['Weeks']=forecst_weeks
    result_xgb['Weeks']=forecst_weeks
    rf_xgb=result_RF.merge(result_xgb,how="left",on="Weeks")
    Predicted_df=pd.DataFrame()
    Predicted_df=Predicted_df.append(rf_xgb)
    
    # Predicted_df['SKU']=SKU_code
    
    #Final_fcast=Final_fcast.append(Predicted_df) 
    #Final_fcast=Final_fcast[['Weeks','Predicted_rf','Predicted_xgb']]
    
    # Final_fcast.to_csv(SKUOutDir+'/RF_XGB_pred_'+ str(sku) +'.csv',index=False)
    Predicted_df.to_csv(outDir+'/RF_XGB_pred.csv',index=False)


