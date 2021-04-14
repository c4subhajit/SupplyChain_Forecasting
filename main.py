# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:12:50 2021

@author: Priyanka.Dawn
"""


#Importing required libraries
import pandas as pd
import numpy as np
import os
#rom os import path
import datetime
from datetime import datetime
# from isoweek import Week
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pylab import savefig
import seaborn as sns
from matplotlib import pyplot
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from pmdarima.arima import auto_arima

import warnings
warnings.filterwarnings("ignore")

import models

#File path
#os.chdir('C:\\Users\\priyanka.dawn\\OneDrive - EY\\Documents\\PD\\Kaggle')
os.chdir('D:\\PyCharm_Projects\\forecasting_v1')

# data_path = './Sample data'
data_path = './Data'


## Module 1 - Collating all data files and data understanding - variable definition
## Module 2 - Data cleaning and EDA - Boxplot - Correlation
## Module 3 - Feature engineering
## Module 4 - Model Building
## Module 5 - Output with KPI 

# Module 1 - Data Preparation
#============================
#Reading input files  
Sales_weekly=pd.read_csv(os.path.join(data_path,"Sales_sample.csv"))
Promo_weekly = pd.read_csv(os.path.join(data_path,"Promo_sample.csv"))
Season_data= pd.read_csv(os.path.join(data_path,"Season_sample.csv"))


def sales_prep(sales_input, *args, **kwargs):
    #Change column names and date format
    sales_input=sales_input.sort_values(['SKU', 'ISO_week'])
    
    #Maintaining continuity in data
    SKU = sales_input["SKU"].unique()
    all_weeks = list(sales_input["ISO_week"].unique())
    sales_continuous = pd.MultiIndex.from_product([SKU,all_weeks], names = ["SKU", "ISO_week"])
    sales_continuous = pd.DataFrame(index = sales_continuous).reset_index()
    sales_continuous = sales_continuous.merge(sales_input,how='left',on=['SKU','ISO_week'])
    sales_continuous["Sales"].fillna(0,inplace=True)
    #Initial zero removal for sales 
    sales_initial=sales_continuous.sort_values(['SKU','ISO_week'],ascending=True).reset_index(drop=True)
    sales_initial.set_index(['SKU', 'ISO_week'], inplace = True)
    sales_zeroSearch = (sales_initial['Sales'] != 0).groupby(level=0).cumsum()
    sales_treated = sales_initial[sales_zeroSearch != 0].reset_index() 
    sales_treated=sales_treated.reset_index(drop = True)
    sales_treated.sort_values(['SKU','ISO_week'])
    return sales_treated

sales_processed=sales_prep(Sales_weekly)

def merge_df(sales_data,promo_data,season, *args, **kwargs):
    promo_data.rename(columns={'EAN':'SKU'},inplace=True)
    promo_data=promo_data.drop_duplicates()
    merged_data=sales_data.merge(promo_data,how='left',on=['SKU','ISO_week'])
    merged_data=merged_data.merge(Season_data,how='left',on=['ISO_week'])
    merged_data.loc[merged_data['Promo_flag'].isnull(),'Promo_flag']=0
    return merged_data
    
merged_data=merge_df(sales_processed,Promo_weekly,Season_data)
    
# Module 2 - Data cleaning and EDA
##################################################

#Creating the output directory if it does not exist

path = os.getcwd()
if not os.path.exists('Output'):
        os.makedirs('Output')
        
# os.listdir(os.getcwd())

##subsetting the merged data for one SKU
SKU_code=10305
merged_data=merged_data[merged_data['SKU']==SKU_code]


#Creating output directory
outDir= os.path.join(path,'Output')

#Creating SKU code folder in output directory
if not os.path.exists(os.path.join(outDir,str(SKU_code))):
        os.makedirs(os.path.join(outDir,str(SKU_code)))
SKUOutDir = os.path.join(outDir,str(SKU_code))

# Total Sales per week
weekly_sales = pd.DataFrame(merged_data.groupby(['ISO_week'])['Sales'].sum().reset_index())
plt= weekly_sales['Sales'].plot(kind='line')
plt.tick_params(axis='x',which='minor',direction='out',bottom=True,length=5)
plt.set_xlabel('ISO_week')
plt.set_ylabel('Sales')
fig = plt.get_figure()
plt.plot()
fig.savefig(SKUOutDir+'/sale_trend.jpg')

#Missing data - Need to add promo and holiday

# Box plots
'''n=10
sku_sales= pd.DataFrame(merged_data.groupby(['SKU'])['Sales'].sum())
sku_topn =sku_sales.sort_values(['Sales'],ascending = False)['SKU'].unique()[0:n]
sku_sales_top=sku_sales[sku_sales['SKU'].isin(sku_topn)]'''

for sku, sku_df in merged_data.groupby(['SKU']):
    ax = sku_df.boxplot(by='SEASON',column='Sales', grid=False)
    ax.set_title('Season for {}'.format(sku))

fig = ax.get_figure()
fig.savefig(SKUOutDir+'/box_plot.jpg')
    
#Outlier treatment
def outlier_mean3sd(df,column_name, *args, **kwargs):
    upper_level=df[column_name].mean()+3*df[column_name].std()
    lower_level=df[column_name].mean()-3*df[column_name].std()
    df['sales_treated']=np.where(df[column_name]>upper_level,upper_level,df[column_name])
    df['sales_treated']=np.where(df[column_name]<lower_level,lower_level,df['sales_treated'])
    del df[column_name]
    df.rename(columns={'sales_treated':column_name},inplace=True)
    return(df)

def outlier_mean2sd(df,column_name, *args, **kwargs):
    upper_level=df[column_name].mean()+2*df[column_name].std()
    lower_level=df[column_name].mean()-2*df[column_name].std()
    df['sales_treated']=np.where(df[column_name]>upper_level,upper_level,df[column_name])
    df['sales_treated']=np.where(df[column_name]<lower_level,lower_level,df['sales_treated'])
    del df[column_name]
    df.rename(columns={'sales_treated':column_name},inplace=True)
    return(df)

treated_data_3sd = outlier_mean3sd(merged_data,'Sales')

## Module 3 - Feature engineering - Not many features right now- Promo flag ans season
df_new=merged_data.copy(deep=True)

skuStatsFile = open(os.path.join(SKUOutDir,str('stats_' + str(SKU_code) +'.txt')),'w+')
skuStatsFile.close()

# Test for stationarity
def adf_check(time_series,sku,lag, *args, **kwargs):
    """
    Pass in a time series, returns ADF report
    """
    # skuStatsFile=open(os.path.join(SKUOutDir,str('stats_' + str(SKU_code) +'.txt')),'a')
    
    result = adfuller(time_series)
    print('{} :'.format(lag))
    print('{} :'.format(lag),file = skuStatsFile)
    print(str("-"*len(str(format(lag)+" :"))), file=skuStatsFile)
    
    # print('Augmented Dickey-Fuller Test for SKU - {} for {} :'.format(sku,lag),file = skuStatsFile)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    # print('Augmented Dickey-Fuller Test for SKU - {} for {} :'.format(sku,lag),file = skuStatsFile)

    for value,label in zip(result,labels):
        print(label+' : '+str(value))
        print(label+' : '+str(value),file = skuStatsFile)
    # print('\n');print('\n',file = skuStatsFile);
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary",file = skuStatsFile)
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ",file = skuStatsFile)
    
    print('\n')
    return '\n'
    
for sku,sku_df in df_new.groupby(['SKU']): # Change this as per promo
    
    try:
      if sku is not None:
          sku=format(sku)
      else:
          raise TypeError("Null values not allowed")
    except:
        print("SKU value Null")
    else:
        sku_df.set_index(['ISO_week'], inplace=True)
        sku_df.drop(['SKU','Promo_flag'], axis=1, inplace=True)
        
        skuStatsFile=open(os.path.join(SKUOutDir,str('stats_' + str(SKU_code) +'.txt')),'a')
        
        print("ADF checks for {}".format(sku), file=skuStatsFile)
        print(str("#"*len(str("ADF checks for " + sku))) + "\n\n", file=skuStatsFile)
        
        print('Augmented Dickey-Fuller Test for SKU: {}\n'.format(sku))
        print('Augmented Dickey-Fuller Test for SKU: {}'.format(sku),file = skuStatsFile)
        print(str("="*len(str("Augmented Dickey-Fuller Test for SKU: " + sku))) + "\n", file=skuStatsFile)
    
        sku_df['Sales Weekly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(1)
        # print(adf_check(sku_df['Sales Weekly Difference'].dropna(),sku,"Weekly Difference"))
        print(adf_check(sku_df['Sales Weekly Difference'].dropna(),sku,"Weekly Difference",'hello'),file = skuStatsFile)
        
        sku_df['Sales Monthly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(4)
        # print(adf_check(sku_df['Sales Monthly Difference'].dropna(),sku,"Monthly Difference"))
        print(adf_check(sku_df['Sales Monthly Difference'].dropna(),sku,"Monthly Difference"), file=skuStatsFile)
        
        sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
        # print(adf_check(sku_df['Sales Seasonal Difference'].dropna(),sku,"Seasonal Difference"))
        print(adf_check(sku_df['Sales Seasonal Difference'].dropna(),sku,"Seasonal Difference"), file=skuStatsFile)
    
        print(str("="*len(str("Augmented Dickey-Fuller Test for SKU: " + sku))) + "\n\n", file=skuStatsFile)
        
        skuStatsFile.close()  

# Autocorrelation Plots
for sku,sku_df in df_new.groupby(['SKU']):
    sku_df_main = sku_df.copy()
    sku_df.set_index(['ISO_week'], inplace=True)
    sku_df.drop(['SKU','SEASON','Promo_flag'], axis=1, inplace=True)
    
    sku_df['Sales Weekly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(1)
    fig1 = plot_acf(sku_df['Sales Weekly Difference'].dropna())
    fig1.suptitle("Sales Weekly Difference ACF plot for - {}".format(sku))
    fig1.savefig(SKUOutDir+'/ACF_Weekly')
    
    sku_df['Sales Monthly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(4)
    fig2 = plot_acf(sku_df['Sales Monthly Difference'].dropna())
    fig2.suptitle("Sales Monthly Difference ACF plot for - {}".format(sku))
    fig2.savefig(SKUOutDir+'/ACF_montly')
    
    sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
    fig3 = plot_acf(sku_df['Sales Seasonal Difference'].dropna())
    fig3.suptitle("Sales Seasonal Difference ACF plot for - {}".format(sku))
    fig3.savefig(SKUOutDir+'/ACF_seasonal')

#Sample zise
ss=df_new.shape[0]
    
# Partial - Autocorrelation Plots
for sku,sku_df in df_new.groupby(['SKU']):
    sku_df.set_index(['ISO_week'], inplace=True)
    sku_df.drop(['SKU','SEASON','Promo_flag'], axis=1, inplace=True)
    
    sku_df['Sales Weekly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(2)
    fig4 = plot_pacf(sku_df['Sales Weekly Difference'].dropna(),lags=int(ss/2-3))
    fig4.suptitle("Sales Weekly Difference PACF plot for - {}".format(sku))    
    fig4.savefig(SKUOutDir+'/PACF_Weekly',bbox_inches='tight')
    
    #DYNAMIC LAG CREATION BASED ON SAMPLE SIZE
    sku_df['Sales Monthly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(4)
    fig5 = plot_pacf(sku_df['Sales Monthly Difference'].dropna(),lags=int(ss/2-4))
    fig5.suptitle("Sales Monthly Difference PACF plot for - {}".format(sku))
    fig5.savefig(SKUOutDir+'/PACF_montly')
    
    sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
    fig6 = plot_pacf(sku_df['Sales Seasonal Difference'].dropna(),lags=int(ss/2-13))
    fig6.suptitle("Sales Seasonal Difference PACF plot for - {}".format(sku))
    fig6.savefig(SKUOutDir+'/PACF_seasonal')
#############################################3
    
# sku_df=df_new.groupby(['SKU'])['Sales'].sum().reset_index()

# EWMA on using different spans
for sku,sku_df in df_new.groupby(['SKU']):
    #sku_df.set_index(['ISO_week'], inplace=True)
    #sku_df.drop(['SKU','ISO_Week'], axis=1, inplace=True)
    sku_df['EWMA_2_weeks'] = sku_df['Sales'].ewm(span=2).mean()
    sku_df['EWMA_4_weeks'] = sku_df['Sales'].ewm(span=4).mean()
    sku_df['EWMA_8_weeks'] = sku_df['Sales'].ewm(span=8).mean()
    sku_df['EWMA_13_weeks'] = sku_df['Sales'].ewm(span=13).mean()
    ax = sku_df[['Sales','EWMA_2_weeks','EWMA_4_weeks','EWMA_8_weeks','EWMA_13_weeks']].plot()
    ax.set_title('EWMA Sales plot for - {}'.format(sku))
    fig = ax.get_figure()
    fig.savefig(SKUOutDir+'/EWMA_nspans')



#Define test and train
def test_train_split(df,x, *args, **kwargs):
    points = np.round(len(df['Sales'])*(x)/100).astype(int) 
    train_data = df.loc[:points]
    test_data = df.loc[points:]
    return train_data,test_data

#train_df,test_df=test_train_split(df_new,x=70)


Final_fcast=pd.DataFrame()
#Running random forest regressor for each SKU
for Sku in df_new['SKU'].unique():
    df_subset=df_new[df_new['SKU']==Sku]
    # Seasonal dummies
    dt_dummy=df_subset[['SEASON']]
    if(dt_dummy.shape[0]>0):
      dt_fit_dummy=pd.DataFrame()    
      dt_fit_dummy=pd.get_dummies(dt_dummy,drop_first=True)
    dt_fit_dummy['ISO_week']=df_subset['ISO_week']
    df_season=dt_fit_dummy
    
    # Adding it to merged data
    
    df_subset=df_subset.merge(df_season,how='left',on='ISO_week')
    
    del df_subset['SEASON'] 
    
    for i in range(1,5):
      df_subset['Lag_'+str(i)]=df_subset.Sales.shift(i)        

    #For the lag columns fill -999
    df_subset=df_subset.fillna(-999)
    
    
    #the train , test and prediction df
    training_data=pd.DataFrame()
    testing_data=pd.DataFrame()
    predicted_lag4=pd.DataFrame()

    lag_columns = ([col for col in df_subset if col.startswith('Lag_')])
    
    train_df,test_df=test_train_split(df_subset,x=70)
    result_rf = pd.DataFrame()

    result_rf = models.execute_random_forest(train_df,test_df,lag_columns)
    result_xgb = models.execute_xgboost(train_df,test_df)
    
    forecst_weeks=test_df['ISO_week'].unique()
    result_rf['Weeks']=forecst_weeks
    result_xgb['Weeks']=forecst_weeks
    rf_xgb=result_rf.merge(result_xgb,how="left",on="Weeks")
    Predicted_df=pd.DataFrame()
    Predicted_df=Predicted_df.append(rf_xgb)
    Predicted_df['SKU']=Sku
    Final_fcast=Final_fcast.append(Predicted_df) 
    Final_fcast=Final_fcast[['Weeks','Predicted_rf','Predicted_xgb']]
    Final_fcast.to_csv(SKUOutDir+'/RF_XGB_pred_'+ str(sku) +'.csv',index=False)



   
##Edit here onwards
'''feature_importance=random_forest_importance_matrix(training_data.copy(deep=True))
    feature_importance_dev=feature_importance.sort_values('Importance', ascending= False).iloc[:15,]
    feature_importance_dev['Key']= fu
    feature_importance_dev['Snapshot_Week']= fcast_weeks[0]
    feature_importance_dev = feature_importance_dev.pivot(index=['Key','Snapshot_Week'], columns='Variable',values='Importance').reset_index() '''   
    




# Additional univariate models
'''def run_fcast(ser,model,h,period):
    if model == "arima":
        print("Running Arima.........")
        stepwise_model_arima = auto_arima(ser, start_p=0, start_q=0,max_p=2, max_q=2, m=period,seasonal=False,max_d=1,
                           trace=False,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        fcast = np.round(stepwise_model_arima.predict(n_periods=h)).astype(int)
        fcast[fcast<0] = 0
        return fcast;

    elif model == "ma":
        print("Running Moving Average...........")
        fcast_12 = float(ser.rolling(12).mean().iloc[-1])
        fcast_8 = float(ser.rolling(8).mean().iloc[-1])
        fcast_4 = float(ser.rolling(4).mean().iloc[-1])
        fcast_1 = float(ser.rolling(1).mean().iloc[-1])
        fcast = np.round([statistics.mean([fcast_12,fcast_8,fcast_4,fcast_1])]*h).astype(int)
        fcast[fcast<0] = 0
        return fcast;
    elif model == "simexp":
        print("Running Simple Exp..............")
        fcast = np.round(SimpleExpSmoothing(np.asarray(ser)).fit(smoothing_level=0.4,optimized=False).forecast(h)).astype(int)
        fcast[fcast<0] = 0
        return fcast;
    elif model == "holtlinear":
        print("Running Holt Linear............")
        fcast = np.round(Holt(ser).fit(smoothing_level = 0.3,smoothing_slope = 0.1).forecast(h)).astype(int)
        fcast[fcast<0] = 0
        return fcast;
    elif model == "holtwinter":
        print("Running Holt Winters...........")
        fcast = np.round(ExponentialSmoothing(ser ,seasonal_periods=52 ,trend='add', 
                                seasonal='add').fit(optimized = True).forecast(h)).astype(int)
        fcast[fcast<0] = 0
        return fcast;
    elif model == "ucm":
        print("Running UCM.........")
        model = sm.tsa.UnobservedComponents(ser, level = True, cycle=True, stochastic_cycle=True,seasonal = 52)
        pred_uc = list(model.fit().get_forecast(steps = h).predicted_mean)
        fcast = [int(round(x)) for x in pred_uc]
        pred = [0 if i < 0 else i for i in fcast]

        #fcast = fcast.astype(int)
        return pred;  
    elif model == "croston":
        print("Running Croston.........")
        pred_cros = Croston(ser,h,0.4)
        pred_cros = pred_cros[pred_cros['Demand'].isnull()]['Forecast']
        fcast = round(pred_cros).astype(int)
        fcast[fcast<0] = 0
        return fcast;  
    elif model == "naive_seasonal":
        print("Running Naive Seasonal......")
        if len(ser)>period:
            fcast = round(ser[-52:][:h]).astype(int)
            fcast[fcast<0] = 0
            return fcast;
    else:
            fcast = np.repeat(np.nan,h)
            return fcast;

    if model == "arima":
        print("Running Arima.........")
        stepwise_model_arima = auto_arima(ser, start_p=0, start_q=0,max_p=2, max_q=2, m=period,seasonal=False,max_d=1,
                           trace=False,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        fcast = np.round(stepwise_model_arima.predict(n_periods=h)).astype(int)
        fcast[fcast<0] = 0
        return fcast'''