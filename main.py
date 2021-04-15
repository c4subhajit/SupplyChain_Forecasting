# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:12:50 2021

@author: Priyanka.Dawn
"""


#Importing required libraries
import pandas as pd
import numpy as np
import os
import shutil
import datetime
from datetime import datetime
import matplotlib as plt
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from pmdarima.arima import auto_arima

import warnings
warnings.filterwarnings("ignore")


# def __init__(self,path_wd,data_path):
#     self.path_wd=os.chdir('D:\\PyCharm_Projects\\forecasting_v1')
#     self.data_path = './Data'
    
#File path
#os.chdir('C:\\Users\\Priyanka\\Documents\\Git')
#data_path = './Sample data'

os.chdir('D:\\PyCharm_Projects\\forecasting_v1')
data_path = './Data'

import models

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

try:
    # Path(os.path.join(path,'Output')).rmdir()
    shutil.rmtree(os.path.join(path,'Output'), ignore_errors=True)
except OSError as e:
    print("Error: %s : %s" % (os.path.join(path,'Output'), e.strerror))
    
if not os.path.exists('Output'):
        os.makedirs('Output')
        
# os.listdir(os.getcwd())
# SKU_code=10305
#merged_data=merged_data[merged_data['SKU']==SKU_code]


#Creating output directory
outDir= os.path.join(path,'Output')

#Model input for all SKUs
Model_input=merged_data.copy(deep=True)

##subsetting the merged data for one SKU

for SKU_code in Model_input.SKU.unique():
    
    Model_input_subset=Model_input[Model_input['SKU']==SKU_code]
    Model_input_subset=Model_input_subset.drop_duplicates(subset=['ISO_week'])
    
    #Creating SKU code folder in output directory
    if not os.path.exists(os.path.join(outDir,str(SKU_code))):
            os.makedirs(os.path.join(outDir,str(SKU_code)))
    SKUOutDir = os.path.join(outDir,str(SKU_code))
    
    # Total Sales per week
    weekly_sales = pd.DataFrame(Model_input_subset.groupby(['ISO_week'])['Sales'].sum().reset_index())
    plt = weekly_sales['Sales'].plot(kind='line')
    plt.tick_params(axis='x',which='minor',direction='out',bottom=True,length=5)
    plt.set_xlabel('ISO_week')
    plt.set_ylabel('Sales')
    fig = plt.get_figure()
    plt.plot()
    fig.savefig(SKUOutDir+'/sale_trend.jpg')
       
    # Box plots
    #sku_df = Model_input_subset.groupby(['SKU'])
    for sku, sku_df in Model_input_subset.groupby(['SKU']):
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
    
    treated_data_3sd = outlier_mean3sd(Model_input_subset,'Sales')
    
    ## Module 3 - Feature engineering - Not many features right now- Promo flag ans season
    df_new=Model_input_subset.copy(deep=True)
    
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
        
    for sku,sku_df in Model_input_subset.groupby(['SKU']): # Change this as per promo
        
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
            
            '''sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
            # print(adf_check(sku_df['Sales Seasonal Difference'].dropna(),sku,"Seasonal Difference"))
            print(adf_check(sku_df['Sales Seasonal Difference'].dropna(),sku,"Seasonal Difference"), file=skuStatsFile)'''
        
            print(str("="*len(str("Augmented Dickey-Fuller Test for SKU: " + sku))) + "\n\n", file=skuStatsFile)
            
            skuStatsFile.close()  
    
    # Autocorrelation Plots
    for sku,sku_df in Model_input_subset.groupby(['SKU']):
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
        
        '''sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
        fig3 = plot_acf(sku_df['Sales Seasonal Difference'].dropna())
        fig3.suptitle("Sales Seasonal Difference ACF plot for - {}".format(sku))
        fig3.savefig(SKUOutDir+'/ACF_seasonal')'''
    
    #Sample zise
    ss=Model_input_subset.shape[0]
        
    # Partial - Autocorrelation Plots
    for sku,sku_df in Model_input_subset.groupby(['SKU']):
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
        
        '''sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
        fig6 = plot_pacf(sku_df['Sales Seasonal Difference'].dropna(),lags=int(ss/2-13))
        fig6.suptitle("Sales Seasonal Difference PACF plot for - {}".format(sku))
        fig6.savefig(SKUOutDir+'/PACF_seasonal')'''
    #############################################3
        
    # sku_df=df_new.groupby(['SKU'])['Sales'].sum().reset_index()
    
    # EWMA on using different spans
    for sku,sku_df in Model_input_subset.groupby(['SKU']):
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

    # Seasonal dummies
    dt_dummy=Model_input_subset[['SEASON']]
    if(dt_dummy.shape[0]>0):
      dt_fit_dummy=pd.DataFrame()    
      dt_fit_dummy=pd.get_dummies(dt_dummy,drop_first=True)
    dt_fit_dummy['ISO_week']=Model_input_subset['ISO_week']
    df_season=dt_fit_dummy
    
    # Adding it to merged data
    
    Model_input_subset=Model_input_subset.merge(df_season,how='left',on='ISO_week')
    
    del Model_input_subset['SEASON'] 
    
    for i in range(1,5):
      Model_input_subset['Lag_'+str(i)]=Model_input_subset.Sales.shift(i)        

    #For the lag columns fill -999
    Model_input_subset=Model_input_subset.fillna(-999)
    
    
    #the train , test and prediction df
    training_data=pd.DataFrame()
    testing_data=pd.DataFrame()
    predicted_lag4=pd.DataFrame()

    lag_columns = ([col for col in Model_input_subset if col.startswith('Lag_')])
    
    train_df,test_df=test_train_split(Model_input_subset,x=70)
    result_rf = pd.DataFrame()

    result_rf = models.execute_random_forest(train_df,test_df,lag_columns)
    result_xgb = models.execute_xgboost(train_df,test_df)
    
    forecst_weeks=test_df['ISO_week'].unique()
    result_rf['Weeks']=forecst_weeks
    result_xgb['Weeks']=forecst_weeks
    rf_xgb=result_rf.merge(result_xgb,how="left",on="Weeks")
    Predicted_df=pd.DataFrame()
    Predicted_df=Predicted_df.append(rf_xgb)
    Predicted_df['SKU']=SKU_code
    Final_fcast=Final_fcast.append(Predicted_df) 
    Final_fcast=Final_fcast[['Weeks','Predicted_rf','Predicted_xgb']]
    Final_fcast.to_csv(SKUOutDir+'/RF_XGB_pred_'+ str(sku) +'.csv',index=False)


