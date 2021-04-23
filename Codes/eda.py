# -*- coding: utf-8 -*-
"""
Updated on Fri Apr 23 22:36:30 2021

@author: Priyanka Dawn
"""

"""
Created on Sat Apr 17 14:11:51 2021

@author: Subhajit.Debnath
"""

# Module 2 - Data cleaning and EDA
##################################################
import os
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
# from . import utils

# import matplotlib.pyplot as plt
# plt.ioff()

#Model input for all SKUs
def processedData(SKUOutDir, Model_input_subset, SKU_code):
    
    # Total Sales per week
    weekly_sales = pd.DataFrame(Model_input_subset.groupby(['ISO_week'])['Sales'].sum().reset_index())
    plt = weekly_sales['Sales'].plot(kind='line')
    plt.tick_params(axis='x',which='minor',direction='out',bottom=True,length=5)
    plt.set_xlabel('ISO_week')
    plt.set_ylabel('Sales')
    fig = plt.get_figure()
    plt.plot()
    fig.savefig(SKUOutDir+'/sale_trend.jpg')
    # plt.close(fig)
    

    sku=SKU_code
    sku_df=Model_input_subset
    ax = sku_df.boxplot(by='SEASON',column='Sales', grid=False)
    ax.set_title('Season for {}'.format(SKU_code))
    
    fig = ax.get_figure()
    fig.savefig(SKUOutDir+'/box_plot.jpg')
    # plt.close(fig)
        
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
    
        
    skuStatsFile = open(os.path.join(SKUOutDir,str('stats_' + str(SKU_code) +'.txt')),'w+')
    skuStatsFile.close()
    
    # Test for stationarity
    def adf_check(time_series,sku,lag, *args, **kwargs):
        """
        Pass in a time series, returns ADF report
        """
        # skuStatsFile=open(os.path.join(SKUOutDir,str('stats_' + str(SKU_code) +'.txt')),'a')
        
        result = adfuller(time_series)
        # print('{} :'.format(lag))
        print('{} :'.format(lag),file = skuStatsFile)
        print(str("-"*len(str(format(lag)+" :"))), file=skuStatsFile)
        
        # print('Augmented Dickey-Fuller Test for SKU - {} for {} :'.format(sku,lag),file = skuStatsFile)
        labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
        # print('Augmented Dickey-Fuller Test for SKU - {} for {} :'.format(sku,lag),file = skuStatsFile)
    
        for value,label in zip(result,labels):
            # print(label+' : '+str(value))
            print(label+' : '+str(value),file = skuStatsFile)
        # print('\n');print('\n',file = skuStatsFile);
        
        if result[1] <= 0.05:
            # print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
            print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary",file = skuStatsFile)
        else:
            # print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
            print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ",file = skuStatsFile)
        
        # print('\n')
        return '\n'
        
    # Change this as per promo
        
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
        
        # print('Augmented Dickey-Fuller Test for SKU: {}\n'.format(sku))
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
    
    if sku_df.shape[0]>100:
        
        sku_df.set_index(['ISO_week'], inplace=True)
        sku_df.drop(['SKU','SEASON','Promo_flag'], axis=1, inplace=True)
        
        sku_df['Sales Weekly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(1)
        fig1 = plot_acf(sku_df['Sales Weekly Difference'].dropna())
        fig1.suptitle("Sales Weekly Difference ACF plot for - {}".format(sku))
        fig1.savefig(SKUOutDir+'/ACF_Weekly')
        # plt.close(fig1)
        
        sku_df['Sales Monthly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(4)
        fig2 = plot_acf(sku_df['Sales Monthly Difference'].dropna())
        fig2.suptitle("Sales Monthly Difference ACF plot for - {}".format(sku))
        fig2.savefig(SKUOutDir+'/ACF_montly')
        # plt.close(fig2)
            
        sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
        fig3 = plot_acf(sku_df['Sales Seasonal Difference'].dropna())
        fig3.suptitle("Sales Seasonal Difference ACF plot for - {}".format(sku))
        fig3.savefig(SKUOutDir+'/ACF_seasonal')
        #plt.close(fig3)
    else:
        
        sku_df.set_index(['ISO_week'], inplace=True)
        sku_df.drop(['SKU','SEASON','Promo_flag'], axis=1, inplace=True)
        
        sku_df['Sales Weekly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(1)
        fig1 = plot_acf(sku_df['Sales Weekly Difference'].dropna())
        fig1.suptitle("Sales Weekly Difference ACF plot for - {}".format(sku))
        fig1.savefig(SKUOutDir+'/ACF_Weekly')
        # plt.close(fig1)
        
        sku_df['Sales Monthly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(4)
        fig2 = plot_acf(sku_df['Sales Monthly Difference'].dropna())
        fig2.suptitle("Sales Monthly Difference ACF plot for - {}".format(sku))
        fig2.savefig(SKUOutDir+'/ACF_montly')
        # plt.close(fig2)

       
       
    
    
    #Sample zise
    ss=Model_input_subset.shape[0]
        
    # Partial - Autocorrelation Plots
    if sku_df.shape[0]>100:
        sku_df.set_index(['ISO_week'], inplace=True)
        sku_df.drop(['SKU','SEASON','Promo_flag'], axis=1, inplace=True)
        
        sku_df['Sales Weekly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(2)
        fig4 = plot_pacf(sku_df['Sales Weekly Difference'].dropna(),lags=int(ss/2-3))
        fig4.suptitle("Sales Weekly Difference PACF plot for - {}".format(sku))    
        fig4.savefig(SKUOutDir+'/PACF_Weekly',bbox_inches='tight')
        # plt.close(fig4)
        
        #DYNAMIC LAG CREATION BASED ON SAMPLE SIZE
        sku_df['Sales Monthly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(4)
        fig5 = plot_pacf(sku_df['Sales Monthly Difference'].dropna(),lags=int(ss/2-4))
        fig5.suptitle("Sales Monthly Difference PACF plot for - {}".format(sku))
        fig5.savefig(SKUOutDir+'/PACF_montly')
        # plt.close(fig5)
        
        sku_df['Sales Seasonal Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(13)
        fig6 = plot_pacf(sku_df['Sales Seasonal Difference'].dropna(),lags=int(ss/2-13))
        fig6.suptitle("Sales Seasonal Difference PACF plot for - {}".format(sku))
        fig6.savefig(SKUOutDir+'/PACF_seasonal')
        #plt.close(fig6)
    else:
        sku_df.set_index(['ISO_week'], inplace=True)
        sku_df.drop(['SKU','SEASON','Promo_flag'], axis=1, inplace=True)
        
        sku_df['Sales Weekly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(2)
        fig4 = plot_pacf(sku_df['Sales Weekly Difference'].dropna(),lags=int(ss/2-3))
        fig4.suptitle("Sales Weekly Difference PACF plot for - {}".format(sku))    
        fig4.savefig(SKUOutDir+'/PACF_Weekly',bbox_inches='tight')
        # plt.close(fig4)
        
        #DYNAMIC LAG CREATION BASED ON SAMPLE SIZE
        sku_df['Sales Monthly Difference'] = sku_df['Sales'] - sku_df['Sales'].shift(4)
        fig5 = plot_pacf(sku_df['Sales Monthly Difference'].dropna(),lags=int(ss/2-4))
        fig5.suptitle("Sales Monthly Difference PACF plot for - {}".format(sku))
        fig5.savefig(SKUOutDir+'/PACF_montly')
        # plt.close(fig5)
           
    # EWMA on using different spans

    sku_df['EWMA_2_weeks'] = sku_df['Sales'].ewm(span=2).mean()
    sku_df['EWMA_4_weeks'] = sku_df['Sales'].ewm(span=4).mean()
    sku_df['EWMA_8_weeks'] = sku_df['Sales'].ewm(span=8).mean()
    sku_df['EWMA_13_weeks'] = sku_df['Sales'].ewm(span=13).mean()
    ax = sku_df[['Sales','EWMA_2_weeks','EWMA_4_weeks','EWMA_8_weeks','EWMA_13_weeks']].plot()
    ax.set_title('EWMA Sales plot for - {}'.format(sku))
    fig = ax.get_figure()
    fig.savefig(SKUOutDir+'/EWMA_nspans')
    
