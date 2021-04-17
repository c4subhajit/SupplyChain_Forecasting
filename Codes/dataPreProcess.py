"""
Created on Sat Apr 17 13:29:07 2021

@author: Subhajit.Debnath
"""

import pandas as pd
import os
from . import settings 

# Module 1 - Data Preparation
#============================
#Reading input files  
Sales_weekly=pd.read_csv(os.path.join(settings.dataDir,"Sales_sample.csv"))
Promo_weekly = pd.read_csv(os.path.join(settings.dataDir,"Promo_sample.csv"))
Season_data= pd.read_csv(os.path.join(settings.dataDir,"Season_sample.csv"))

def sales_prep(sales_input=Sales_weekly, *args, **kwargs):
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



def merge_df(sales_data,promo_data=Promo_weekly,season=Season_data, *args, **kwargs):
    promo_data.rename(columns={'EAN':'SKU'},inplace=True)
    promo_data=promo_data.drop_duplicates()
    merged_data=sales_data.merge(promo_data,how='left',on=['SKU','ISO_week'])
    merged_data=merged_data.merge(Season_data,how='left',on=['ISO_week'])
    merged_data.loc[merged_data['Promo_flag'].isnull(),'Promo_flag']=0
    return merged_data
    
