# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:12:50 2021

@author: Priyanka.Dawn
"""

#Importing required libraries
import os,datetime

import warnings
warnings.filterwarnings("ignore")


try:
    # Change the current working Directory    
    os.chdir(os.path.dirname(__file__))
except OSError:
    print("Can't change the Current Working Directory")  
        

import settings    

from Codes import dataPreProcess
from Codes import eda
from Codes import utils
from Codes import modelTrain

# Delete output directory
utils.deleteDir(settings.rootDir,'Output')

# Create Output directory
outDir= utils.createDir(settings.rootDir,'Output')

# Data preprocessiong
sales_processed=dataPreProcess.sales_prep()
salesPromoData=dataPreProcess.merge_df(sales_processed)

Model_input=salesPromoData.copy(deep=True)


# EDA on processed data by each SKU
for SKU_code in salesPromoData.SKU.unique():
    t0 = datetime.datetime.now()
    
    ##subsetting the merged data for one SKU
    Model_input_subset=Model_input[Model_input['SKU']==SKU_code]
    Model_input_subset=Model_input_subset.drop_duplicates(subset=['ISO_week'])
    
    # Create SKU directory
    SKUOutDir=utils.createDir(outDir, SKU_code)
    
    # Run EDA
    eda.processedData(SKUOutDir, Model_input_subset, SKU_code)

    # Run Models
    modelTrain.executeModels(SKUOutDir,Model_input_subset)

    t1 = datetime.datetime.now() - t0
    print('Processed for ' + str(SKU_code) + '. Time Elapsed: ' + str(t1))
    
print('\nExecution Complete!')