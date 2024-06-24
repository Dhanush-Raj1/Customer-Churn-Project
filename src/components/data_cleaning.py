# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:06:37 2024

@author: Dhanush
"""

import pandas as pd
import numpy as np

df_original = pd.read_csv('data/data.csv')
df = df_original.copy()

df




# 1. changing dtypes 
df.info()

# Change the dtype of 'TotalCharges' to float64
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')




# 2. Change the values and dtype of 'SeniorCitizen' to object 
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})

df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')

print(df['SeniorCitizen'].dtype)


# 3. handling missing values
df.isna().sum()

# there are just 11 missing values out of the 7043 records we can just remove them 
df.dropna(axis=0, how='any', inplace=True)




# 4. duplicate records
#dup_rows = df.duplicated().sum()
#print(f"Number of dupliated rows: {dup_rows}")

# remove duplicate rows 
#df.drop_duplicates(inplace=True)




# 5. saving the dataframe
df.to_csv('data/data_cleaned.csv', index=False)
