import os 
import sys

from src.exception_handling import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass


 
@dataclass()
class DataCleaningConfig:
    
    # path for cleaned data
    cleaned_data_path: str = os.path.join('artifacts', 'data_cleaned.csv')
    
    

class DataCleaning:
    
    def __init__(self):
        self.data_cleaning_config = DataCleaningConfig()
    
    def initiate_data_cleaning(self):
        """
        Data cleaning function 
        Saves the cleaned data as a csv file
        """
        try:
            logging.info("Data cleaning process has been started.")

            df = pd.read_csv('data/data.csv')
            logging.info("Data has been loaded.")
                   
            
            
            # Change the dtype of 'TotalCharges' to float64
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')



            # Change the values and dtype of 'SeniorCitizen' to object 
            df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
            df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
            print(df['SeniorCitizen'].dtype)

 

            # duplicate records
            dup_rows = df.duplicated().sum()
            # remove duplicate rows 
            df.drop_duplicates(inplace=True)
            
            
            
            # handling missing values
            df.isna().sum()
            # there are just 11 missing values out of the 7043 records we can just remove them 
            df.dropna(axis=0, how='any', inplace=True)
            logging.info("Missing values has been handled.")
            
            
            
            # removing outliers function 
            def remove_outliers(df, cols, threshold=2.6):
                """
                Removes outliers as per z-score threshold of 2.6
                """
                df_clean = df.copy()
                total_outliers = 0
                for col in cols:
                    mean = np.mean(df_clean[col])
                    std = np.std(df_clean[col])
                    z_scores = np.abs((df_clean[col] - mean) / std)
                    outliers = df_clean[z_scores >= threshold]
                    df_clean = df_clean[z_scores < threshold]
                    
                    num_outliers = len(outliers)
                    total_outliers += num_outliers
    
                return df_clean, total_outliers
            
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            df_cleaned, total_outliers = remove_outliers(df, num_cols)
            
            logging.info(f"The number of the outliers is: {total_outliers}")
            logging.info("Outliers has been removed.")
            
            
            # converting values of Churn
            df_cleaned.Churn.value_counts()
            df_cleaned['Churn'] = df_cleaned['Churn'].replace({'No': 0, 'Yes':1})
            
            
            
            # removing 'customerID' column
            df_cleaned.drop('customerID', axis=1, inplace=True)
            
            
            
            # grouping tenure into categories of 12 months
            labels = ["{0} - {1}".format(i, i+11) for i in range(1, 72, 12)]
            df_cleaned['tenure'] = pd.cut(df['tenure'], range(1, 80, 12), right=False, labels=labels)
            
            
            
            logging.info(f"The shape of the cleaned dataframe is: {df_cleaned.shape}")
            
             
            
            # creating artifacts folder & save the df
            os.makedirs(os.path.dirname(self.data_cleaning_config.cleaned_data_path), exist_ok=True)
            
            df_cleaned.to_csv(self.data_cleaning_config.cleaned_data_path, index=False)
            logging.info("Cleaned data has been saved successfully.")
            logging.info("Data cleaning process has been completed.")
            
            
            
        except Exception as e:
            raise CustomException(e, sys)
            

#if __name__ == "__main__":
    
    #cleaning_obj = DataCleaning()
    #cleaning_obj.initiate_data_cleaning()
    