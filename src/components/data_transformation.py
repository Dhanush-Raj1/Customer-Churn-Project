import sys
import os
from src.exception_handling import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import KMeansSMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

 

# Data transformation configuration class
@dataclass
class DataTransformationConfig:
    
    # path for preprocessor object 
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    

# Data transformation class     
class DataTransformation:
    """
    Import the cleaned data 
    Performs train, test split & Saves it in artifacts folder 
    Preprocess the train, test sets and returns them as an array 
    """ 
    
    # Initiate the configuration object
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformer_object(self):
        
        """
        Building preprocessor object and returning it
        """
        try:
            num_cols = ['MonthlyCharges', 'TotalCharges']
            
            cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
            
            num_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='median')),
                                             #('scaler', MinMaxScaler()),
                                             ('scaler', StandardScaler() )])
            
            cat_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),
                                             ('encoder', OneHotEncoder() )])
            
            logging.info(f"Numerical columns: {num_cols}")
            logging.info(f"Categorical columns: {cat_cols}")
            
            preprocessor = ColumnTransformer([('num_pipeline', num_pipeline, num_cols),
                                              ('cat_pipeline', cat_pipeline, cat_cols)])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
            
            
    def initiate_data_transformation(self, train_path, test_path):
        
        """
        Function to initiate the data transformation
        """ 
        
        try: 
            
            logging.info("Data transformation process has been started.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Imported train and test data.")
            
            
            
            logging.info("Obtaining preprocessing object.")
            # calling the preprocessor from get_data_transformer_object 
            preprocessing_obj = self.get_data_transformer_object()
            
            
            target_column_name = 'Churn'
            num_cols = ['MonthlyCharges', 'TotalCharges']
            
            
            
            # train set
            X_train = train_df.drop(columns=target_column_name, axis=1)
            y_train = train_df[target_column_name]
            
            # test set
            X_test = test_df.drop(columns=target_column_name, axis=1)
            y_test = test_df[target_column_name]
            
    
            
            logging.info("Preprocessing train and test sets has been started.")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)
            
            
            
            # handling imbalance 
            logging.info("Handling imbalance.")
            #smote_enn = SMOTEENN(random_state=42)
            kmeans_smote = KMeansSMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42)
            X_train_resampled, y_train_resampled = kmeans_smote.fit_resample(X_train_arr, y_train)
            
            
            #X_train_df = pd.DataFrame(X_train_resampled, columns=X_train.columns)
            #y_train_df = pd.DataFrame(y_train_resampled, columns=y_train.columns)
            
            
            logging.info(f"Shape of X_train before SMOTEENN:{X_train_arr.shape}")
            logging.info(f"Shape of X_train after SMOTEENN: {X_train_resampled.shape}")
            logging.info(f"Shape of y_train before SMOTEEN: {y_train.shape}")
            logging.info(f"Shape of y_train after SMOTEENN: {y_train_resampled.shape}")
            
            logging.info(f"Distribution of Churn before SMOTEENN: {y_train.value_counts()}")
            logging.info(f"Distribution of Churn after SMOTEENN: {y_train_resampled.value_counts()}")
            
            logging.info("Imbalance has been handled.")


        
            # concatenate the transformed train & test_arr to a single numpy array 
            # np.c_ function to concatenate column wise                                                                               
            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            
            
            
            # saving the preprocessing object 
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj)
            
            

            logging.info("Preprocessing object has been saved.")
            logging.info("Data transformation process has been completed.")
            
            return(train_arr, 
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)
        
            
        except Exception as e:
            raise CustomException(e,sys)
            
                                             
                                             
                                    
            
            
            