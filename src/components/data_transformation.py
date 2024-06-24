import sys
import os
from src.exception_handling import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass



# Data transformation configuration class
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    

# Data transformation class     
class DataTransformation:    
    
    # Initiate the configuration object
    def __init__(self):
        self.config = DataTransformationConfig()
    

    def get_data_transformer_object(self):
        
        """
        Building preprocessor object and returning it
        """
        
        try:
            num_cols = ['MonthlyCharges', 'TotalCharges']
            
            cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling']
            
            num_pipeline = Pipeline(steps = [('scaler', Normalizer())])
            
            cat_pipeline = Pipeline(steps = [('encoder', OneHotEncoder())])
            
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
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Imported train and test data.")
            logging.info("Obtaining preprocessing object.")
            
            # calling the preprocessor from get_data_transformer_object 
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'Churn'
            num_cols = ['MonthlyCharges', 'TotalCharges']
            
            input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Preprocessing train and test set has been started.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # concatenate the transformed input and target features to a single numpy array 
            # np.c_ function to concatenate column wise                                                                               
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # saving the preprocessing object 
            save_object(file_path = self.config.preprocessor_obj_file_path,
                        obj = preprocessing_obj)
            
            logging.info("Preprocessing object has been saved.")
            
            return(train_arr, 
                   test_arr,
                   self.config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)
            
                                             
                                             
                                    
            
            
            