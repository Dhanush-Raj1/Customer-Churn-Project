import os
import sys
from src.exception_handling import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer1 import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



# Data ingestion configuration class
@dataclass
class DataIngestionConfig:
    
    # paths for train, test sets
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    


# Data ingestion class 
class DataIngestion:
    """
    Import the cleaned data 
    Performs train, test split
    Saves it in artifacts folder 
    """ 
    
    # initiate the configuration object 
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        """
        Data ingestion function
        Returns the path of train and test data
        """
        try: 
            logging.info("Data ingestion process has been started.")
            df = pd.read_csv('artifacts/data_cleaned.csv')
            logging.info("Cleaned data has been loaded.")
            
            
            logging.info("Train and test split has been initiated.")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Shape of train set: {train_set.shape}")
            logging.info(f"Shape of test set: {test_set.shape}")
            
            
            logging.info("Data ingestion process has been completed.")
            
            
            return(self.ingestion_config.train_data_path, 
                   self.ingestion_config.test_data_path)
        
        
        except Exception as e:
            raise CustomException(e, sys)
    


#if __name__ =="__main__":
    
    # DataIngestion object
    #data_ingestion = DataIngestion()
    # initiate_data_ingestion function    
    #train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    
    
    # DataTransformation object 
    #data_transformation = DataTransformation()
    # initiate_data_transformation function 
    #train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    
    # ModelTrainer object
    #model_trainer = ModelTrainer()
    #model_trainer.initiate_model_trainer(train_arr, test_arr)
    
    