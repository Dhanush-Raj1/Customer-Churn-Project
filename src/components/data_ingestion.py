import os
import sys
from src.exception_handling import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



# Data ingestion configuration class
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    

# Data ingestion class 
class DataIngestion:
    # initiate the configuration object 
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion has been started")
        
        try: 
            df = pd.read_csv('data/data.csv')
            logging.info("Data has been exported as a dataframe.")
            
            # making the 'artifacts' directory, extracting the directory name if it already exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train and test split has been initiated.")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion has been completed.")
            
            return(self.ingestion_config.train_data_path, 
                   self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)
    

if __name__ =="__main__":
    ingestion_obj = DataIngestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    