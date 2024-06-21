import os
import sys
from src.exception_handling import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestinConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_path = DataIngestinConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion has been started")
        
        try: 
            df = pd.read_csv('data/data_eda.csv')
            logging.info("Data has been exported as a dataframe.")
            
            # making the 'artifacts' directory, extracting the directory name if it already exists
            os.makedirs(os.path.dirname(self.ingestion_path.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_path.raw_data_path, index=False, header=True)
            
            logging.info("Train and test split has been initiated.")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_path.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_path.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion has been completed.")
            
            return(self.ingestion_path.train_data_path, 
                   self.ingestion_path.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)
    
    