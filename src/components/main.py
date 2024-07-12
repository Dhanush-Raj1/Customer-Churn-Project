import os 
import sys

from src.exception_handling import CustomException
from src.logger import logging

from src.components.data_cleaning import DataCleaning
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer1 import ModelTrainer

# cleaning the raw data
data_cleaning = DataCleaning()
data_cleaning.initiate_data_cleaning()


# performing data ingestion
data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion()


# performing data transformation
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)


# model training 
model_trainer = ModelTrainer()
model_trainer.initiate_model_trainer(train_arr, test_arr)


