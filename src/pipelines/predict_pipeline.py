import os 
import sys 
from src.exception_handling import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd


class Predict:
    def __init__(self):
        pass  
    
    def predict_data(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/ preprocessor.pkl"
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info(f"Features before transformation \n{features}")
            data_transformed = preprocessor.transform(features)
            logging.info(f"Features after transformation \n{data_transformed}")
            
            predicted = model.predict(data_transformed)
            
            return predicted
        
        except Exception as e:
            raise CustomException(e, sys)
    

class NewData:
    """
    Returns the newdata as a dataframe
    """
    
    def __init__( self, gender: str,
                  SeniorCitizen: str, Partner: str, 
                  Dependents: str, tenure: float, 
                  PhoneService: str, MultipleLines: str, 
                  InternetService: str, OnlineSecurity: str, 
                  OnlineBackup: str, DeviceProtection: str, 
                  TechSupport: str, StreamingTV: str, 
                  StreamingMovies: str, Contract: str, 
                  PaperlessBilling: str, PaymentMethod: str, 
                  MonthlyCharges: float, TotalCharges: float):
        
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents 
        self.tenure = tenure 
        self.PhoneService = PhoneService 
        self.MultipleLines = MultipleLines 
        self.InternetService = InternetService 
        self.OnlineSecurity = OnlineSecurity 
        self.OnlineBackup = OnlineBackup 
        self.DeviceProtection = DeviceProtection 
        self.TechSupport = TechSupport 
        self.StreamingTV = StreamingTV 
        self.StreamingMovies = StreamingMovies 
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling 
        self.PaymentMethod = PaymentMethod 
        self.MonthlyCharges = MonthlyCharges 
        self.TotalCharges = TotalCharges
        
    
    def get_data_as_dataframe(self):
        
        try:
            new_data_input = { "gender": self.gender,
                               "SeniorCitizen": self.SeniorCitizen, 
                               "Partner": self.Partner, 
                               "Dependents": self.Dependents, 
                               "tenure": self.tenure, 
                               "PhoneService": self.PhoneService, 
                               "MultipleLines": self.MultipleLines, 
                               "InternetService": self.InternetService, 
                               "OnlineSecurity": self.OnlineSecurity, 
                               "OnlineBackup": self.OnlineBackup, 
                               "DeviceProtection": self.DeviceProtection, 
                               "TechSupport": self.TechSupport, 
                               "StreamingTV": self.StreamingTV, 
                               "StreamingMovies": self.StreamingMovies, 
                               "Contract": self.Contract, 
                               "PaperlessBilling": self.PaperlessBilling, 
                               "PaymentMethod": self.PaymentMethod, 
                               "MonthlyCharges": self.MonthlyCharges, 
                               "TotalCharges": self.TotalCharges  } 
            
            return pd.DataFrame(new_data_input)
        
        except Exception as e:
            raise CustomException(e, sys)
