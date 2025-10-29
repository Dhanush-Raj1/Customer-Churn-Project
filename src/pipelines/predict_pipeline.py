import os 
import sys 
from src.exception_handling import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class NewData:
    """
    Returns the new data as a dataframe
    """
    
    def __init__( self, 
                  gender: str,
                  SeniorCitizen: str, 
                  Partner: str, 
                  Dependents: str, 
                  tenure: str, 
                  PhoneService: str, 
                  MultipleLines: str, 
                  InternetService: str, 
                  OnlineSecurity: str, 
                  OnlineBackup: str, 
                  DeviceProtection: str, 
                  TechSupport: str, 
                  StreamingTV: str, 
                  StreamingMovies: str, 
                  Contract: str, 
                  PaperlessBilling: str, 
                  PaymentMethod: str, 
                  MonthlyCharges: float, 
                  TotalCharges: float):
        
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
            new_data_input = { "gender": [self.gender],
                               "SeniorCitizen": [self.SeniorCitizen], 
                               "Partner": [self.Partner], 
                               "Dependents": [self.Dependents], 
                               "tenure": [self.tenure], 
                               "PhoneService": [self.PhoneService], 
                               "MultipleLines": [self.MultipleLines], 
                               "InternetService": [self.InternetService], 
                               "OnlineSecurity": [self.OnlineSecurity], 
                               "OnlineBackup": [self.OnlineBackup], 
                               "DeviceProtection": [self.DeviceProtection], 
                               "TechSupport": [self.TechSupport], 
                               "StreamingTV": [self.StreamingTV], 
                               "StreamingMovies": [self.StreamingMovies], 
                               "Contract": [self.Contract], 
                               "PaperlessBilling": [self.PaperlessBilling], 
                               "PaymentMethod": [self.PaymentMethod], 
                               "MonthlyCharges": [self.MonthlyCharges], 
                               "TotalCharges": [self.TotalCharges]  } 
            
            logging.info("Converting the data in to a DataFrame.")
            return pd.DataFrame(new_data_input)
        
        except Exception as e:
            raise CustomException(e, sys)





class Predict:
    """
    Make predictions and returns them using the saved model(pickle file)
    """
    def __init__(self):
        pass  
    
    
    def predict_data(self, df):
        try:
            
            # base_path = r"F:\\Data Science\\Projects\\3.Customer-Churn-Project"
            # model_path = r"F:\\Data Science\\Projects\\3.Customer-Churn-Project\\artifacts\\model.pkl"
            # preprocessor_path = os.path.join(base_path, "artifacts", "preprocessor.pkl")

            # works for both local and cloud environment (eks cluster)
            base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            # how it works locally, 
            # __file__ = (current file) F:\Data Science\Projects\3.Customer-Churn-Project\src\pipelines\predict_pipeline.py
            # os.path.dirname(__file__) = F:\Data Science\Projects\3.Customer-Churn-Project\src\pipelines
            # os.path.dirname(os.path.dirname(__file__)) = F:\Data Science\Projects\3.Customer-Churn-Project\src
            # os.path.dirname(os.path.dirname(os.path.dirname(__file__))) = F:\Data Science\Projects\3.Customer-Churn-Project ✅
            # Final path: F:\Data Science\Projects\3.Customer-Churn-Project\artifacts\model.pkl

            # how it works in cloud, 
            # __file__ = /app/src/pipelines/predict_pipeline.py
            # rest same logic 
            # Final path: /app/artifacts/model.pkl

            model_path = os.path.join(base_path, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(base_path, "artifacts", "preprocessor.pkl")
            
            logging.info("Attempting to load model object.")
            model = load_object(file_path=model_path)
            logging.info("Model has been loaded successfully.") 
            
            logging.info("Attempting to load preprocessor object.")
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Preprocessor has been loaded successfully.")

            logging.info(f"Data before transformation \n{df.head()}")
            df_transformed = preprocessor.transform(df)
            logging.info(f"Data after transformation \n{df_transformed}")
            
            logging.info("Predicting process has been started.")
            predicted = model.predict(df_transformed)
            logging.info("Prediction has been completed.")
            
            return predicted
        
        except Exception as e:
            raise CustomException(e, sys)
    

 