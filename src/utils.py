import os
import sys

from src.exception_handling import CustomException
from src.logger import logging

import dill
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj): 
    """
    Save a python object to a file using dill
    
    file_path : The path where the object should be saved
    obj : The python object to be saved
    """
    
    try:
        
        # get the directory name
        dir_path = os.path.dirname(file_path)
        
        # create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # open the file present in 'file_path' in 'write-binary' mode and dump the file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
            
    except Exception as e:
        raise CustomException(e, sys)
        
        
        
        
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    
    try:
        
        report = {} 
        logging.info("Starting model evaluation process.")
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, param, cv=5)
            logging.info("Starting gridsearch for {model}")
            gs.fit(X_train, y_train)
            logging.info("Completed gridsearch for {model}")
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        logging.info("Model evalutation process has beeen completed.")
        return report
    
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
    
    
    
def load_object(file_path):
    
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
        
    except Exception as e:
        raise CustomException(e, sys)
            
            
            
            
            