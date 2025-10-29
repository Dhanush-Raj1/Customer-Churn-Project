import os
import sys
import dill
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception_handling import CustomException
from src.logger import logging


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
        
        
    
def load_object(file_path):
    """
    Loads a saved object from the respective file path
    object: pkl file (saved using dill.dump)
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        
        
        
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates a model using gridsearchcv
    Returns: report containing model names and its train, test accuracy
    """
    try:
        report = {} 
        logging.info("Starting model evaluation process.")
        
        for model_name, model in models.items():
            param = params[model_name]
            
            gs = GridSearchCV(model, param, cv=5)
            logging.info(f"Starting gridsearch for {model_name}")
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            logging.info(f"Train and test accuracy score for {model_name}: {train_model_score, test_model_score}")
            logging.info(f"Completed gridsearch for {model_name}")
            
            report[model_name] = {'train_accuracy': train_model_score, 
                                  'test_accuracy': test_model_score }
            
        logging.info("Model evalutation process has beeen completed.")
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
    
    

            
            