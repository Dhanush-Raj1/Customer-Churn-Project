import os
import sys

from src.exception_handling import CustomException

import dill


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