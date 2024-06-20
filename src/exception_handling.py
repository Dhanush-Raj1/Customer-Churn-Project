import sys
from src.logger import logging



def error_message_detail(error, error_detail:sys):           # sys module to extract error details 
    """custom exception handling 

    Args:
        error (_type_): _description_
        error_detail (sys): _description_
    """                                     
                                             
    _, _, exc_tb = error_detail.exc_info()           # returns a tuple of 3 values(type, value, traceback) we are discarding the first two using _, _, 
                                                     # exc_tb - exception trace back object  
    
    file_name = exc_tb.tb_frame.f_code.co_filename   # tb_frame - Frames information of trace back object 'exc_tb' i.e code execution, local variables
                                                     # f_code - code information of Frame object 'tb_frame' i.e line number, byte code
                                                     # co_filename - File name information of code object 'f_code' 
                                                     
    error_message = "Error occured in python script [{0}] line number [{1}] error message [{2}]".format(
                                                   file_name, exc_tb.tb_lineno, str(error)  )
    
    return error_message 
    
    
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)             # initializing the base Exception with error_message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)       # create and store the detailed error message
    
    def __str__(self):                    # '__str__' - representation method - string 
        return self.error_message
    
