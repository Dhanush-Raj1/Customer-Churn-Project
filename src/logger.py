import logging
import os
from datetime import datetime


# create log file name                                         
logs_file_name = datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + ".log"          # ".log" adds .log extension to the date,time resulting in a log file

# create "logs" folder
os.makedirs("logs", exist_ok=True)     # 'exist_ok=True' if the "logs" folder already exits don't raise an error

# path for storing log files(in current directory, inside the logs folder, with the log file name)
logs_file_path = os.path.join(os.getcwd(), "logs", logs_file_name) 

# configure logging
logging.basicConfig(filename=logs_file_path,
                    # format of logging message 
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    # log those messages with severity level "INFO" and above
                    level=logging.INFO,)


# test logging setup 
if __name__ == "__main__":
    logging.info("Logging has started")
    