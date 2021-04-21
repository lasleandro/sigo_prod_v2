import sys
import os
import re
import logging
from logging.handlers import TimedRotatingFileHandler
## modules
sys.path.insert(0, './_modules')
from path_definitions import control_dir

folder = os.path.dirname(__file__)
path_log = os.path.join(os.path.dirname(folder), '_collect_control')
if os.path.isdir(path_log) == False:
    os.mkdir(path_log)


def StartLogger():

    # logger
    logname = 'logging_app.log'
    file_path = os.path.join(path_log, logname)

    logging.getLogger().handlers.clear()
    logger = logging.getLogger('main_logger')
    #logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO) 


    fh = logging.FileHandler(file_path)
    #fh.setLevel(logging.DEBUG) # ensure all messages are logged to file
    fh.setLevel(logging.INFO) # ensure all messages are logged to file

    # create a formatter and set the formatter for the handler.
    #frmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    frmt = logging.Formatter('%(asctime)s - %(levelname)s -')
    fh.setFormatter(frmt)
    logger.addHandler(fh)


    handler = TimedRotatingFileHandler(file_path, when="midnight", interval=1)
    # add a suffix which you want
    handler.suffix = "%Y%m%d"
    #need to change the extMatch variable to match the suffix for it
    handler.extMatch = re.compile(r"^\d{8}$") 

    logger.addHandler(handler)


    logger.propagate = False


    return logger


logger = StartLogger()