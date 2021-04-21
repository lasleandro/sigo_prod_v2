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


def getlogger():

    # logger
    logname = 'logging_app.log'
    logging.basicConfig(filename = logname, filemode = 'a', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger = logging.getLogger() 

    handler = TimedRotatingFileHandler(os.path.join(path_log, logname), when="midnight", interval=1)
    # add a suffix which you want
    handler.suffix = "%Y%m%d"
    #need to change the extMatch variable to match the suffix for it
    handler.extMatch = re.compile(r"^\d{8}$") 

    logger.addHandler(handler)


    return logger