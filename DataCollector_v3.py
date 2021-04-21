import os
import pandas as pd
import sys
import logging


## redirect output:
sys.stdout = open('output_DataCollector.dat', 'w')


## modules
sys.path.insert(0, './_modules')
from path_definitions import control_dir, att_dir
from orchestra_module import mailId_query, TheGreatFilter, TheCollector, Orchestrator
from logging_module import getlogger
import random
import numpy as np
import time
import schedule



## log
logging = getlogger()


## current dir
cwd = os.getcwd()
print(cwd)


## specifying the path to attachments:
attachments_path = att_dir()
logging.info('Creating/verifying attachments folder: {}'.format(attachments_path))


## specifying the path to collect control:
control_path = control_dir()
logging.info('Creating/verifying control folder: {}'.format(control_path))



## getting the process id
pid = os.getpid()
with open(os.path.join(control_path, 'pid_DataCollector.dat'), 'w') as f:
    f.write(str(pid)+'\n')
f.close()
logging.info('PID saved: {}'.format(pid))





## scheduler
##############################
def job():
    '''
    The job for the scheduler
    '''

    ## choose randomly between a "full" or daily lookout
    query_type = random.choice(['full', 'day', 'day', 'day', 'day'])
    ## tests
    #query_type = random.choice(['day'])
    df, df_attachments, df_to = Orchestrator(control_path, attachments_path, new_query = True, query_type  = query_type)

    return
##############################

time_interval = 10
schedule.every(time_interval).minutes.do(job)

while True:

    schedule.run_pending()
    time.sleep(1)
