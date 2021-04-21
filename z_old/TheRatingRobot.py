import os
import pandas as pd
import sys
import numpy as np
import time
import schedule

sys.path.insert(0, './_modules')
from db_functions import TheDB
#from logging_module import getlogger
from logging_module_v2 import StartLogger



logging = StartLogger()


#############################################################
def EvalMean(cqdb):

    df_emissary = cqdb.GetTable('Emissary')
    df_rating = cqdb.GetTable('Rating')

    df_rating['rating'] = df_rating['rating'].astype(float)
    df_mean = df_rating.groupby('emissary_id')['rating'].apply(list).apply(np.mean).to_frame().reset_index()
    df_mean.columns = ['emissary_id', 'avg']
    df_mean = df_mean.dropna()
    
    
    
    return df_mean
############################################################



## bots BD connections
conn_dict = {'host':'localhost', 'database': 'brspcqadev',
             'user':'root', 'password':'quito2020'}

cqdb = TheDB(conn_dict)




## scheduler
##############################
def job():
    '''
    The job for the scheduler
    '''

    try:
        df_mean = EvalMean(cqdb)
        logging.info('Function EvalMean success: Average rating evaluated.')
    except Exception as e:
        logging.error('Function EvalMean error: {}'.format(e))
        
        
    try:
        df_queries = cqdb.MainUpdate(df_mean, 'Emissary')
        logging.info('Method MainUpdate success: average rating updated in the DB.')
    except Exception as e:
        logging.error('Method MainUpdate error: {}'.format(e))
        
        
    return
##############################

time_interval = 1
schedule.every(time_interval).minutes.do(job)

while True:

    schedule.run_pending()
    time.sleep(1)