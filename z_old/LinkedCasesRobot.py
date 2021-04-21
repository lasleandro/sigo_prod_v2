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
from datetime import datetime



logging = StartLogger()


def IKYDLS():
    
    ## bots BD connections
    conn_dict = {'host':'localhost', 'database': 'brspcqadev',
                 'user':'root', 'password':'quito2020'}

    cqdb_prod = TheDB(conn_dict)    
    
    
    def CleanList(input_list):
        final_list = [item for item in input_list if item != '' and item != ' ']
        return final_list

    def SetList(input_list):
        final_list = list(set(input_list))
        return final_list
    
    ### working the Screening table
    
    ## getting the table
    df_full = cqdb_prod.GetTable('Screening')

    ## converting to time object
    df_full['mail_hms_received'] = pd.to_datetime(df_full['mail_hms_received']).dt.time

    ## converting to datetime object
    df_full['mail_date_received'] = pd.to_datetime(df_full['mail_date_received'])


    ## unifying date and time columns into one single datetime object:
    df_full['mail_datetime_received'] = pd.to_datetime(df_full['mail_date_received'].apply(str) +\
                                                  ' ' + df_full['mail_hms_received'].apply(str))

    df_full['mail_datetime_received'] = pd.to_datetime(df_full['mail_datetime_received'])

    df_full = df_full.sort_values('mail_datetime_received', ascending = False).reset_index(drop = True)   
    
    
    
    ## case view
    df_case_view = df_full.groupby('case_id')[['main_id', 'external_ref_number', 'thread_id']].agg(list).reset_index()
    df_case_view = df_case_view[(df_case_view['case_id'] != '') & (df_case_view['case_id'] != ' ')]
    df_case_view.columns = ['case_id', 'main_id_list', 'external_ref_number_list', 'thread_id_list']
    df_case_view['main_id_list'] = df_case_view['main_id_list'].apply(CleanList).apply(SetList)
    df_case_view['external_ref_number_list'] = df_case_view['external_ref_number_list'].apply(CleanList).apply(SetList)
    df_case_view['thread_id_list'] = df_case_view['thread_id_list'].apply(CleanList).apply(SetList)
    
    
    
    ## getting case_id - main_id relationship:
    df_case_main = df_case_view.set_index(['case_id'])['main_id_list'].apply(pd.Series)\
                                             .stack()\
                                             .to_frame().reset_index(level=1, drop=True).reset_index()
    df_case_main.columns = ['case_id', 'main_id']
    df_case_main['main_id'] = df_case_main['main_id'].astype(int)
    
    
    
    ## getting the case_id - external_ref_number relationship
    df_case_ext = df_case_view.set_index(['case_id'])['external_ref_number_list'].apply(pd.Series)\
                                         .stack()\
                                         .to_frame().reset_index(level=1, drop=True).reset_index()
    df_case_ext.columns = ['case_id', 'external_ref_number_list']
    
    
    
    ## case_id - thread_id relationship
    df_case_thread = df_case_view.set_index(['case_id'])['thread_id_list'].apply(pd.Series)\
                                             .stack()\
                                             .to_frame().reset_index(level=1, drop=True).reset_index()
    df_case_thread.columns = ['case_id', 'thread_id']    
    
    
    
    ## external ref number view
    df_ex_view = df_full.groupby('external_ref_number')[['main_id', 'case_id', 'thread_id']].agg(list).reset_index()
    df_ex_view = df_ex_view[(df_ex_view['external_ref_number'] != '') & (df_ex_view['external_ref_number'] != ' ')]
    df_ex_view.columns = ['external_ref_number', 'main_id_list', 'case_id_list', 'thread_id_list']
    df_ex_view['main_id_list'] = df_ex_view['main_id_list'].apply(CleanList).apply(SetList)
    df_ex_view['case_id_list'] = df_ex_view['case_id_list'].apply(CleanList).apply(SetList)
    df_ex_view['thread_id_list'] = df_ex_view['thread_id_list'].apply(CleanList).apply(SetList) 
    
    
    ## external_ref_number - main_id relation
    df_ex_main = df_ex_view.set_index(['external_ref_number'])['main_id_list'].apply(pd.Series)\
                                             .stack()\
                                             .to_frame().reset_index(level=1, drop=True).reset_index()
    df_ex_main.columns = ['external_ref_number', 'main_id']
    df_ex_main['main_id'] = df_ex_main['main_id'].astype(int)   
    
    
    ## external_ref_number - case_id relation
    df_ex_case = df_ex_view.set_index(['external_ref_number'])['case_id_list'].apply(pd.Series)\
                                             .stack()\
                                             .to_frame().reset_index(level=1, drop=True).reset_index()
    df_ex_case.columns = ['external_ref_number', 'case_id']    
    
    
    ## external_ref_number - thread_id relation
    df_ex_thread = df_ex_view.set_index(['external_ref_number'])['thread_id_list'].apply(pd.Series)\
                                             .stack()\
                                             .to_frame().reset_index(level=1, drop=True).reset_index()
    df_ex_thread.columns = ['external_ref_number', 'thread_id']    
    
    
    ## trix_id - external_ref_number relation
    df_trix = cqdb_prod.GetTable('Trix')
    df_trix_ext = df_trix.groupby('external_ref_number')[['case_id', 'line_id']].agg(list).reset_index()
    df_trix_ext = df_trix_ext.rename(columns = {'line_id': 'line_id_list'})
    df_trix_ext = df_trix_ext.loc[:, ['external_ref_number', 'line_id_list']]    
    
    
    
    ## trix_id - external_ref_number relation
    df_trix_ext2 = df_trix_ext.set_index(['external_ref_number'])['line_id_list'].apply(pd.Series)\
                                             .stack()\
                                             .to_frame().reset_index(level=1, drop=True).reset_index()
    df_trix_ext2.columns = ['external_ref_number', 'line_id']
    df_trix_ext2['line_id'] = df_trix_ext2['line_id'].astype(int)
    
    
    ## trix_id - case_id relation
    df_trix_case = df_trix.groupby('case_id')[['external_ref_number', 'line_id']].agg(list).reset_index()
    df_trix_case = df_trix_case.rename(columns = {'line_id': 'line_id_list'})
    df_trix_case = df_trix_case.loc[:, ['case_id', 'line_id_list']]    
    
    
    df_trix_case2 = df_trix_case.set_index(['case_id'])['line_id_list'].apply(pd.Series)\
                                             .stack()\
                                             .to_frame().reset_index(level=1, drop=True).reset_index()
    df_trix_case2.columns = ['case_id', 'line_id']
    df_trix_case2['line_id'] = df_trix_case2['line_id'].astype(int)
    
    
    result_dict = {'case_main': df_case_main, 'case_ext': df_case_ext, 'case_thread': df_case_thread,\
                  'ext_main':df_ex_main, 'ext_case': df_ex_case, 'ext_thread':df_ex_thread, \
                  'trix_ext': df_trix_ext2, 'trix_case': df_trix_case2}
    
    
    
    
    ## Writing data to DB:
    for relation, df in result_dict.items():
        try:
            #print(relation)
            results_db = cqdb_prod.InsertDB(df, 'relation_' + relation, key_column = '' ,\
                          return_table = False, append_table = False)


            #print(' ')
            #print(' ')
        except Exception as e:
            #print('Some error ocurred!')
            logging.error('Linked Cases Robot error: {}'.format(e))
            pass    
    
    
    
    
    return result_dict
######################################################################    




## scheduler
##############################
def job():
    '''
    The job for the scheduler
    '''

    try:
        result_dict = IKYDLS()
        logging.info('Linked Cases Robot success: relations saved into DB.')
    except Exception as e:
        logging.error('Linked Cases Robot error: {}'.format(e))
        
       
    return
##############################

time_interval = 5
schedule.every(time_interval).minutes.do(job)

while True:

    schedule.run_pending()
    time.sleep(1)