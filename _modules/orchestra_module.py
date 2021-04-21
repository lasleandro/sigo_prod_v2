import os
import pandas as pd
import sys
import logging
from datetime import datetime
from datetime import timedelta
import random
import numpy as np
import time
import schedule
from sqlalchemy import create_engine

import mysql.connector
from mysql.connector import Error


## modules
sys.path.insert(0, './_modules')
from scanner_module import scanner
from regex_module import FindEmail, FindNumbersBraket, FindLawsuit
from general_functions import DateConv, datetag_func
from Gmail_module import GetAttachments_v2, GetMessage_v2, ListMessagesMatchingQuery, GetBody,\
                         FDS, PrintMail, GetCredentials, MailData, GetBody_v2
from logging_module import getlogger
from db_functions import TheDB

from path_definitions import control_dir, att_dir



## log
logging = getlogger()



## bots BD connections
conn_dict = {'host':'localhost', 'database': 'brspcqadev',
             'user':'root', 'password':'quito2020'}

cqdb_prod = TheDB(conn_dict)





######################################################
def mailId_query(control_path, new_query = True, query_type = 'day'):
    '''
    Query for new emails starting from the current day.
    Input: i) the path to the directory containing mid_base.csv;
           ii) new_query = True or False (If False, only looks at the mail list stored locally)
           iii) query_type = 'day' or 'full' (date range starting from 2018)
    Output: i) a list with the history all emails Id, with the ones of the current day appended.
    Files: i) creates/updates mid_base.csv
    '''

    ## path to the csv where all mail ids are stored
    file_path = os.path.join(control_path, 'mid_base.csv')


    if new_query == True:

        ## query:
        ## yyyy-mm-dd format or yyyy/mm/dd
        if query_type == 'day':
            start_time2 = (datetime.today()).strftime('%Y-%m-%d')
        elif query_type == 'full':
            start_time2 = '2018-01-01'
        else:
            logging.warning('You need specify a "day" or "full" query. Following with a full query.')
            start_time2 = '2018-01-01'

        end_time2 = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

        mail_list = None
        while mail_list is None:
            try:
                mail_list = ListMessagesMatchingQuery(query='after:{} before:{}'.format(start_time2, end_time2))
                logging.info('Mail counting from {} to {}: {}'.format(start_time2, end_time2, len(mail_list)))
            except Exception as e:
                logging.error("Exception occurred", exc_info=True)
                sleeptime = 600
                logging.warning("Sleeping for {} seconds.".format(sleeptime))
                time.sleep(sleeptime)
                pass


        ## saving to a dataframe
        df_mid = pd.DataFrame(mail_list)
        df_mid['after'] = start_time2
        df_mid['before'] = end_time2


        ## appending or creating the csv
        if os.path.exists(file_path) == True:
            with open(file_path, 'a') as f:
                df_mid.to_csv(f, header=False, sep ='\t', encoding='utf-8', index=False)
        else:
            df_mid.to_csv(file_path, sep ='\t', encoding='utf-8',index=False)




    ## reading the csv from directory
    df_mid = pd.read_csv(file_path, sep = '\t', encoding = 'utf-8')

    ## dropping eventual duplicates
    df_mid = df_mid.drop_duplicates(subset = ['id']).reset_index(drop = True)

    ## rewriting the dataframe to the directory
    df_mid.to_csv(file_path, sep = '\t', encoding = 'utf-8', index = False)

    ## extracting the mail list
    all_mid_list = list(set(df_mid['id'].tolist()))
    print('All mails counting: {}'.format(len(all_mid_list)))



    return all_mid_list
###################################################################







######################################################
def TheGreatFilter(all_mid_list, control_path):
    '''
    Given a list of mail ids, finds the ones that didn't get collected yet.
    Input: a list of all email ids
    Output: a list with emails to be collected
    Files: reads mail_data.csv
    '''

    ## checking if there is mail to be collected
    df_db = cqdb_prod.GetTable('Screening', 'main_id, mail_id')
    collected_list = list(set(df_db['mail_id'].tolist())) 

    ## excluding the mails already collected
    mail_list = list(set(all_mid_list) - set(collected_list))
    mail_counter = len(mail_list)
    #print(mail_counter)


    ## for long lists, it is more safe to split them into small pieces to collect and store them
    nparts = 100
    if mail_counter >= nparts:
        splitted_mail_list = np.array_split(mail_list, nparts)
    else:
        nparts = 1
        splitted_mail_list = [mail_list.copy()]



    return splitted_mail_list, mail_counter
##############################################################################



##############################################################################
def TheCollector(splitted_mail_list, control_path, attachments_path):
    '''
    Collects emails and attachments from a given list of mail ids.
    Input: a list of lists of email ids (a list of ids splitted into chuncks)
    Output: i) df = dataframe containing data from emails
            ii) df_attachments = df containing the files attached to a given mail
            iii) df_to = df containing all destinataries of a given mail
    Files: i) creates/updates mail_data.csv
           ii) creates/updates att_data.csv
           iii) creates/updates to_data.csv

    '''

    nparts = len(splitted_mail_list)
    counter = 0
    time_list = []
    global_start_time = time.time()
    for subset_mail_list in splitted_mail_list:
        start_time = time.time()
        logging.info('>>> Subset {}'.format(counter+1))

        try:
            df, df_attachments, df_to = FDS(subset_mail_list, attachments_path, control_path)
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)



            continue



        ## tests


        ## converting time to MySQL format
        df['mail_date_received'] = pd.to_datetime(df['mail_date_received_full'])
        df['mail_hms_received'] = pd.to_datetime(df['mail_date_received_full'])
        df['mail_body'] = df['mail_body'].str.encode('utf-8')
        df['has_attachment'] = df['has_attachment'].astype(int)

        if 'mail_date_received_full' in df.columns:
            df = df.drop('mail_date_received_full', axis = 1)





        ## create sqlalchemy engine
        engine = create_engine("mysql://{user}:{pw}@localhost/{db}?charset=utf8mb4"
                               .format(user="root",
                                       pw="quito2020",
                                       db='brspcqadev'), encoding="utf8")
        logging.info('Loading MySQL database.')

        ## Insert whole DataFrame into MySQL
        try:
            df.to_sql('Screening', con = engine, if_exists = 'append', index = False, chunksize = 1000)
            logging.info('Data written to MySQL database.')
            engine.dispose()
        except Exception as e:
            logging.error(e, exc_info=True)
            engine.dispose()
        




        att_path = os.path.join(control_path, 'att_data.csv')
        destinatary_path = os.path.join(control_path, 'to_data.csv')


        ## saving data about attachments
        df_screening = cqdb_prod.GetTable('Screening', 'main_id, mail_id')
        df_att_final = df_screening.merge(df_attachments, on = 'mail_id', how = 'right').dropna()
        df_att_final['main_id'] = df_att_final['main_id'].astype(int)
        df_att_final = df_att_final.drop('mail_id', axis = 1)

        ## inserting into DB:
        result = cqdb_prod.InsertDB(df_att_final, 'Attachment', key_column = 'doc_link', return_table = False)
        logging.info('Attachments persisted into DB!')





        ## saving destinataries
        if os.path.exists(destinatary_path) == True:
            with open(destinatary_path, 'a') as f:
                df_to.to_csv(f, header = False, sep ='\t', encoding = 'utf-8', index=False)
        else:
            df_to.to_csv(destinatary_path, sep ='\t', encoding = 'utf-8', index=False)
        logging.info('Destinataries data saved!')




        #print(' ')
        #print(' ')
        counter += 1
        time.sleep(5)

        dt = time.time() - start_time
        time_list.append(dt)
        forecasting = np.mean(time_list)*nparts

        #print('Total time elapsed: {}'.format(time.time() - global_start_time))
        #print('Step time: {}'.format(dt))
        #print('Forecasting: {}'.format(forecasting))
        #print(' ')
        #print('>>>>>>>>>>>>>>')
        #print(' ')
        #print(' ')
        logging.info('Total time elapsed: {}'.format(time.time() - global_start_time))
        logging.info('Step time: {}'.format(dt))
        logging.info('Forecasting: {}'.format(forecasting))

    return df, df_attachments, df_to
######################################################


######################################################
def Orchestrator(control_path, attachments_path, new_query, query_type):
    '''
    Orchestrates all pipeline of mail extraction.
    Input: the control path, attachments path and the type of query (full or day)
    Output: a log csv
    Files: i) inserts new mail data into the MySQL screening_table
           ii) generates a log csv containing the following infos: 'date',
           'step_time', 'new_mail_counter', 'total_mail_counter' and 'query_type'.
    '''

    global_start_time = time.time()
    ## looking for new emails:
    logging.info('Looking for new emails')
    start_time = time.time()
    all_mid_list = mailId_query(control_path, new_query = new_query, query_type = query_type)
    dt = time.time() - start_time
    logging.info('Time spent: {}'.format(dt))


    ## filtering new email:
    logging.info('Filtering new emails')
    start_time = time.time()
    splitted_mail_list, mail_counter = TheGreatFilter(all_mid_list, control_path)
    dt = time.time() - start_time
    logging.info('Time spent: {}'.format(dt))

    ## Getting mail data and attachments
    start_time = time.time()
    logging.info('Getting mail data and attachments')
    if mail_counter > 0:

        df, df_attachments, df_to = TheCollector(splitted_mail_list, control_path, attachments_path)
  

    else:
        ## defining empty dfs for return purposes:
        df = pd.DataFrame()
        df_attachments = pd.DataFrame()
        df_to = pd.DataFrame()
        logging.info('No new mail found.')



    dt = time.time() - start_time
    dt_final = time.time() - global_start_time
    logging.info('Time spent: {}'.format(dt))
    logging.info('Total time spent: {}'.format(dt_final))


    ## for log purposes:
    df_log = pd.DataFrame(columns = ['date', 'step_time', 'new_mail_counter',
                                     'total_mail_counter', 'query_type'])
    df_log.at[0, 'date'] = str(datetime.today())
    df_log.at[0, 'step_time'] = dt_final
    df_log.at[0, 'new_mail_counter'] = mail_counter
    df_log.at[0, 'total_mail_counter'] = len(all_mid_list)
    df_log.at[0, 'query_type'] = query_type

    log_path = os.path.join(control_path, 'log.csv')
    if os.path.exists(log_path) == True:
        with open(log_path, 'a') as f:
            df_log.to_csv(f, header = False, sep ='\t', encoding = 'utf-8', index=False)
    else:
        df_log.to_csv(log_path, sep ='\t', encoding = 'utf-8', index=False)
    logging.info('Log saved to .csv')



    return df, df_attachments, df_to
######################################################
