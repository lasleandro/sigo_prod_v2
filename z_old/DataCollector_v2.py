import os
import pandas as pd
import sys
import logging


## redirect output:
sys.stdout = open('output_DataCollector.dat', 'w')


## modules
sys.path.insert(0, './_modules')
from QRCode_module import decode
from scanner_module import scanner
from regex_module import FindEmail, FindNumbersBraket, FindLawsuit
from general_functions import DateConv, datetag_func
from Gmail_module import GetAttachments_v2, GetMessage_v2, ListMessagesMatchingQuery, GetBody,\
                         FDS, PrintMail, GetCredentials, MailData, GetBody_v2
from datetime import datetime
from datetime import timedelta
import random
import numpy as np
import time
import schedule
from sqlalchemy import create_engine


import mysql.connector
from mysql.connector import Error



##date_tag
date_tag = datetag_func()

## log
logname = 'logging_output_DataCollector_{}.dat'.format(date_tag)
logging.basicConfig(filename = logname, filemode = 'a', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)






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
            print('You need specify a "day" or "full" query')
            print('Following with a full query.')
            start_time2 = '2018-01-01'

        end_time2 = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

        mail_list = None
        while mail_list is None:
            try:
                mail_list = ListMessagesMatchingQuery(query='after:{} before:{}'.format(start_time2, end_time2))
                print('Mail counting from {} to {}: {}'.format(start_time2, end_time2, len(mail_list)))
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
    collected_file = os.path.join(control_path, 'mail_data.csv')
    if os.path.exists(collected_file):
        df_collected = pd.read_csv(os.path.join(control_path, 'mail_data.csv'), sep = '\t', encoding = 'utf-8')
        collected_list = list(set(df_collected['mail_id'].tolist()))

    else:
        collected_list = []

    ## excluding the mails already collected
    mail_list = list(set(all_mid_list) - set(collected_list))
    mail_counter = len(mail_list)
    print(mail_counter)


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
        print('>>> Subset {}'.format(counter+1))

        try:
            df, df_attachments, df_to = FDS(subset_mail_list, attachments_path)
        except Exception as e:
            print('Error')
            logging.error("Exception occurred", exc_info=True)


        ## saving data
        file_path = os.path.join(control_path, 'mail_data.csv')
        att_path = os.path.join(control_path, 'att_data.csv')
        destinatary_path = os.path.join(control_path, 'to_data.csv')
        if os.path.exists(file_path) == True:
            with open(file_path, 'a') as f:
                df.to_csv(f, header = False, sep ='\t', encoding = 'utf-8', index=False)

            with open(att_path, 'a') as f:
                df_attachments.to_csv(f, header = False, sep ='\t', encoding = 'utf-8', index=False)

            with open(destinatary_path, 'a') as f:
                df_to.to_csv(f, header = False, sep ='\t', encoding = 'utf-8', index=False)

        else:
            df.to_csv(file_path, sep ='\t', encoding = 'utf-8',index=False)
            df_attachments.to_csv(att_path, sep ='\t', encoding = 'utf-8', index=False)
            df_to.to_csv(destinatary_path, sep ='\t', encoding = 'utf-8', index=False)


        print(' ')
        print(' ')
        counter += 1
        time.sleep(5)

        dt = time.time() - start_time
        time_list.append(dt)
        forecasting = np.mean(time_list)*nparts

        print('Total time elapsed: {}'.format(time.time() - global_start_time))
        print('Step time: {}'.format(dt))
        print('Forecasting: {}'.format(forecasting))
        print(' ')
        print('>>>>>>>>>>>>>>')
        print(' ')
        print(' ')

    return df, df_attachments, df_to
######################################################


######################################################
def Orchestrator(control_path, attachments_path, query_type):
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
    print('Looking for new emails')
    logging.info('Looking for new emails')
    start_time = time.time()
    all_mid_list = mailId_query(control_path, query_type = query_type)
    dt = time.time() - start_time
    print('Time spent: {}'.format(dt))
    print(' ')

    ## filtering new email:
    print('Filtering new emails')
    logging.info('Filtering new emails')
    start_time = time.time()
    splitted_mail_list, mail_counter = TheGreatFilter(all_mid_list, control_path)
    dt = time.time() - start_time
    print('Time spent: {}'.format(dt))
    print(' ')

    ## Getting mail data and attachments
    start_time = time.time()
    print('Getting mail data and attachments')
    logging.info('Getting mail data and attachments')
    if mail_counter > 0:
        df, df_attachments, df_to = TheCollector(splitted_mail_list, control_path, attachments_path)

        df = df.drop('mail_date_received_full', axis = 1)

        ## converting time to MySQL format
        df['mail_date_received'] = pd.to_datetime(df['mail_date_received'])
        df['mail_hms_received'] = df['mail_hms_received'].astype(str)
        df['mail_hms_received'] = pd.to_datetime(df['mail_hms_received']).dt.time
        df['mail_body'] = df['mail_body'].str.encode('utf-8')
        df['has_attachment'] = df['has_attachment'].astype(str)


        
        ## getting the mail ids already present in the DB:
        df_db = get_table('screening_table', 'mail_id')
        db_mail_id_list = list(set(df_db['mail_id'].tolist()))        
        
        
        
        ## dropping mails already in DB in order to avoid repeated entries:
        df = df[df['mail_id'].isin(db_mail_id_list) == False].reset_index(drop = True)
        df_attachments = df_attachments[df_attachments['mail_id'].isin(db_mail_id_list) == False].reset_index(drop = True)
        df_to = df_to[df_to['mail_id'].isin(db_mail_id_list) == False].reset_index(drop = True)
        

        # create sqlalchemy engine
        engine = create_engine("mysql://{user}:{pw}@localhost/{db}?charset=utf8mb4"
                               .format(user="root",
                                       pw="Le@ndro159753",
                                       db="screening"), encoding="utf8")
        logging.info('Loading MySQL database.')

        # Insert whole DataFrame into MySQL
        df.to_sql('screening_table', con = engine, if_exists = 'append', index = False, chunksize = 1000)
        logging.info('Data written to MySQL database.')
        engine.dispose()


    else:
        ## defining empty dfs for return purposes:
        df = pd.DataFrame()
        df_attachments = pd.DataFrame()
        df_to = pd.DataFrame()
        
        print('No new mail at this time!')
        logging.info('No new mail found.')



    dt = time.time() - start_time
    print('Time spent: {}'.format(dt))
    print(' ')
    dt_final = time.time() - global_start_time
    print('Total time spent: {}'.format(dt_final))


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



#######################################################################
def get_table(tablename, option = 'full'):
    
    '''
    Get the table and converts to a pandas dataframe.
    Input: i) tablename
           ii) option = 'full' (full table) or option = column_name 
    Output: i) df

    '''
    
    
    
    table_rows = None
    
    connection = mysql.connector.connect(host='localhost',
                                         database='screening',
                                         user='root',
                                         password='Le@ndro159753')    
    
    
    db_cursor = connection.cursor()
    db_cursor.execute("SHOW TABLES")
    tables_list = db_cursor.fetchall()
    tables_list = [item[0] for item in tables_list] 
    
    ## check if the inputed table name is in the database
    if tablename in tables_list:
    
        db_cursor = connection.cursor()
        ## getting all columns names:
        db_cursor.execute("SHOW columns FROM {}".format(tablename))
        columns_list = [column[0] for column in db_cursor.fetchall()]
        if option == 'full':
            db_cursor.execute("SELECT * FROM {}".format(tablename))
            table_rows = db_cursor.fetchall()
        else:
            if option in columns_list:
                columns_list = [option]
                db_cursor.execute("SELECT {} FROM {}".format(option, tablename))
                table_rows = db_cursor.fetchall()
            else:
                print('Column name not present in the table.')
                logging.error('Column name not present in the table.')
                
    else:
        print('Table not present in the database.')
        logging.error('Table not present in the database.')
    
    
    
    ## writing rows to a dataframe
    if table_rows is not None:
        df_test = pd.DataFrame(table_rows)   
        df_test.columns = columns_list
    else:
        df_test = pd.DataFrame()
    
    
    ## closing the connection
    connection.close()
   
    return df_test
#######################################################################





cwd = os.getcwd()
print(cwd)





## specifying the path to attachments:
attachments_path = os.path.join(cwd, '_attachments')
print(attachments_path)
if os.path.isdir(attachments_path) == False:
    os.mkdir(attachments_path)
    print('Folder created!')
else:
    print('Folder already exists!')
logging.info('Creating/verifying attachments folder')


## specifying the path to collect control:
control_path = os.path.join(cwd, '_collect_control')
print(control_path)
if os.path.isdir(control_path) == False:
    os.mkdir(control_path)
    print('Folder created!')
else:
    print('Folder already exists!')
logging.info('Creating/verifying control folder')



## getting the process id
pid = os.getpid()
with open(os.path.join(control_path, 'pid_DataCollector.dat'), 'w') as f:
    f.write(str(pid)+'\n')
f.close()
logging.info('Saving the PID.')





## scheduler
##############################
def job():
    '''
    The job for the scheduler
    '''

    ## choose randomly between a "full" or daily lookout
    query_type = random.choice(['full', 'day', 'day'])
    df, df_attachments, df_to = Orchestrator(control_path, attachments_path, query_type  = query_type)

    return
##############################

time_interval = 5
schedule.every(time_interval).minutes.do(job)

while True:

    schedule.run_pending()
    time.sleep(1)
