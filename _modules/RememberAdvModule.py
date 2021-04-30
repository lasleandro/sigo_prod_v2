import os
import pandas as pd
import sys
import numpy as np
import time
import schedule
from datetime import datetime
import shutil
from unidecode import unidecode
from datetime import datetime
import time
import logging
import random
import schedule
from scipy import stats


sys.path.insert(0, './')
from db_functions import TheDB
from logging_RememberAdv import StartLogger




###############################################################################
def GetAdvs():

    logger.info('GetAdvs function requested.')

    ## getting the screening table    
    df_screening = cqdb_prod.GetTable('Screening')
    logger.info('Screening table loaded.')    

    ## getting specific columns
    df = df_screening.loc[:,['mail_from', 'case_id', 'external_ref_number']]


    ## keeping only target emails:
    df = df[(df['mail_from'].str.contains('@carinaquito')) & \
                     (df['mail_from'].str.contains('denise') == False) &\
                     (df['mail_from'].str.contains('equipe') == False)].reset_index(drop = True)
    logger.info('Dropping internal emails (@carinaquito, denise, equipe etc).')

    ## dropping rows with null case_id and external_ref_number
    df = df[(df['case_id'] != '') | (df['external_ref_number'] != '')]


    ## agrupando por 'case_id' + 'external_ref_number' e pegando a moda dos emails
    df_adv = df.groupby(['case_id', 'external_ref_number'])['mail_from'].apply(list).\
    apply(stats.mode).to_frame().\
    reset_index()

    ############################
    def GetMode(input_tuple):
        return input_tuple[0][0]
    #############################

    ## pegando a moda
    df_adv['mail_from'] = df_adv['mail_from'].apply(GetMode)
    logger.info('Moda capturada! Exiting GetAdvs function.')
    
    
    return df_adv
################################################################################


################################################################################
def GetAdvsFullTable():
    '''
    Merging df_adv with users table (DB)
    '''

    logger.info('GetAdvsFullTable function requested.')

    df_adv = GetAdvs()
    logger.info('df_adv dataframe created.')

    df_users = cqdb_prod.GetTable('Users')
    logger.info('Users table loaded.')

    df_users = df_users.loc[:, ['users_id', 'email', 'first_name', 'last_name']]

    df_adv_final = df_adv.merge(df_users, left_on = 'mail_from', right_on = 'email', how = 'left')
    df_adv_final = df_adv_final.drop('email', axis = 1)
    logger.info('df_adv and df_users merged!')

    ## usuários não cadastrados recebem o id 99999
    df_adv_final['users_id'] = df_adv_final['users_id'].fillna(99999)
    df_adv_final['users_id']  = df_adv_final['users_id'].astype(int)

    logger.info('df_adv_final created! Exiting GetAdvsFullTable function.')
    
    return df_adv_final
################################################################################




################################################################################
def ConsultAdv(case_id, external_ref_number):
    '''
    Retorna o id do adv (user_id) associado ao case_id/external_ref_number.
    '''
    logger.info('ConsultAdv function requested.')

    df_adv = GetAdvsFullTable()
    logger.info('df_adv created!')

    users_list = []
    if case_id != '' and external_ref_number != '': 
        logger.info('Both inputs are non null.')
        result_list = \
        df_adv[(df_adv['case_id'] == case_id) &\
               (df_adv['external_ref_number'] == external_ref_number)]['users_id'].tolist()
        if len(result_list) != 0:
            user_id = result_list[0]
            users_list.append(user_id)



        result_list = \
        df_adv[(df_adv['case_id'] == case_id)]['users_id'].tolist()
        if len(result_list) != 0:
            user_id = result_list[0]
            users_list.append(user_id)
        


        result_list = \
        df_adv[(df_adv['external_ref_number'] == external_ref_number)]['users_id'].tolist()
        if len(result_list) != 0:
            user_id = result_list[0]
            users_list.append(user_id)
            

            
            
    elif case_id != '' and external_ref_number == '':
        logger.info('Only case_id is non null')
        result_list = \
        df_adv[(df_adv['case_id'] == case_id)]['users_id'].tolist()
        if len(result_list) != 0:
            user_id = result_list[0]
            users_list.append(user_id)
  
            
            
    elif case_id == '' and external_ref_number != '':
        logger.info('Only external_ref_number is non null')
        result_list = \
        df_adv[(df_adv['external_ref_number'] == external_ref_number)]['users_id'].tolist()
        if len(result_list) != 0:
            user_id = result_list[0]
            users_list.append(user_id)
        
        
    if len(set(users_list)) >= 1:
        users_id = users_list[0]
    else:
        users_id = 99999
    
    logger.info('User_id determined. Exiting ConsultAdv. Success!!')
    
    return users_id
################################################################################


scriptname = os.path.basename(__file__)

## lets load the logger:
logger = logging.getLogger('RememberAdv_logger')
logger.info('{} running!'.format(scriptname))

## bots BD connections
conn_dict = {'host':'localhost', 'database': 'brspcqadev',
             'user':'root', 'password':'quito2020'}

cqdb_prod = TheDB(conn_dict) 
logger.info('DB object created.')



