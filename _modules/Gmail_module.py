from __future__ import print_function
import pickle
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from oauth2client import file, client, tools
from apiclient import errors
import base64
from datetime import datetime
import pandas as pd
from regex_module import FindEmail, FindNumbersBraket, FindLawsuit
import logging
from logging_module import getlogger


## log
logging = getlogger()

dirname = os.path.dirname(__file__)
path_cred = os.path.join(dirname, 'gmail_credentials')


###################################################################
def GetCredentials(path_cred):

    '''
    Reading (or asking for) credentials.
    Output: credentials
    '''

    ## Credentials

    #cwd = os.getcwd()
    #path_cred = os.path.join(cwd, 'gmail_credentials')


    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(os.path.join(path_cred, 'token.pickle')):
        with open(os.path.join(path_cred, 'token.pickle'), 'rb') as token:
            creds = pickle.load(token)
            print('Credentials ok!')
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        print('Credentials not ok! Asking for one...')
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join(path_cred, 'credentials.json'), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(os.path.join(path_cred, 'token.pickle'), 'wb') as token:
            pickle.dump(creds, token)

    return creds
####################################################################################





######################################################################################
def GetAttachments_v2(message, output_path):
    """Get and store attachment from Message with given id.

    Inputs: i)   service: Authorized Gmail API service instance.
            ii)  user_id: User's email address. The special value "me" can be used to indicate the authenticated user.
            iii) mail_id: ID of Message containing attachment.
    Output: i) a dictionary whose key is the mail_id and the value is a list containing
            all attachment filenames.
            ii) The files saved in the output path
    """


    mail_id = message['id']
    user_id = 'me'
    ## list the filenames:
    attachment_dict = {mail_id:[]}

    try:

        for part in message['payload']['parts']:
            if part['filename']:
                ## getting the filename
                attachment_filename = part['filename']


                if '.jpg' not in attachment_filename and '.png' not in attachment_filename:
                    attachment_dict[mail_id].append(attachment_filename)
                    if 'data' in part['body']:
                        data = part['body']['data']
                    else:
                        att_id = part['body']['attachmentId']
                        att = service.users().messages().attachments().get(userId=user_id,\
                                                                           messageId=mail_id,id=att_id).execute()
                        data = att['data']
                    file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))


                    final_path = os.path.join(output_path, str(mail_id))
                    if os.path.isdir(final_path) == False:
                        os.mkdir(final_path)

                    with open(os.path.join(final_path, attachment_filename), 'wb') as f:
                        f.write(file_data)



    except:
        print('An error occurred. GetAttachments_v2')
        pass



    return attachment_dict
##############################################################################################################



######################################
def GetMessage_v2(message):
    try:
        snippet = message['snippet']
    except:
        snippet = ''

    return snippet
#######################################



#################################################################
def ListMessagesMatchingQuery(query=''):
  """List all Messages of the user's mailbox matching the query.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    query: String used to filter messages returned.
    Eg.- 'from:user@some_domain.com' for Messages from a particular sender.

  Returns:
    List of Messages that match the criteria of the query. Note that the
    returned list contains Message IDs, you must use get with the
    appropriate ID to get the details of a Message.
  """
  try:
    response = service.users().messages().list(userId=user_id,
                                               q=query).execute()
    messages = []
    if 'messages' in response:
      messages.extend(response['messages'])

    while 'nextPageToken' in response:
      page_token = response['nextPageToken']
      response = service.users().messages().list(userId=user_id, q=query,
                                         pageToken=page_token).execute()
      messages.extend(response['messages'])

    return messages
  except:
    print ('An error occurred')
########################################################################




#####################################################################
def GetBody(message):

    '''
    Gets the email body in text/string format.
    Input: i) message JSON
    Output: ii) the text
    '''



    if 'data' in message['payload']['body']:
        data = message['payload']['body']['data']

    elif 'data' in message['payload']['parts'][0]['body']:
        data = message['payload']['parts'][0]['body']['data']

    elif 'data' in message['payload']['parts'][0]['parts'][0]['body']:
        data = message['payload']['parts'][0]['parts'][0]['body']['data']

    elif 'data' in message['payload']['parts'][0]['parts'][0]['parts'][0]['body']:
        data = message['payload']['parts'][0]['parts'][0]['parts'][0]['body']['data']

    else:
        data = ''








    conv_data = base64.urlsafe_b64decode(data.encode('UTF-8')).decode('utf-8')

    return conv_data
#################################################################################


#################################################################################
def GetBody_v2(message):


    data = ''

    if 'data' in message['payload']['body']:
        data = message['payload']['body']['data']


    else:

        for part1 in message['payload']['parts']:
            mimeType = part1['mimeType']
            #print(mimeType)

            if 'multipart' in mimeType:
                for part2 in part1['parts']:
                    mimeType2 = part2['mimeType']
                    #print(mimeType2)
                    if 'multipart' in mimeType2:
                        for part3 in part2['parts']:
                            mimeType3 = part3['mimeType']
                            if 'text/plain' in mimeType3:
                                #print(part3['mimeType'])
                                #print(part3['body']['size'])
                                data = part3['body']['data']
                                #print('Got data 3!')

                    elif 'text/plain' in mimeType2:
                        #print(part2['mimeType'])
                        #print(part2['body']['size'])
                        data = part2['body']['data']
                        #print('Got data 1!')

            elif 'text/plain' in mimeType:
                data = part1['body']['data']
                #print('Got data 2!')


    if data != '':
        data = base64.urlsafe_b64decode(data.encode('UTF-8')).decode('utf-8')


    return data
########################################################################################



#########################################################################################
def FDS_old(mail_list, att_output_path, control_path):

    df = pd.DataFrame(columns = ['mail_id', 'thread_id', 'mail_date_received_full',\
                                 'mail_from', 'mail_subject',
                                 'mail_body', 'mail_body_snippet', 'has_attachment'])




    count = 0
    df_list = []
    df_list2 = []
    for m_id in mail_list:
        #m_id = mail['id']
        print(m_id)

        try: 
            message = service.users().messages().get(userId='me', id=m_id).execute()
        except:
            logging.error("Message request failed")

            ## save error mails to file
            filename = 'subset_mail_error.dat'
            f = open(os.path.join(control_path, filename), 'a+')
            f.write(str(m_id)+'\n')
            f.close()

            continue

        thread_id = message['threadId']
        payload = message['payload']
        header = payload['headers']
        print('thread_id: '.format(thread_id))
        

        ## in case of not capturing the destinataries list, lets define an empty list
        destinataries_list = []
        for item in header:
            if item['name'] == 'From':
                sender = item['value']
                print('From: {}'.format(sender))
            if item['name'] == 'Date':
                date = item['value']
                print('Date: {}'.format(date))
            if item['name'] == 'Subject':
                subject = item['value']
                print('Subject: {}'.format(subject))
            if item['name'] == 'To':
                destinataries = item['value']
                destinataries_list = destinataries.split(',')
                destinataries_list = [item.strip() for item in destinataries_list]

        print(' ')
        print(' ')



        ## getting the text/mail body

        try:
            text = GetBody_v2(message)
        except:
            logging.error("Text not found!")
            text = ''

        try:
            snippet = GetMessage_v2(message)
        except:
            logging.error("Snippet not found!")
            snippet = ''



        ## writing to frame
        df.at[count, 'mail_id'] = m_id
        df.at[count, 'thread_id'] = thread_id
        df.at[count, 'mail_date_received_full'] = date
        df.at[count, 'mail_from'] = sender
        df.at[count, 'mail_subject'] = subject
        df.at[count, 'mail_body'] = text
        df.at[count, 'mail_body_snippet'] = snippet




        ## getting the attachments
        attachment_dict = GetAttachments_v2(message, att_output_path)
        files_list =  list(attachment_dict.values())[0]
        tot_attachments = len(files_list)

        ## recreating the full path to the files:
        link_list = [os.path.join(att_output_path, str(m_id), item) for item in files_list]


        ## creating a child dataframe that stores all attachments filenames
        if tot_attachments != 0:
            has_attachment = 1

            df_attachments = pd.DataFrame(columns = ['mail_id', 'doc_name', 'doc_link'])

            df_attachments['doc_name'] = files_list
            df_attachments['mail_id'] = m_id
            df_attachments['doc_link'] = link_list
            df_list.append(df_attachments)

        else:
            has_attachment = 0


        ## inserting the existence or not of attachment
        df.at[count, 'has_attachment'] = has_attachment


        ## creating a child dataframe that stores all destinataries of the mail
        df_destiny = pd.DataFrame(columns = ['mail_id', 'destinatary_mail'])
        df_destiny['destinatary_mail'] = destinataries_list
        df_destiny['mail_id'] = m_id
        df_list2.append(df_destiny)



        count += 1



    df['mail_date_received_full'] = pd.to_datetime(df['mail_date_received_full'], utc = True)
    df['mail_date_received'] = [d.date() for d in df['mail_date_received_full']]
    df['mail_hms_received'] =  df['mail_date_received_full'].dt.time


    ## concatenating the attachments dataframes
    if len(df_list) >= 1:
        df_attachments_final = pd.concat(df_list).reset_index(drop = True)
    else:
        df_attachments_final = pd.DataFrame(columns = ['mail_id', 'doc_name', 'doc_link'])

    ## concatenating the destinataries dataframes
    if len(df_list2) >= 1:
        df_destiny_final = pd.concat(df_list2).reset_index(drop = True)
    else:
        df_destiny_final = pd.DataFrame(columns = ['mail_id', 'destinatary_mail'])



    ## some Regex
    df['mail_from'] = df['mail_from'].apply(FindEmail)
    df['case_id'] = df['mail_subject'].apply(FindNumbersBraket)
    df['external_ref_number'] = df['mail_body'].apply(FindLawsuit)

    df_destiny_final['destinatary_mail'] = df_destiny_final['destinatary_mail'].apply(FindEmail)

    return df, df_attachments_final, df_destiny_final
##########################################################################






#########################################################################################
def FDS(mail_list, att_output_path, control_path):

    df = pd.DataFrame(columns = ['mail_id', 'thread_id', 'mail_date_received_full',\
                                 'mail_from', 'mail_subject',
                                 'mail_body', 'mail_body_snippet',\
                                 'has_attachment', 'attachment_folder_link'])




    count = 0
    df_list = []
    df_list2 = []
    for m_id in mail_list:
        
        logging.info("Mail_id {}".format(m_id))
 

        try: 
            message = service.users().messages().get(userId='me', id=m_id).execute()
            got_mail = True
           
        except:
            got_mail = False
            logging.error("Message request failed")

            ## save error mails to file
            filename = 'subset_mail_error.dat'
            f = open(os.path.join(control_path, filename), 'a+')
            f.write(str(m_id)+'\n')
            f.close()
            
            
            ## tests
            ## faking message fields:
            message = {'threadId':'', 'payload':{'headers':[{'name':''}]}}
            sender = ''
            date = ''
            subject = ''

            pass

        
        

        thread_id = message['threadId']
        payload = message['payload']
        header = payload['headers']
 

        ## in case of not capturing the destinataries list, lets define an empty list
        destinataries_list = []
        for item in header:
            if item['name'] == 'From':
                sender = item['value']
                print('From: {}'.format(sender))
            if item['name'] == 'Date':
                date = item['value']
                print('Date: {}'.format(date))
            if item['name'] == 'Subject':
                subject = item['value']
                print('Subject: {}'.format(subject))
            if item['name'] == 'To':
                destinataries = item['value']
                destinataries_list = destinataries.split(',')
                destinataries_list = [item.strip() for item in destinataries_list]

        print(' ')
        print(' ')



        ## getting the text/mail body
        if got_mail == True:
            try:
                text = GetBody_v2(message)
            except:
                logging.error("Text not found!")
                text = ''

            try:
                snippet = GetMessage_v2(message)
            except:
                logging.error("Snippet not found!")
                snippet = ''
        else:
            text = ''
            snippet = ''



        ## writing to frame
        df.at[count, 'mail_id'] = m_id
        df.at[count, 'thread_id'] = thread_id
        df.at[count, 'mail_date_received_full'] = date
        df.at[count, 'mail_from'] = sender
        df.at[count, 'mail_subject'] = subject
        df.at[count, 'mail_body'] = text
        df.at[count, 'mail_body_snippet'] = snippet
        df.at[count, 'attachment_folder_link'] = str(os.path.join(att_output_path, m_id))



        if got_mail == True:
            ## getting the attachments
            attachment_dict = GetAttachments_v2(message, att_output_path)
            files_list =  list(attachment_dict.values())[0]
            tot_attachments = len(files_list)

            ## recreating the full path to the files:
            link_list = [os.path.join(att_output_path, str(m_id), item) for item in files_list]


            ## creating a child dataframe that stores all attachments filenames
            if tot_attachments != 0:
                ## sum 1 to match Marcio's DB definition
                has_attachment = 1 + 1

                df_attachments = pd.DataFrame(columns = ['mail_id', 'doc_name', 'doc_link'])

                df_attachments['doc_name'] = files_list
                df_attachments['mail_id'] = m_id
                df_attachments['doc_link'] = link_list
                df_list.append(df_attachments)

            else:
                ## sum 1 to match Marcio's DB definition
                has_attachment = 0 + 1
                
        else:
            df_attachments = pd.DataFrame(columns = ['mail_id', 'doc_name', 'doc_link'])
            has_attachment = 0


        ## inserting the existence or not of attachment
        df.at[count, 'has_attachment'] = has_attachment

        
        if got_mail == True:
            ## creating a child dataframe that stores all destinataries of the mail
            df_destiny = pd.DataFrame(columns = ['mail_id', 'destinatary_mail'])
            df_destiny['destinatary_mail'] = destinataries_list
            df_destiny['mail_id'] = m_id
            df_list2.append(df_destiny)
        else:
            ## creating a child dataframe that stores all destinataries of the mail
            df_destiny = pd.DataFrame(columns = ['mail_id', 'destinatary_mail'])
            df_list2.append(df_destiny)            



        count += 1

    ## eliminating problematic mails:
    df = df[df['mail_date_received_full'] != ''].reset_index(drop = True)
    
    ## converting date
    df['mail_date_received_full'] = pd.to_datetime(df['mail_date_received_full'], utc = True)
    df['mail_date_received'] = [d.date() for d in df['mail_date_received_full']]
    df['mail_hms_received'] =  df['mail_date_received_full'].dt.time


    ## concatenating the attachments dataframes
    if len(df_list) >= 1:
        df_attachments_final = pd.concat(df_list).reset_index(drop = True)
    else:
        df_attachments_final = pd.DataFrame(columns = ['mail_id', 'doc_name', 'doc_link'])

    ## concatenating the destinataries dataframes
    if len(df_list2) >= 1:
        df_destiny_final = pd.concat(df_list2).reset_index(drop = True)
    else:
        df_destiny_final = pd.DataFrame(columns = ['mail_id', 'destinatary_mail'])



    ## some Regex
    df['mail_from'] = df['mail_from'].apply(FindEmail)
    df['case_id'] = df['mail_subject'].apply(FindNumbersBraket)
    df['external_ref_number'] = df['mail_body'].apply(FindLawsuit)

    df_destiny_final['destinatary_mail'] = df_destiny_final['destinatary_mail'].apply(FindEmail)

    return df, df_attachments_final, df_destiny_final
##########################################################################




######################################################################################3
def MailData(mail_list):

    df = pd.DataFrame(columns = ['mail_id', 'thread_id', 'mail_date_received_full',\
                                 'mail_from', 'mail_subject',
                                 'mail_body', 'mail_body_snippet'])




    count = 0
    df_list2 = []
    for mail in mail_list:
        m_id = mail['id']
        thread_id = mail['threadId']

        message = service.users().messages().get(userId='me', id=m_id).execute()
        payload = message['payload']
        header = payload['headers']

        print(m_id)
        for item in header:
            if item['name'] == 'From':
                sender = item['value']
                print('From: {}'.format(sender))
            if item['name'] == 'Date':
                date = item['value']
                print('Date: {}'.format(date))
            if item['name'] == 'Subject':
                subject = item['value']
                print('Subject: {}'.format(subject))
            if item['name'] == 'To':
                destinataries = item['value']
                destinataries_list = destinataries.split(',')
                destinataries_list = [item.strip() for item in destinataries_list]

        print(' ')
        print(' ')



        ## getting the text/mail body
        text = GetBody_v2(message)
        snippet = GetMessage_v2(message)



        ## writing to frame
        df.at[count, 'mail_id'] = m_id
        df.at[count, 'thread_id'] = thread_id
        df.at[count, 'mail_date_received_full'] = date
        df.at[count, 'mail_from'] = sender
        df.at[count, 'mail_subject'] = subject
        df.at[count, 'mail_body'] = text
        df.at[count, 'mail_body_snippet'] = snippet






        ## creating a child dataframe that stores all destinataries of the mail
        df_destiny = pd.DataFrame(columns = ['mail_id', 'destinatary_mail'])
        df_destiny['destinatary_mail'] = destinataries_list
        df_destiny['mail_id'] = m_id
        df_list2.append(df_destiny)



        count += 1



    df['mail_date_received_full'] = pd.to_datetime(df['mail_date_received_full'], utc = True)
    df['mail_date_received'] = [d.date() for d in df['mail_date_received_full']]
    df['mail_hms_received'] =  df['mail_date_received_full'].dt.time


    ## concatenating the destinataries dataframes
    df_destiny_final = pd.concat(df_list2).reset_index(drop = True)



    df['mail_from'] = df['mail_from'].apply(FindEmail)
    df['case_id'] = df['mail_subject'].apply(FindNumbersBraket)
    df['external_ref_number'] = df['mail_body'].apply(FindLawsuit)


    return df, df_destiny_final
##########################################################################






######################################################################
def PrintMail(mail_list, df, full = False):

    for mail_id in mail_list:
        print(mail_id)
        idx = df[df['mail_id'] == mail_id].index[0]
        print(df[df['mail_id'] == mail_id]['mail_subject'][idx])
        print(df[df['mail_id'] == mail_id]['mail_date_received'][idx])
        print(df[df['mail_id'] == mail_id]['mail_from'][idx])
        print('-----------')
        if full == True:
            print(df[df['mail_id'] == mail_id]['mail_body'][idx])
        else:
            print(df[df['mail_id'] == mail_id]['mail_body_snippet'][idx])
        print('-----------')
        print('-----------')
        print('-----------')
##############################################################################





## reading credentials:
creds = GetCredentials(path_cred)

user_id = 'me'
## creating service
## cache_discovery = False in order to fix a bug (https://github.com/googleapis/google-api-python-client/issues/817)
service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
