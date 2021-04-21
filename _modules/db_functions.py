from pandas.io import sql
import pandas as pd
import os
import json
import ast
from sqlalchemy import create_engine
import pymysql
#import logging
import mysql.connector
from mysql.connector import Error
import  numpy as np





dirname = os.path.dirname(__file__)




class TheDB:
    
    ###########################################
    def __init__(self, conn_dict):
        self.host = conn_dict['host']
        self.database = conn_dict['database']
        self.user = conn_dict['user']
        self.password = conn_dict['password']
        if 'port' in conn_dict.keys():
            self.port = conn_dict['port']
    ###########################################

    
    ##############################################
    def Conn(self):
        
        if hasattr(self, 'port') == True:
            connection = mysql.connector.connect(host=self.host,
                                                 database = self.database,
                                                 user=self.user,
                                                 password=self.password,
                                                 port=self.port)
        else:
            connection = mysql.connector.connect(host=self.host,
                                                 database = self.database,
                                                 user=self.user,
                                                 password=self.password)
        return connection
    ##################################################
        
    
    ###################################################
    def TablesList(self):
        '''
        Lists the tables of a given database
        '''



        connection = self.Conn()
        db_cursor = connection.cursor()    



        db_cursor.execute('USE {}'.format(self.database))
        db_cursor.execute("SHOW TABLES")
        tables_list = db_cursor.fetchall()
        tables_list = [item[0] for item in tables_list]


        return tables_list
    #######################################################
    
    ###################################################
    def TablesDataType(self, tablename):
        '''
        Lists the tables of a given database
        '''



        connection = self.Conn()
        db_cursor = connection.cursor()    



        db_cursor.execute("SHOW columns FROM {}".format(tablename))
        table_rows = db_cursor.fetchall()
        df_test = pd.DataFrame(table_rows)
        

        return df_test
    #######################################################    
  
    
    #######################################################################
    def GetTable(self, tablename, option = 'full'):

        '''
        Get the table and converts to a pandas dataframe.
        Input: i) tablename
               ii) option = 'full' (full table) or option = column_name 
        Output: i) df

        '''

        host = self.host
        user = self.user
        password = self.password
        database = self.database


        ## check if the inputed table name is in the database
        if tablename in self.TablesList():

            db_connection_str = 'mysql+pymysql://{}:{}@{}/{}'.format(user, password, host, database)
            db_connection = create_engine(db_connection_str)        


            if option == 'full':
                df_test = pd.read_sql('SELECT * FROM {}'.format(tablename), con=db_connection)
            else:

                try:
                    df_test = pd.read_sql('SELECT {} FROM {}'.format(option, tablename), con=db_connection)
                except:
                    print('Column do not exist. Please correct and try again.')



        else:
            logging.error('Table not present in the database.')
            df_test = pd.DataFrame()


        if tablename == 'examples_table':
            df_test['example_text'] = df_test['example_text'].str.decode('utf-8', 'surrogatepass')



        return df_test
    #######################################################################
    
    ####################################################################### 
    def InsertDB(self, df, table_db, key_column = '' , return_table = False, append_table = True):


        def PandasDiff(df1, df2):
            
            '''
            In set theory notation: df_diff = df1 - df2
            Inputs: i) df1, ii) df2
            Output: the difference df1 - df2
            '''

            df1=df1.drop_duplicates(keep="first") 
            df2=df2.drop_duplicates(keep="first") 
            df_diff = pd.concat([df1,df2]).drop_duplicates(keep=False).reset_index(drop = True)
            
            
            return df_diff



        ## create sqlalchemy engine
        engine = create_engine("mysql://{user}:{pw}@{host}/{db}?charset=utf8mb4"
                               .format(user= self.user,
                                       pw=self.password,
                                       host = self.host,
                                       db=self.database), encoding="utf8")


        try:
            print('Original length: {}'.format(len(df)))


            ## checking the examples already in db
            df_db = self.GetTable(table_db, option = 'full')


            
            if key_column != '':
                examples_list = df_db[key_column].tolist()
                rows_db = len(examples_list)


                print('DB length: {}'.format(rows_db))

                ## filtering out the examples already in db
                df = df[df[key_column].isin(examples_list) == False].reset_index(drop = True)

            ## if no key column is specified, then the difference between the input dataframe
            ## and the DB table (with columns matched to the input df) is considered to be inserted in the DB
            else:
                columns_list = list(df.columns)
                df_db = df_db.loc[:, columns_list]
                df = PandasDiff(df, df_db)

            new_rows = len(df)
            print('Final length: {}'.format(new_rows))
            new_dict = {'new_rows': new_rows}


            if new_rows != 0:

                ## Insert whole df into MySQL db

                if append_table == True:
                    df.to_sql(table_db, con = engine, if_exists = 'append',\
                                index = False, chunksize = 1000)
                else:
                   df.to_sql(table_db, con = engine, if_exists = 'replace',\
                                index = False, chunksize = 1000)
                                                    
                #logging.info('Data written to MySQL database.')
                engine.dispose()
                print('Success! Data writtten in db!')
            else:
                print('No data to be written in the db.')



        except Exception as e:
            print('Error. Please correct and try again.')
            print(e.__doc__)
            print(e)
            print(' ')
            pass

        if return_table == True:
            ## extracting the table from the database:
            df_db = self.GetTable(table_db, option = 'full')
            #df_db['example_text'] = df_db['example_text'].str.decode('utf-8', 'surrogatepass')

            return df_db, new_dict

        else:
            return
        
        
        
        
    #######################################################################    
    def InsertExamples(self, df, return_table = False):

        ## create sqlalchemy engine
        engine = create_engine("mysql://{user}:{pw}@{host}/{db}?charset=utf8mb4"
                               .format(user= self.user,
                                       pw=self.password,
                                       host = self.host,
                                       db=self.database), encoding="utf8")


        if 'example_text' in df.columns:

            ## locking the interest column to match the db table
            df2 = df.loc[:, ['example_text']]
            print('Original length: {}'.format(len(df2)))


            ## checking the examples already in db
            df_db = self.GetTable('examples_table', option = 'full')
            examples_list = df_db['example_text'].tolist()

            rows_db = len(examples_list)
            print('DB length: {}'.format(rows_db))

            ## filtering out the examples already in db
            df2 = df2[df2['example_text'].isin(examples_list) == False].reset_index(drop = True)

            new_rows = len(df2)
            print('Final length: {}'.format(new_rows))
            new_dict = {'new_rows': new_rows}


            if new_rows != 0:
                ## Insert whole df into MySQL db
                df2.to_sql('examples_table', con = engine, if_exists = 'append',\
                               index = False, chunksize = 1000)
                #logging.info('Data written to MySQL database.')
                engine.dispose()
                print('Success! Data writtten in db!')
            else:
                print('No data to be written in the db.')



        else:
            print('Error. Dataframe needs an "example_text" column. Please try again.')
            print(' ')


        if return_table == True:
            ## extracting the table from the database:
            df_db = self.GetTable('examples_table', option = 'full')
            #df_db['example_text'] = df_db['example_text'].str.decode('utf-8', 'surrogatepass')

            return df_db, new_dict

        else:
            return 
#######################################################################


#######################################################################
    def InputW2V(self, df, text_column, basename, return_table = False):
        '''
        Inputs text data to the w2v_table in metabot DB
        Input: i) df: the df input
            ii) text_column: the name of the column that contains the interest text
            iii) basename: a label that identifies the origin of the text being inputed
            iv) db_object: the DB object
        Output: nothing
        '''
        
        df = df.loc[:, [text_column]]
        df = df.rename(columns = {text_column: 'text'})
        
        df = df.dropna().reset_index(drop = True)
        
        if type(df['text'][0]) != bytes:
            
            try:
                df['text'] = df['text'].astype(str)
            except:
                df['text'] = df['text'].str.decode('utf-8', 'ignore').astype(str)
                
                
            df['text'] = df['text'].str.encode('utf-8', 'ignore')
            self.InsertDB(df, 'word2vec_table', key_column = 'text')

        if return_table == True:
            ## extracting the table from the database:
            df_db = self.GetTable('word2vec_table', option = 'full')
            


        if return_table == True:
            return df_db

        else:
            return   
            
            
        
 #######################################################################



 #######################################################################
    def MainUpdate(self, df, tablename):
        '''
        Updates a table based on df values and columns
        Input: i) the DB object
            ii) df with columns = ['id', 'value']. 'id' and 'value' columns name must be in the table inside DB.
            iii) the tablename
        Output: the sql queries, only for check purposes.
        '''
        
        
        def UpdateRows(x, df, tablename, connection):
            '''
            Updates columns values based on df rows. Function to be applied to a Pandas df.
            Input: i) the df row (for use with .apply)
                ii) df with columns = ['id', 'value']. 'id' and 'value' columns name must be in the table inside DB.
                iii) the tablename
            Output: the sql queries, only for check purposes.
            '''

            db_cursor = connection.cursor()

            ## getting the columns of the df
            columns_list = df.columns
            
            ## identifying the id columns and the value column inside a dictionary
            columns_dict = {'id': columns_list[0], 'value': columns_list[1]}

            ## rounding and converting id and value to int
            theId = int(round(x[0],0))
            theValue = int(round(x[1],0))

            ## writing the query to update the table
            sql = "UPDATE {} SET {} = {} WHERE {} = {};".format(tablename, 
                                                                columns_dict['value'], theValue,\
                                                                columns_dict['id'], theId)

            ## executing and commiting the update
            db_cursor.execute(sql)
            connection.commit()


            return sql    
        
        
        
        
        ## creating the connection
        connection = self.Conn()
        
        ## applying the function UpdateRows to the entire df
        result = df.apply(UpdateRows, df = df, tablename = tablename, connection = connection, axis=1)
        
        ## closing connection
        connection.close()
        
        ## returning a df with all queries
        df_queries = result.to_frame()
        df_queries.columns = ['query']
        
        return df_queries
 #######################################################################




