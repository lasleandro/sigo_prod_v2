import pandas as pd
from datetime import datetime
import os
import glob
from bs4 import BeautifulSoup
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
tqdm.pandas()
import time
import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import itertools
import networkx as nx
#from pyvis.network import Network

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn import metrics

import numpy as np

#import pygraphviz
import pandas as pd
#import holoviews as hv
import networkx as nx
from langdetect import detect

import logging


###################################################################
def datetag_func():
    '''
    Creates a datetag.
    Input: none
    Output: string yyyy-m-dd-hh-mm
    '''
    ## datetag
    date = datetime.today()
    date_tag = '-'.join([str(date.year), str(date.month), str(date.day),\
    str(date.hour), str(date.minute)])
    
    
    return date_tag
########################################################################


####################################################
def FlattenList(input_list):
    
    '''
    Flattens a list of lists
    Input: a list
    Output: flatten list
    '''

    flatten_list = list(itertools.chain(*input_list))
    
    return flatten_list
#####################################################


######################################################################################
def upper_test(token):
    '''
    checks if a given token is completely upper case
    input: string
    output: boolean: True or False
    '''
    
    test_list = []
    for character in token:
        ## skipping digits:
        if character.isalpha() == True:
            result = character.isupper()
            test_list.append(result)

            
        
    ## eliminating repetition
    test2_list = list(set(test_list))

    ## if len = 0 or len > 1, it means there were some error
    ## or there are True and Falses inside the list
    if len(test2_list) > 1:
        final_result = False
    elif len(test2_list) == 0:
        final_result = True
    else:
        element = test2_list[0]
        if element == True:
            final_result = True
        else:
            final_result = False
    
    
    return final_result
################################################################################


################################################################################
## gets the last element in a list
def get_except_first(input_list):
    '''
    Function that gets the last element of a
    list. Usefull for Pandas.
    '''
    
    
    if len(input_list) >= 2:
        clean_title = ' '.join(input_list[1:])
    else:
        clean_title = input_list[-1]
    
    
    return clean_title
##############################################################################


############################################################################
def clean_text(text, switch = 0):
    
    '''
    Clean text from non alphanumeric characters.
    If switch = 0, filters stopwords, numbers, punctuation and alphanumeric tokens.
    Elif switch = 1, filters stopwords, keeps punctuation and numbers.
    Elif switch = 2, filters only alphanumeric characters.
    
    Input: text string and switch (optional, default switch = 0)
    '''
    

    if type(text) == str:



        ## detecting the text language
        lang = language(text)


        
        def hasNumbers(inputString):
            return any(char.isdigit() for char in inputString)    
        
        
        
        
        #text = text.decode('utf-8')
        text = text.lower()
        
            
        new_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        new_text = re.sub(r'www(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', new_text)
        new_text = re.sub(r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', ' ', new_text)
        new_text = ' '.join(new_text.split())
        
        
        
        delete_list =  ['gostei de um video @youtube', '_', 'utc', '&quot', '&lt', ';' \
                        'www.','.com','.',',','!','?',';','@','...','/','(',')','"',':',\
                    '=','#','-','&','|','%','*','"',"'s",'[', ']'," ' ","$", '\n', '>', '<', '+',\
                    'rss', '{', '}',
                    'website', 'https:','https', 'http', 'http:', "'", 'feedly', 'sign', 'comment', \
                    'clique aqui', 'click here', 'comments', 'quot', 'quote']


        delete_list2 = ['|','§','ª', '°', '’', '“', '”', '•', '\n']
    
        
        
        

        
        
        if switch == 0:
            
            delete_list = delete_list + delete_list2
            
            for item in delete_list:
                new_text = new_text.replace(item, ' ')

            new_text = ' '.join(new_text.split())
            new_text = unidecode(new_text)




            tokens_list = new_text.split()        
            
            

            if lang == 'pt':
                filtered_tokens_list = [token for token in tokens_list if token not in stopwords.words('portuguese') ]
            elif lang == 'en':
                filtered_tokens_list = [token for token in tokens_list if token not in stopwords.words('english')]
            else:
                filtered_tokens_list = tokens_list[:]
            
            filtered_tokens_list = [token for token in filtered_tokens_list if hasNumbers(token) == False \
                            and len(token) > 1]
            
        elif switch == 1:
            
                
            for item in delete_list2:
                new_text = new_text.replace(item, ' ')

            new_text = ' '.join(new_text.split())
            new_text = unidecode(new_text)
            
            

            tokens_list = new_text.split()         
            
            
            
            if lang == 'pt':
                filtered_tokens_list = [token for token in tokens_list if token not in stopwords.words('portuguese') ]
            elif lang == 'en':
                filtered_tokens_list = [token for token in tokens_list if token not in stopwords.words('english') ]
            else:
                filtered_tokens_list = tokens_list[:]

        else:
            
            tokens_list = new_text.split()  
            
            filtered_tokens_list = tokens_list[:]
            
            filtered_tokens_list = [token for token in filtered_tokens_list if hasNumbers(token) == False \
                            and len(token) > 1]        
            
            
        
        

        
        
        final_text = ' '.join(filtered_tokens_list)


    else:
        print('The input is not a string')
        logging.warning('The input is not a string')
        final_text = ''
    
    
    return final_text
#############################################################################






############################################################################
# return the similarity matrix (based on cosine distance)
def cos_matrix(df, cod, desc):
    '''
    Calculates the cosine similarity matrix.
    Inputs: i) the dataframe
            ii) the labels column
            iii) the text column
            
    Outputs: the df_cos matrix and the tfidf matrix
    
    '''
    
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix_insiders = tfidf_vectorizer.fit_transform(df[desc])

    cos_sim_array = cosine_similarity(tfidf_matrix_insiders, tfidf_matrix_insiders)

    df_cos = pd.DataFrame(cos_sim_array, columns = df[cod].tolist(),\
                          index = df[cod].tolist() )
    
    return df_cos, tfidf_matrix_insiders
#################################################################################





##################################################################################
# return the similarity matrix (based on cosine distance)
def cos_matrix_v2(df, desc):
    
    '''
    Calculates the cosine similarity matrix.
    Inputs: i) the dataframe
            ii) the text column
            
    Outputs: the df_cos matrix and the tfidf matrix
    
    '''    
    
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix_insiders = tfidf_vectorizer.fit_transform(df[desc])

    cos_sim_array = cosine_similarity(tfidf_matrix_insiders, tfidf_matrix_insiders)

    df_cos = pd.DataFrame(cos_sim_array, columns = df.index,\
                          index = df.index )
    
    return df_cos, tfidf_matrix_insiders
######################################################################################



########################################################################################
# return the similarity matrix (based on cosine distance)
def tfidf_df(df, desc):
    '''
    Return the similarity matrix (based on cosine distance)
    Inputs: i) the dataframe
            ii) the column containing text
    '''
    
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix_insiders = tfidf_vectorizer.fit_transform(df[desc])

   
    return tfidf_matrix_insiders
########################################################################################










########################################################################
def remaining(files_list, control_file_fullpath):
    '''
    Look for the files remaining to be processed.
    Input: the input path and the file where the processed filenames
    are stored (control_file_fullpath).
    Output: list of files to be processed
    '''
   
    
    if os.path.isfile(control_file_fullpath) == True:
        processed_files_list = pd.read_table(control_file_fullpath, header = None)[0].tolist()
        remaining_list = list(set(files_list) - set(processed_files_list))
    else:
        remaining_list = files_list[:]
    
    
    
    return remaining_list
################################################################################


###############################################################################
def tfidf_df(df, desc):
    
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix_insiders = tfidf_vectorizer.fit_transform(df[desc])

   
    return tfidf_matrix_insiders
###############################################################################



##################################################################################
def corpus_tool(dftext):
    '''
    Analysis a given corpus in a dataframe
    input: df['text']
    outputs: text_list, len_text_list, tokenized_text_list, lexical_diversity_per_document_list,\
    mean_tokens, std_tokens, tot_tokens, unique_tokens_list,\
    tot_unique_tokens, lexical_diversity
    '''


    print ('>> Reading document list... ')
    print (' ')
    print (' ')
    text_list = dftext.tolist()
    
   
    len_text_list = len(text_list)
    print ('>> Document counting: ', len_text_list)
    print (' ')
    print (' ')
    
    tokenized_text_list = []
    for k in range(0, len_text_list):
        try:
            tokenized_text_list.append(text_list[k].split())
        except:
            #print text_list[k]
            continue
    
    
    tokens_per_document_list = []
    unique_tokens_per_document_list = []
    for k in range(0, len(tokenized_text_list)):
        tokens_per_document_list.append(float(len(tokenized_text_list[k])))
        unique_tokens_per_document_list.append(float(len(list(set(tokenized_text_list[k])))))
        
    mean_tokens = np.mean(tokens_per_document_list)
    std_tokens = np.std(tokens_per_document_list)
    
    print ('>> Average words per document: ', mean_tokens)
    print ('>> Standard deviation of words per document: ', std_tokens)
    print (' ')
    print (' ')
    
    lexical_diversity_per_document_list = []
    for k in range(0, len(tokens_per_document_list)):
        ld_document = unique_tokens_per_document_list[k]/tokens_per_document_list[k]
        lexical_diversity_per_document_list.append(ld_document)
    
    
    mean_ld = np.mean(lexical_diversity_per_document_list)
    std_ld = np.std(lexical_diversity_per_document_list)
    
    print ('>> Average lexical diversity: ', mean_ld)
    print ('>> Standard deviation of lexical diversity: ', std_ld)
    print (' ')
    print (' ')    
    
    
    tokens_list = list(itertools.chain(*tokenized_text_list))
    
    tot_tokens = len(tokens_list)
    print ('>> Total words in corpus: ', tot_tokens)
    print (' ')
    print (' ')
    
    unique_tokens_list = list(set(tokens_list))
    
    tot_unique_tokens = len(unique_tokens_list)
    print ('>> Unique words counting: ', tot_unique_tokens)
    print (' ')
    print (' ')
    
    
    lexical_diversity = float(tot_unique_tokens)/float(tot_tokens)
    print ('>> Global lexical diversity: ', lexical_diversity)
    print (' ')
    print (' ')
    
    
    
    return text_list, len_text_list, tokenized_text_list, lexical_diversity_per_document_list,\
           mean_tokens, std_tokens, tot_tokens, unique_tokens_list,\
           tot_unique_tokens, lexical_diversity
#################################################################################################################







###############################################################################
def representative(dftext, nwords):
    '''
    This function creates a representative document composed by the most representative words of the
    corpus, based on the tf-idf analysis.
    Inputs: i) df['text], ii) nwords: number of words to be considered.
    Output: i) representative_doc, ii) df_top_tfidf
    '''


    ## definition of the corpus
    text_corpus = dftext.tolist()
    
    ## vectorizing the corpus
    cvec = CountVectorizer()
    cvec.fit(text_corpus)
    
    
    ## counting the words in each document
    cvec_counts = cvec.transform(text_corpus)
    
    ## computing the df-idf matrix
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(cvec_counts)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    
    ## sending the results to a pandas dataframe
    df_weights = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    
    ## getting the top n = nwords words:
    df_top_tfidf = df_weights.sort_values(by='weight', ascending=False).head(nwords).reset_index(drop=True)
    
    ## finally, the representative document:
    representative_doc =  ' '.join(df_top_tfidf['term'].tolist())
    
    return representative_doc, df_top_tfidf
######################################################################################




########################################################################################
def cloud(word_list, title):
    '''
    Word cloud.
    Input: i) words list, ii) title
    Output: plot of the wordcloud
    '''

 

    #d = path.dirname(file_path)

    ### Read the whole text.
    #text = open(path.join(d, text_file)).read()
    text_str = " ".join(str(x) for x in word_list)
    ### Generate a word cloud image
    wordcloud = WordCloud().generate(text_str)



    ### take relative word frequencies into account, lower max_font_size
    #wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
    #wordcloud = WordCloud(width=400, height=200, max_font_size=40, relative_scaling=.5).generate(text)
    wordcloud = WordCloud(width=1600, height=800 , max_font_size=500).generate(text_str)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title, fontsize = 40)
    plt.show()
    #plt.savefig(figout_relative,facecolor='k', bbox_inches='tight')
############################################################################################




##############################################################################################
def TimeOrderedFiles(path_w2v):
    '''
    Input: a path to a directory
    Output: a list of files of the directory ordered by date
    '''

    files_list = glob.glob(os.path.join(path_w2v, '*'))
    files_list.sort(key=os.path.getmtime, reverse=True)
    files_list = [item.split('/')[-1] for item in files_list]
    
    
    return files_list
##############################################################################################


############################################################################################
def DateConv(date_str):
    '''
    Input: 'Y-m-d H:M:S'
    '''
    
    ## datetime object
    date_object = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

    ## timestamp
    ts = datetime.timestamp(date_object) 
    
    
    return ts, date_object
#############################################################################################


##################################################################################
def language(input_text):
    '''
    Detect the language of an input text
    Input: text
    Output: the language of the text (string)
    '''
    
    try:
        lang = detect(input_text)
    except:
        lang = ''
        logging.warning('Language not found.')
        print('Language not found')
        
    return lang
#####################################################################################



####################################################################################
def showcolors():
    '''
    List pre-defined colors in mcolors
    output: a list of colors name
    '''
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    sorted_names = [item for item in sorted_names if len(item) > 1]

    return sorted_names
#####################################################################################