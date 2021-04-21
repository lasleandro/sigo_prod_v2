import os
import gensim
from gensim.test.utils import common_texts
from gensim.test.utils import get_tmpfile
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phraser
import pandas as pd



###################################################################    
def load_ngrams_model(path_word2vec, model_name):
    
    '''
    Loads a ngram model trained using gensim word2vec.
    Inputs: i) path to the model folder and ii) model name
    Output: the ngramer model. Ex.: trigram[]
    '''
    
    model_ngrams = gensim.models.Phrases.load(os.path.join(path_word2vec, model_name))
    
    
    print('Loading model')
    ngramer = Phraser(model_ngrams)
    print('Success! Model loaded.')
    print(' ')
    
    return ngramer
####################################################################


#####################################################################
def load_w2v_embedding(path_word2vec, model_name):
    
    '''
    Loads an w2v embedding model.
    Inputs: i) path to the w2v weights, ii) model name
    Output: the embedding model. Ex: model.wv['foto']
    '''
    
    print('Loading model')
    ## loading model if already trained
    model_w2v = gensim.models.Word2Vec.load(os.path.join(path_word2vec,model_name))
    ## Normalizes the vectors in the word2vec class. 
    model_w2v.init_sims(replace=True)
    print('Success! Model loaded.')
    print(' ')
    
    
    return model_w2v
#########################################################################



########################################################################
def text2ngram(input_text, ngram_model):
    '''
    Converts an input text to the n-gram representation,
    given a model.
    Input: text string
    Output: text string
    '''
    
    
    split_text = input_text.split()
    new_tokens = ngram_model[split_text]
    final_text = ' '.join(new_tokens)
    
    return final_text
##########################################################################



###################################################################
def word_graph(seed_zero, nwords, model):

    '''
    A function to extract a graph from a dict obtained from the word2vec model
    Inputs: i) the starting word seed, ii) the total number of words to visit
    Outputs: i) words_dict, ii) nodes_list, iii) edges_list
    '''



    # getting the first neighbour of the sequence of words starting from the seed_zero:
    words_list = []
    words_list.append(seed_zero)
    words_dict = OrderedDict([(seed_zero, 1)])
    sim_word = seed_zero
    for k in range(0, nwords):
        # remember changing the topn value if nwords is very large
        sim_words_list = model.wv.most_similar(sim_word, topn=200)
        sim_word = sim_words_list[0][0]
        sim_strength = sim_words_list[0][1]


        count = 0
        while sim_word in words_list:
            count += 1
            sim_word = sim_words_list[count][0]
            sim_strength = sim_words_list[count][1]


        words_list.append(sim_word)
        words_dict.update({sim_word: sim_strength})
        
        
    
     #graph definitions:   
        
    words_ordered_list = words_dict.keys()
    
    nodes_list = words_ordered_list[:]
    
    edges_list = []
    for k in range(0, len(words_ordered_list)):
        try:
            word1 = words_ordered_list[k]
            word2 = words_ordered_list[k+1]
            edges_list.append((word1,word2, {'weight': 100*words_dict[word2]}))

        except:
            continue
    
        
        
    return words_dict, nodes_list, edges_list
###################################################################


##########################################################################
def wmd_eval(doc1, doc2, model):
    '''
    calculates the distance between documents using WMD (word mover's distance).
    Inputs: i) doc1 ii) doc2 e iii) w2v model
    '''
    dist = model.wmdistance(doc1.split(), doc2.split())
    return dist
#########################################################################


###########################################################################
def SimWord(input_string, nresults, TheModel):
    '''
    Uses an w2v model to return the most siilar words to a given token.
    Inputs: i) an string, ii) max number of results, iii) the w2v model
    Output: df containing the top n similar words
    '''
    results = TheModel.wv.most_similar(input_string, topn=nresults)
    df = pd.DataFrame(data = results, columns = ['word', 'score'])
    return df
############################################################################
