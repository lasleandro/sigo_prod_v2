import sys
import pandas as pd
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import re
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import statsmodels.api as sm


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import sklearn.cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min


import networkx as nx

from scipy.optimize import curve_fit
#from scipy.misc import factorial
import scipy
from scipy.stats import lognorm
from scipy.stats import powerlaw
from scipy.stats import pareto
from scipy.stats import beta
from wordcloud import WordCloud
import os
from datetime import datetime
from joblib import dump, load

plt.style.use('ggplot')




############################################################################################################
def graph(matching_idx_pictures_list):

    '''
    Desenha um grafo dada a lista de nós. Retorna ainda os ciclos presentes no
    grafo e gera um .html com o grafo.
    usa como input o output da funcao "sim_threshold".    
    '''
    
    ## defining the node list:
    nodes_list = list(set(list(itertools.chain(*matching_idx_pictures_list))))
    
    ## creating the graph:
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(matching_idx_pictures_list)  
    
    ## getting all the cycles:
    cycles_list = nx.cycle_basis(G.to_undirected())    

    
    
    all_node_list = []
    for node in nodes_list:
        neighbors_list = list(nx.all_neighbors(G, node))
        if len(neighbors_list) == 1:
            if node not in all_node_list:
                #print(node)
                temp_list = list(nx.all_neighbors(G, node))
                temp_list.append(node)
                #print(temp_list)
                for item in temp_list:
                    all_node_list.append(item)
                #print(all_node_list)
                #print(' ')
                #print(' ')    
    
    
    
    ## exporting to pyvis
    #H = Network()
    #H = Network(notebook=True)
    #H.from_nx(G)
    #H.show_buttons(filter_=['physics'])
    #H.show("simgraph.html")    
    

    #paths = nx.all_simple_paths(G, source=0, target=3, cutoff=2)
    cliques_list = list(nx.find_cliques(G))


    ## plotting the graph:
    plt.figure(figsize=[15,10])
    options = {'node_color': 'blue',
    'node_size': 400,
    'width': 0.5,
    'label_color':'w'}
    nx.draw_spring(G, with_labels=True, font_weight='bold', edge_color = 'b', **options)    
    


    #all_connected_subgraphs = []
    # here we ask for all connected subgraphs that have at least 2 nodes AND have less nodes than the input graph
    #for nb_nodes in range(2, G.number_of_nodes()):
    #    for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
    #        if nx.is_connected(SG):
    #            print(SG.nodes)
    #            all_connected_subgraphs.append(SG)




    
    return cycles_list, all_node_list, cliques_list, G
###################################################################    
    
    
    
###################################################################
def sim_threshold(df, threshold):
    '''
    Input the df_full and collect the pair (row, column) whose similarity is above a given threshold
    similarity \in [-1, 1].
    '''
    
    ## getting all index
    all_idx_list = list(df.index)
    
    ## locking the dataframe to the index list defined above
    df_sim = df.loc[all_idx_list, all_idx_list]
    
    ## filtering the pairs of pictures with similarity above the specified threshold
    passed_list = list(df_sim[df_sim >= threshold].stack().index)
    
    matching_idx_pictures_list = [(item[0], item[1]) for item in \
                                  passed_list if item[0] != item[1] and item[0] < item[1]]
    
    return matching_idx_pictures_list

########################################################################


#######################################################################
def sim_hist(df_sim):
    '''
    Generates the statistics of a given similarity (or other metric) dataframe
    '''
       
    sim_list = []
    for column in df_sim.columns:
        sim_list.append(df_sim[column].tolist())

    merged_sim_list = list(itertools.chain(*sim_list))


    mean = np.mean(merged_sim_list)
    std = np.std(merged_sim_list)
    CV = std/mean

    q1 = np.quantile(merged_sim_list, 0.25)
    q2 = np.quantile(merged_sim_list, 0.50)
    q3 = np.quantile(merged_sim_list, 0.75)


    results_dict = {'mean':mean, 'std':std, 'CV':CV, 'q1':q1, 'q2':q2, 'q3':q3}  
    
    plt.figure(figsize = [10,5])
    plt.title('Similarity distribution', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.hist(merged_sim_list, bins = 50, alpha = 0.5)    
    
    return results_dict
################################################################################


##################################################################################
def cluster_func(df, column, nclusters, path_cluster, path_output, date_tag, save = True):
    '''
    Given a dataframe, a text column and the number of clusters, this function
    clusterizes the text data.
    Inputs: i) df, ii) target text column, iii) the number of clusters
    Output: i) df with new 'cluster' column, ii) the model and iii) the vectorizer
    '''

    
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
    
    
    model = sklearn.cluster.KMeans(n_clusters = nclusters)
    
    
    model.fit(tfidf_matrix)
    labels = model.predict(tfidf_matrix)
    centers = np.array(model.cluster_centers_)
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, tfidf_matrix)
    
    silhouette_avg = silhouette_score(tfidf_matrix, labels)




    df['cluster'] = labels
    
    

    
    
    if save == True:

        dump(model, os.path.join(path_cluster, 'model_clustering_{}_{}.joblib'.format(nclusters, date_tag)))
        dump(tfidf_vectorizer, os.path.join(path_cluster, 'vectorizer_{}_{}.joblib'.format(nclusters, date_tag)))

        df.to_csv(os.path.join(path_output, 'tmail_data_clean_bigrams_cluster_{}_{}.csv'.format(nclusters, date_tag)), sep = '\t',\
                encoding = 'utf-8', index = False)
        
    
    return df, model, tfidf_vectorizer, silhouette_avg,  tfidf_matrix, labels, closest, centers



###################################################################################


####################################################################################
def cluster_predict(new_intent_text, model_clustering, tfidf_vectorizer):
    '''
    Given the grouped text of a new intent, predicts the cluster this intent belongs to.
    Inputs: i) new_intent_text, ii) the clustering model, iii) the vectorizer
    Output: the predicted cluster
    '''
    
    test_list = [new_intent_text]
    test_vec = tfidf_vectorizer.transform(test_list)
    
    cluster_predicted = model_clustering.predict(test_vec)
    
    return cluster_predicted[0]
####################################################################################
    

######################################################################
def sim_df(df, text_column, index_choice = 'idx'):
    '''
    Evaluates the similarity matrix/dataframe of a given text data.
    Inputs: i) the dataframe
            ii) text_column: the column whose content is to be contrasted
            iii) index_choice: the name of the column to be considered the reference
                               in the similarity matrix/df
    Output: the similarity matrix/df 
    
    '''
    
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    
    matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    
    df_sim = pd.DataFrame(matrix)

    if index_choice == 'idx':
        df_sim.columns = df.index
        df_sim.index = df.index
    else:
        df_sim.columns = df[index_choice].tolist()
        df_sim.index = df[index_choice].tolist()
    
    
    return df_sim
#########################################################################

##################################################################################
def knn_func(df, column, nn, path_cluster, path_output, date_tag):
    '''
    Given a dataframe, a text column and the number of clusters, this function
    clusterizes the text data.
    Inputs: i) df, ii) target text column, iii) the number of clusters
    Output: i) df with new 'cluster' column, ii) the model and iii) the vectorizer
    '''

    ## vectorizing
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
    
    
    ## creating model
    model = NearestNeighbors(n_neighbors = nn, algorithm = 'auto').fit(tfidf_matrix)
    
    ## model outputs
    distances, indices = model.kneighbors(tfidf_matrix)    
    
    
    ## dumping in a dataframe
    df_dist = pd.DataFrame(data = distances)
    
    
    ## creating the distance columns
    dist_columns_list = ['itself'] + [ 'dist_n' + str(k) for k in range(1,nn)]
    df_dist.columns = dist_columns_list
    
    
    ## creating the cosine similarity matrix
    df_cos = sim_df(df, 'clean_text_trigram')
    prod_pair_list = zip(range(0, len(df_cos.index)), df_cos.index)
    prod_pair_dict = dict(prod_pair_list)
    
    ## giving the names to the neighbours:
    df_index = pd.DataFrame(data = indices)
    prod_columns_list = ['intent'] + [ 'n' + str(k) for k in range(1,nn)]
    
    df_index.columns = prod_columns_list
    
    
    df_index = df_index.replace(prod_pair_dict)

    
    ## saving to file:
    df_index.to_csv(os.path.join(path_output, 'intents_knn_{}.csv'.format(date_tag)),\
                    sep = '\t', encoding = 'utf-8', index = False)
    
    
    return df_index, model
######################################################################################






##################################################################################
def cluster_func_dbscan(df, column, eps_value, path_cluster, path_output, date_tag):
    '''
    Given a dataframe, a text column and the number of clusters, this function
    clusterizes the text data.
    Inputs: i) df, ii) target text column, iii) the number of clusters
    Output: i) df with new 'cluster' column, ii) the model and iii) the vectorizer
    '''

    
    tfidf_vectorizer =  TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
    
    
    model = DBSCAN(eps = eps_value, min_samples = 3)
    
    
    model.fit(tfidf_matrix)
    labels = model.labels_
    
    df['cluster'] = labels
    
    
    dump(model, os.path.join(path_cluster, 'model_clustering_dbscan_{}.joblib'.format(date_tag)))
    dump(tfidf_vectorizer, os.path.join(path_cluster, 'vectorizer_dbscan_{}.joblib'.format(date_tag)))
    
    
    df.to_csv(os.path.join(path_output, 'text_clusters_dbscan_{}.csv'.format(date_tag)), sep = '\t',\
             encoding = 'utf-8', index = False)
    
    
    return df, model, tfidf_vectorizer



###################################################################################







#####################################################################################
def OptimalNclusters(kmax, df, column, path_cluster, path_output, date_tag, save = True):
    '''
    Finds the optimal number of clusters using the silhouette method.
    Inputs: i) kmax: the max number of clusters to try
            ii)  the data frame
            iii) the target column in the dataframe
    Output: i) the silhouete average for each cluster number choice
            ii) the optimal number of clusters (the number of clusters that maximizes the average silhouette)
    '''
    
    silhouette_dict = {}
    for k in range(2, kmax):
        print(k)
        results = cluster_func(df, column, k, path_cluster, path_output, date_tag, save)
        silhouette_dict[k] = results[3]
        
    max_list = [key for key,value in silhouette_dict.items() if value == np.max(list(silhouette_dict.values()))]
    optimal_clusters = max_list[0]
        
    return silhouette_dict, optimal_clusters
############################################################################################






##########################################################################################################
def elbow(df, cutoff = 50):
    
    '''
    Receives a df containing data to be clusterized (ex df containing tfidf data)
    Input: df, cuttoff (maximum number of clusters to test)
    Output: i) df_results (columns = ['nclusters', 'sum_squared_distances', 'silhouette'])
            ii) optimal number of clusters
    
    
    '''
    
    
    tot_df = len(df)
    sum_squared_distances = []
    silhouette_list = []
    nclusters_list = range(2,cutoff)
    tot_steps = len(nclusters_list)
    
    df_results = pd.DataFrame(columns = ['nclusters', 'sum_squared_distances', 'silhouette'])
    
    
        
    
    idx = 0
    for k in nclusters_list:
        print ('Step {} of {} steps'.format(idx+1, tot_steps))
        model = KMeans(n_clusters = k)
        #model = sklearn.cluster.KMeans(n_clusters = k, random_state = 170)
        model.fit(df)
        
        
        ## elbow
        ssd = model.inertia_
        sum_squared_distances.append(ssd)
        
        df_results.at[idx, 'nclusters'] = k
        df_results.at[idx, 'sum_squared_distances'] = ssd
        
        
        
        ## silhouette
        labels = model.labels_
        sil = metrics.silhouette_score(df, labels, metric='euclidean')
        silhouette_list.append(sil)
        df_results.at[idx, 'silhouette'] = sil
        
        idx += 1
        
    
    # Plot the elbow
    plt.figure(figsize = [15,5])
    plt.plot(nclusters_list, sum_squared_distances, 'bx-')
    plt.xlabel('n clusters')
    plt.ylabel('squared distance')
    plt.title('The Elbow Method')
    plt.show()    
    
    
    # Plot the silhouette
    plt.figure(figsize = [15,5])
    plt.plot(nclusters_list, silhouette_list, 'bx-', color = 'r')
    plt.xlabel('n clusters')
    plt.ylabel('silhouette paramater')
    plt.title('The Silhouette Method')
    plt.show()     
    
    
    optimal_nclusters = df_results[df_results['silhouette'] == df_results['silhouette']\
                                   .max()]['nclusters'].values[0]
    
    
    return df_results, optimal_nclusters
########################################################################################

