import os



###############################################################
def control_dir():
    '''
    Defines the path to the collect control folder.
    '''

    cwd = os.getcwd()
    the_path = os.path.join(cwd, '_collect_control')
    if os.path.isdir(the_path) == False:
        os.mkdir(the_path)
    else:
        print('Folder already exists.')

    print(the_path)


    return the_path
##################################################################



###############################################################
def unigrams_dir():
    '''
    Defines the path to the collect control folder.
    '''

    cwd = os.getcwd()
    the_path = os.path.join(cwd, '_text_unigrams')
    if os.path.isdir(the_path) == False:
        os.mkdir(the_path)
    else:
        print('Folder already exists.')

    print(the_path)


    return the_path
##################################################################


###############################################################
def bigrams_dir():
    '''
    Defines the path to the collect control folder.
    '''

    cwd = os.getcwd()
    the_path = os.path.join(cwd, '_text_bigrams')
    if os.path.isdir(the_path) == False:
        os.mkdir(the_path)
    else:
        print('Folder already exists.')

    print(the_path)


    return the_path
##################################################################

###############################################################
def w2v_dir():
    '''
    Defines the path to the collect control folder.
    '''

    cwd = os.getcwd()
    the_path = os.path.join(cwd, '_w2v_data')
    if os.path.isdir(the_path) == False:
        os.mkdir(the_path)
    else:
        print('Folder already exists.')

    print(the_path)


    return the_path
##################################################################


###############################################################
def cluster_dir():
    '''
    Defines the path to the collect control folder.
    '''

    cwd = os.getcwd()
    the_path = os.path.join(cwd, '_cluster_data')
    if os.path.isdir(the_path) == False:
        os.mkdir(the_path)
    else:
        print('Folder already exists.')

    print(the_path)


    return the_path
##################################################################


###############################################################
def output_dir():
    '''
    Defines the path to the collect control folder.
    '''

    cwd = os.getcwd()
    the_path = os.path.join(cwd, '_data_output')
    if os.path.isdir(the_path) == False:
        os.mkdir(the_path)
    else:
        print('Folder already exists.')

    print(the_path)


    return the_path
##################################################################




###############################################################
def att_dir():
    '''
    Defines the path to the collected attachments.
    '''


    cwd = os.getcwd()
    the_path = os.path.join(cwd, '_attachments')
    if os.path.isdir(the_path) == False:
        os.mkdir(the_path)
    else:
        print('Folder already exists.')

    print(the_path)


    return the_path
##################################################################