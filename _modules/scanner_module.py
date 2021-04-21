from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from regex_module import FindLawsuit
from logging_module import getlogger


## log
logging = getlogger()

##########################################################
def scanner(path_doc):
    '''
    Converts a pdf document into string
    Inputs: i) path_doc: the path to the doc
            ii) pages_limit: the maximum number of pages to retrieve
    Output: a list of pages converted to string
    '''

    exists = os.path.exists(path_doc)

    if exists == True:
        logging.warning('File exists.')
        ext = path_doc.split('.')[-1]


        if ext == 'pdf':
            pages_list = convert_from_path(path_doc)

            converted_list = []
            for page in pages_list:
                result = pytesseract.image_to_string(page, lang = 'por')
                if result != '':
                    converted_list.append(result)


        else:

            try:
                pages_list = [cv2.imread(path_doc)]

                converted_list = []
                for page in pages_list:
                    result = pytesseract.image_to_string(page, lang = 'por')
                    if result != '':
                        converted_list.append(result)
            except:
                print('Image format not supported.')
                logging.warning('Image format not supported.')
                converted_list = []

    else:
        print('File do not exist.')
        logging.warning('File do not exist.')
        converted_list = []


    return converted_list
###############################################################



########################################################
def decode(image_path): 
    '''
    Find barcodes and QR codes
    '''

    ext = image_path.split('.')[-1]
    
    
    if ext == 'pdf':

        try:
            pages_list = convert_from_path(image_path)
            
            data_list = []
            for page in pages_list:
                decodedObjects = pyzbar.decode(pages_list[0])


                nobjects = len(decodedObjects)
                
                for item in decodedObjects:
                    data_list.append(item.data)
        except Exception as e:
            print('This is not a pdf file. {}'.format(e))
            logging.warning('This is not a pdf file.')
            data_list = []
            
            
    else:
        try:
            im = cv2.imread(image_path)
            decodedObjects = pyzbar.decode(im)

            nobjects = len(decodedObjects)

            data_list = []
            for item in decodedObjects:
                data_list.append(item.data)
        except:
            print('File extension not supported.')
            logging.warning('File extension not supported.')
            data_list = []

    
    

    return data_list
#######################################################







