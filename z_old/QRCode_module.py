import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pdf2image import convert_from_path
import os
import logging





########################################################
def decode(image_path): 
    '''
    Find barcodes and QR codes
    '''

    ext = image_path.split('.')[-1]
    
    
    if ext == 'pdf':
        pages_list = convert_from_path(image_path)
        
        data_list = []
        for page in pages_list:
            decodedObjects = pyzbar.decode(pages_list[0])

            # Print results
            for obj in decodedObjects:
                #print('Type : ', obj.type)
                #print('Data : ', obj.data,'\n')


            nobjects = len(decodedObjects)
            
            for item in decodedObjects:
                data_list.append(item.data)
            
            
    else:
        try:
            im = cv2.imread(image_path)
            decodedObjects = pyzbar.decode(im)

            # Print results
            for obj in decodedObjects:
                #print('Type : ', obj.type)
                #print('Data : ', obj.data,'\n')


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




