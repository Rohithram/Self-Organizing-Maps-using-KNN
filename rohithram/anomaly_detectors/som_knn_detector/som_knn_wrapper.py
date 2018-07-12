
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import datetime as dt
import time
import os


# Importing dependency files
from anomaly_detectors.reader_writer import db_properties as db_props
from anomaly_detectors.reader_writer import writer_configs as write_args
from anomaly_detectors.utils.preprocessors import *
from anomaly_detectors.utils.data_handler import *
from anomaly_detectors.utils.error_codes import error_codes
from anomaly_detectors.utils import type_checker as type_checker
from anomaly_detectors.utils import csv_prep_for_reader as csv_helper
from anomaly_detectors.utils import reader_helper
from anomaly_detectors.utils import make_ackg_json
from anomaly_detectors.som_knn_detector import som_knn_detector as som_detector
from anomaly_detectors.som_knn_detector import som_knn_module as som_model


import traceback


import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 12, 9
rcParams[ 'axes.grid']=True


ideal_train_kwargs_type  = {
            'som_shape':tuple,
            'input_feature_size':int,
            'time_constant':float,
            'minNumPerBmu':int,
            'no_of_neighbors':int,
            'initial_radius':float,
            'initial_learning_rate':float,
            'n_iterations':int,
            'N':int,    
            'diff_order':int,
            'is_train':bool,
            'epochs':int,
            'batch_size':int,
            'to_plot':bool,
            'test_frac':float
        }


ideal_eval_kwargs_type = {
            'model_path':str,
            'to_plot':bool,
            'anom_thres':float
        }


mode_options = ['detect only','detect and log','log only']

def train(json_data,network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,
          no_of_neighbours=10,init_radius=None,init_learning_rate=0.01,N=100,diff_order=1,is_train=True
          ,epochs=4,batch_size=4,to_plot=True,test_frac=0.5):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in json_string  and parses json 
                            and gives list of dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class som_knn_detector defined in som_knn_detector.py, which takes in
                            data and algorithm parameters as argument and trains and saves the model and returns
                            the model file path where it saved
        *make_acknowledgement_json - Its function to Make acknowlegement json imported from make_ackg_json.py
        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries, db_properties and table name as args and gives out response code.
        
        Arguments :
        Required Parameter:
            json_data: The Json object in the format of the input json given from reader api
            
        Optional Parameter 
                mode : mode has 3 options -> 'detect only','detect and log' , 'log only'
                    Default: 'detect only'
                network_shape : (Type : Tuple (x,y) ) where x is no of rows of the grid of neuron layer and y is the no of columns of grid. So Total no of neurons in a single layer is (x*y)
                    Default: (8,8)
                input_feature_size: Positive Integer representing the no of features in the input data for which anomaly to be detected 
                    Default: Will be no of metric's given as the input , For ex: For two metrics given the feature size will be taken as 2 since this is a multivariate algorithm
                    Customised input : Give no of features wanted to be extracted per metric (yet to do)
                    Note: (Do not give unrelated metrics together in input data , since all metrics are analyzed together i.e Multivariate)

                time_constant: positive float, Exponential decay factor to decrease the neighborhood radius around BMU
                    Default: n_iterations/(log(init_radius)) , It's calculated in the program
                
                minNumPerBmu: positive integer , It is a minimum no of BMU hits for a neuron. Used to minimise the effect of noise in the data
                    Default : 3
                no_of_neighbors: positive integer , It is no of neighbors for KNN algorithm.
                    Default: 3
                initial_radius : positive float, initial radius to find the group of neurons around each BMU
                    Default: 0.4
                initial_learning_rate : positive float , It is learning rate for the algo
                    Default  : 0.01
                diff_order : positive integer, It is order of differencing to be done on the raw data 
                    Default : 0 
                    Note : use 1 or more for mean shift dataset
                epochs: positive integer , no of epochs to train
                    Default: 4
                batch_size : positive integer, no of samples in data to be processed simultaneously
                    Default: 4
                test_frac: positive float, Ratio of test : train data
                    Default : 0.2
                to_plot : Boolean .Give True to see the plots of change-points detected and False if there is no need for plotting
                    Default : True
                
                
        '''
        
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':True,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
        error_codes1 = error_codes()
              
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_train_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return json.dumps(res)
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(json_data=json_data)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data is not None and type(entire_data)!=dict)):
            
                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''
                
                # Output Format for training the model
                #models: is an list of filepaths where model saved
                out_json = {'header':'','models':[]}

                for i,data_per_asset in enumerate(entire_data):
                    if(len(data_per_asset)!=0):
                        assetno = pd.unique(data_per_asset['assetno'])[0]
                        data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                     )


                        print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                        cols = list(data_per_asset.columns[1:])

                        anomaly_detector = som_detector.Som_Detector(data = data_per_asset,model_input_args=model_input_args,
                                                                     training_args=training_args,
                                                                    eval_args=None)

                        model_path = (anomaly_detector.detect_anomalies())

                        model = {anomaly_detector.assetno:model_path}
                        
                        '''
                        TODO : Add code for saving the model into database here 
                        '''


                        out_json['models'].append(model)
                out_json['header'] = error_codes1['success']
                
                
               
                return json.dumps(out_json)
            elif(type(entire_data)==dict):
                return json.dumps(entire_data)
            else:
                '''
                Data empty error
                '''
                return json.dumps(error_codes1['data_missing'])
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes1['unknown']['message']=str(e)
            return json.dumps(error_codes1['unknown'])

def evaluate(json_data,model_path,mode=mode_options[0],to_plot=True,anom_thres=3.0):

    
        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in json_string  and parses json 
                            and gives list of dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class som_knn_detector defined in som_knn_detector.py, which takes in
                            data and algorithm parameters as argument and evaluates the data using the choosen model
                            and returns anomaly_indexes.      
        * make_acknowledgement_json - Its function to Make acknowlegement json imported from make_ackg_json.py
        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        
        Required Parameter:
            json_data: The Json object in the format of the input json given from reader api to evaluate the model
            model_path : Saved model file path in (string) format
        Optional Parameters: 
            mode - mode has 3 options 'detect only','detect and log','log only'
                Default: 'detect only'
            anom_thres : (Type : Positive float ) Anomaly threshold, used on anomaly scores estimated using K nearest neighbours on BMU of input test sample
                Default: 3.0
            to_plot : Boolean .Give True to see the plots of change-points detected and False if there is no need for plotting
                Default : True

        '''
        
        
        eval_args = {
            'model_path':model_path,
            'to_plot':to_plot,
            'anom_thres':anom_thres
        }
                
        error_codes1 = error_codes()
            
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=eval_args,ideal_args_type=ideal_eval_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return json.dumps(res)
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(json_data=json_data)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):
                
                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''
                    
                for i,data_per_asset in enumerate(entire_data):
                    if(len(data_per_asset)!=0):
                        assetno = pd.unique(data_per_asset['assetno'])[0]
                        data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                     )

                        print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))

                        anomaly_detector = som_detector.Som_Detector(data = data_per_asset,model_input_args=model_input_args,
                                                                     training_args=None,eval_args=eval_args)

                        anom_indexes = anomaly_detector.detect_anomalies()
                        anomaly_detectors.append(anomaly_detector)
                    
                        
                ack_json = {}
                
                if(mode==mode_options[0] or mode==mode_options[1]):
                    ack_json = make_ackg_json.make_ack_json(anomaly_detectors)
                if(mode==mode_options[1] or mode==mode_options[2]):
                    
                    '''
                    Instantiates writer class to write into local database with arguments given below
                    Used for Bulk writing
                    '''
                    sql_query_args = write_args.writer_kwargs
                    table_name = write_args.table_name
                    window_size = 10

                    writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                            table_name=table_name,window_size=window_size)

                    #called for mapping args before writing into db
                    res = writer.map_outputs_and_write()
                    if(res!=error_codes1['success']):
                        return json.dumps(res)
                    
                    if(bool(ack_json)==False):
                        ack_json['header'] = error_codes1['success']
                        
                return json.dumps(ack_json)
                
                
            else:
                '''
                Data empty error
                '''
                return json.dumps(error_codes1['data_missing'])
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes1['unknown']['message']=str(e)
            return json.dumps(error_codes1['unknown'])

reader_kwargs= lambda:{
            'assetno':['TSFAD_A1'],
            'from_timestamp':'',
            'to_timestamp':'',
            'con':'',
            'para_list':'',
            'source_type':'',
            'table_name':'',
            'qry_str':'',
            'impute_fill_method':'forward',
            'down_sampling_method':None,
            'down_sampling_window':None,
            'freq':None,
            'resample_fill_method':None,
            'to_resample':None,
            'to_impute':True,
}

model_input_args = lambda :{
    'network_shape':(8,8),
    'input_feature_size':None,
    'time_constant':None,
    'minNumPerBmu':2,
    'no_of_neighbours':3,
    'init_radius':0.4,
    'init_learning_rate':0.01,
    'N':100,    
    'diff_order':1
}

training_args = lambda:{
            'is_train':True,
            'epochs':5,
            'batch_size':4,
            'to_plot':True,
            'test_frac':0.7
        }


        
eval_args = lambda: {
    'model_path':'',
    'to_plot':True,
    'anom_thres':3.0
}