
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
from anomaly_detectors.utils import error_codes as error_codes
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
            'anom_thres':int
        }

mode_options = ['detect only','detect and log','log only']

def train(json_data,network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,
          no_of_neighbours=10,init_radius=None,init_learning_rate=0.01,N=100,diff_order=1,is_train=True
          ,epochs=4,batch_size=4,to_plot=True,test_frac=0.5):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
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
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
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
                
#                 model_paths = []
                out_json = {'header':'','models':[]}

                for i,data_per_asset in enumerate(entire_data):
                    assetno = pd.unique(data_per_asset['assetno'])[0]
#                     print(assetno)
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                eval_args=None)
                    
                    model_path = (anomaly_detector.detect_anomalies())
                    
                    model = {anomaly_detector.assetno:model_path[0]}
#                     table_name = write_args.table_name
#                     window_size = 10
#                     anomaly_detectors.append(anomaly_detector)
#                     sql_query_args = write_args.writer_kwargs
                    

                    out_json['models'].append(model)
        
                out_json['header'] = error_codes.error_codes['success']
                
#                 if(mode==mode_options[0] or mode==mode_options[1]):
#                     ack_json = make_ackg_json.make_ack_json(anomaly_detectors)
#                     out_json['detect_status'] = ack_json
#                 if(mode==mode_options[1] or mode==mode_options[2]):
#                     '''
#                     Instantiates writer class to write into local database with arguments given below
#                     Used for Bulk writing
#                     '''
#                     writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,
#                                              sql_query_args=sql_query_args,
#                                             table_name=table_name,window_size=window_size)

#                     #called for mapping args before writing into db
#                     res = writer.map_outputs_and_write()
#                     out_json['log_status']=res
               
                return json.dumps(out_json)
            elif(type(entire_data)==dict):
               return json.dumps(entire_data)
            else:
                '''
                Data empty error
                '''
                
                return json.dumps(error_codes.error_codes['data_missing'])
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=str(e)
            return json.dumps(error_codes.error_codes['unknown'])

def evaluate(json_data,model_path,mode=mode_options[0],to_plot=True,anom_thres=3):

    
        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        
        eval_args = {
            'model_path':model_path,
            'to_plot':to_plot,
            'anom_thres':anom_thres
        }
                
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
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
                    assetno = pd.unique(data_per_asset['assetno'])[0]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=None,metric_names=cols,eval_args=eval_args)
                    
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
                    if(res!=error_codes.error_codes['success']):
                        return json.dumps(res)
                    
                return json.dumps(ack_json)
                
                
            else:
                '''
                Data empty error
                '''
                return json.dumps(error_codes.error_codes['data_missing'])
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=str(e)
            return json.dumps(error_codes.error_codes['unknown'])

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
    'anom_thres':3
}