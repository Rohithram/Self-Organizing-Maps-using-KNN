
'''
importing all the required header files
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import datetime as dt

import time
import os
# Importing reader and checker python files as modules
import reader_writer.db_properties as db_props
import reader_writer.writer_configs as write_args
import psycopg2

from utils.preprocessors import *
from utils.data_handler import *
from utils.bayesian_changept_detector import *
# from utils.error_codes import error_codes
import utils.error_codes as error_codes
import type_checker as type_checker
import json
from pandas.io.json import json_normalize
import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 12, 9
rcParams[ 'axes.grid']=True

algo_params_type ={
            'is_train':bool,
            'data_col_index':int,
            'pthres':float,
            'Nw':int,
            'mean_runlen':int
        }

def call(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,thres_prob=0.5,samples_to_wait=10,expected_run_length=100):

        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }

        algo_kwargs={
            'is_train':False,
            'data_col_index':1,
            'pthres':thres_prob,
            'Nw':samples_to_wait,
            'mean_runlen':expected_run_length
        }
                    
        try: 
            error_codes.reset()
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=algo_params_type)
            res = checker.params_checker()
            if(res!=None):
                return res
            
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
                
            entire_data = data_reader.read()

            writer_data = []
            anomaly_detectors = []
            if(len(entire_data)!=0 and entire_data!=None):
                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]])
                    
                    for data_col in range(1,len(data_per_asset.columns[1:])+1):
                        algo_kwargs['data_col_index'] = data_col
                        print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
                        anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
                        data,anom_indexes = anomaly_detector.detect_anomalies()

                        sql_query_args = write_args.writer_kwargs
                        table_name = write_args.table_name
                        window_size = 10

                        anomaly_detectors.append(anomaly_detector)
                    

                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                res = writer.map_outputs_and_write()
                return res
            else:
                return error_codes.error_codes['data_missing']
        except Exception as e:
#             error_message['message'] = e
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']