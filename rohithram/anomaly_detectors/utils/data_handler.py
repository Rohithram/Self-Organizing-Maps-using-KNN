

import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

import datetime as dt
# error code is python file which contains dictionary of mapped error codes and messages for different errors
from anomaly_detectors.utils.error_codes import error_codes

import warnings
warnings.filterwarnings('ignore')
class Data_reader():
    
    
    '''
    Data_reader is a class which contains methods which are used to fetch data from csv file.which converts 
    the resultant dataframe into list of dataframes per asset with different metric(features of data)
    being from column index 1, and assetno being column 0 (unique for one set of data) index being timestamps 
    in epoch format
    '''
    
    def __init__(self,filepath):
        
        #takes json data
        self.filepath = filepath
        print("Data reader initialised \n")

    def read(self):
        
        try:
            response_data = pd.read_csv(self.filepath)
            
        except Exception as e:
            error_codes1 = error_codes()
            error_codes1['param']['message'] = '{},{}'.format(str(e),str(self.filepath))
            return error_codes1['param']
        

        print("Getting the dataset from the reader....\n")
        entire_data = self.parse_dict_to_dataframe(response_data)
        
        return entire_data
    
    def parse_dict_to_dataframe(self,data):
        
        '''
        parses the response data from csv file to list of dataframes per asset and metrics being columns of
        each of the dataframe with timestamps being the index and first column is assetno
        Arguments: data: response dataframe
        Returns -> List of dataframes
        '''
        
        entire_data_set = []
        
        #making the index of the dataframe to be index and deleting the timestamp column
        data.index = data['timestamp']
        del data['timestamp']
        cols = list(data.columns)
        del cols[cols.index('assetno')]
        cols.insert(0,'assetno')
        
        data = data[cols]
        #separating the dataframe into groups of distinct assets
        data_per_assets = data.groupby('assetno')

        #creating list of dataframes of different assetno and with all metrics being columns in each dataframe
        for name,group in data_per_assets:
            entire_data_set.append(group)
        
        
        return entire_data_set