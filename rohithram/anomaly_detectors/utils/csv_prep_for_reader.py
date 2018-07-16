

import numpy as np
import pandas as pd
import datetime as dt
import os


def preparecsvtoread(filepath='../../dataset/sample_csv_files/alcohol-demand-log-spirits-consu.csv',
                     filename='alcohol-demand-log-spirits-consu.csv',
                     target_dir='../../dataset/reader_csv_files/',
                     assetno='A1',n_rows=None,has_time=True):
    
    '''
    Function which takes a filepath of csv file to read and processes it to standardise the csv file like 
    converts datetime object to epoch time format and adds assetno column to distinguish the data from different
    sources like sensors!
    
    Arguments :
        filepath: full filepath of the csv file to read from
        filename: only filename without any paths enclosed in quotes
        target_dir : path of the target directory in which csv file to be saved
        assetno: Asset no of the dataset (Needs to be unique for a given dataset)
        n_rows: Allows to read only first n_rows of the csv dataset
        has_time: If False, then it adds time_stamp column to the dataset
        
    
    Returns: 
    filepath of processed csv file which is ready to read from there and metric names( different column names)
    present in the dataset
    '''
    
    if(n_rows is not None):
        df  = pd.read_csv(filepath,nrows=n_rows)
    else:
        
        df = pd.read_csv(filepath)
        n_rows = df.shape[0]
        
    df['assetno'] = assetno
    
    if(has_time!=True):
        start = pd.Timestamp("19700807 08:30-0400")
        end = pd.Timestamp("20170807 17:30-0400")
        index = pd.DatetimeIndex(start=start, end=end, freq="10min")[:n_rows]
        df.insert(0,'timestamp', index) 
        df = df.dropna(axis=1, how='all')
    else:
        df = df.rename(columns={df.columns[0]:'timestamp'})
    df['timestamp'] = (pd.to_datetime(df['timestamp'],infer_datetime_format=True).astype(np.int64)/(1e6)).astype(np.int64)

    metric_names = df.columns[1:-1]
    target_filepath = os.path.join(target_dir,filename)
    df.to_csv(target_filepath,index=False)
    return target_filepath,list(metric_names)