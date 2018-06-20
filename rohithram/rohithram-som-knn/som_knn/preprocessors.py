

import numpy as np
import pandas as pd
import datetime as dt
from sklearn import preprocessing 

def to_timestamp(dataframe,date_col_index,time_format='%Y-%m',isweek=False):
    '''
    Converts any string datetime object to pandas datetime
    Gets dataframe and date_column index as required args
    timeformat is required for rare timeformats like weekly data
    isweek is bool type which is False for non weekly data
    '''
    if(isweek!=True):
            dateparse = lambda dates: pd.to_datetime(dates,infer_datetime_format=True)
    else:
        dateparse = lambda dates: dt.datetime.strptime(dates+'-0', time_format)
    dataframe[date_col_index].apply(dateparse)
    return dataframe

def ts_to_unix(t):
    '''
    Converts datetime to epoch timestamps
    Arguments:
    single datetime object
    '''
    return int((t - dt.datetime(1970, 1, 1)).total_seconds()*1000)

def normalise_standardise(data):    
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    # Create an object to transform the data to fit minmax processor
    data_norm = pd.DataFrame(min_max_scaler.fit_transform(data.values),
                             columns=data.columns,index=data.index)
    data_standardised = (data_norm - data_norm.mean(axis=0))/(data_norm.std(axis=0))
    return data_standardised

def split_the_data(data,test_frac=0.1):
    '''
    Splitting the data into train and test with default ratio = 0.1
    Splits the data in orderly manner not random
    '''
    train_data = data[0:int(np.ceil((1-test_frac)*data[:,].shape[0])),:]
    test_data = data[-int(np.ceil(test_frac*data[:,].shape[0])):]
    return train_data,test_data

def stationarize(data):
    '''
    Stationarises the data
    '''
    s,t = fit_seasons(data)

    if(s is not None):
        adj_sea = adjust_seasons(data,seasons=s)
        res_data = adj_sea-(data-detrend(data))
    else:
        res_data = detrend(data)
        
    return res_data

def differencing(data,n=1,axis=-1):
    '''
    Does differencing on the data and order of differentiation as parameter
    By default n=1 and axis =-1
    '''
    return np.diff(data,n=n,axis=axis)