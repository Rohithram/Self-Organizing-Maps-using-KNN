

import numpy as np
import pandas as pd
import datetime as dt
import os


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


def preparecsvtoread(filepath,filename,target_dir,assetno='TSFAD_A1',n_rows=None,has_time=True):
    if(n_rows is not None):
        df  = pd.read_csv(filepath,nrows=n_rows)
    else:
        
        df = pd.read_csv(filepath)
        n_rows = df.shape[0]
        
    df['assetno'] = assetno
#     print(df.head())
    if(has_time!=True):
        start = pd.Timestamp("19700807 08:30-0400")
        end = pd.Timestamp("20170807 17:30-0400")
        index = pd.DatetimeIndex(start=start, end=end, freq="10min")[:n_rows]
        df.insert(0,'timestamp', index) 
        df = df.dropna(axis=1, how='all')
#         print(df.head())
    else:
        df = df.rename(columns={df.columns[0]:'timestamp'})
    df['timestamp'] = (pd.to_datetime(df['timestamp'],infer_datetime_format=True).astype(np.int64)/(1e6)).astype(np.int64)
#     print(df.head())
    metric_names = df.columns[1:-1]
    target_filepath = os.path.join(target_dir,filename)
    df.to_csv(target_filepath,index=False)
    return target_filepath,list(metric_names)

def get_csv_kwargs(infile='../../dataset/sample_csv_files/alcohol-demand-log-spirits-consu.csv',
                  filename='alcohol-demand-log-spirits-consu.csv',
                  target_dir='../../dataset/reader_csv_files/',assetno = 'TSFAD_A1',n_rows=None,has_time=True):
    
    kwargs1 = reader_kwargs()
    
    con,param = preparecsvtoread(filepath=infile,filename=filename,target_dir=target_dir,assetno=assetno,
                                 n_rows=n_rows,has_time=has_time)
    kwargs1['con'] = con
    kwargs1['source_type'] = 'csv'
    kwargs1['from_timestamp']=-int(2**63)
    kwargs1['to_timestamp']=int(2**63)
    kwargs1['para_list'] = param
    
    return kwargs1