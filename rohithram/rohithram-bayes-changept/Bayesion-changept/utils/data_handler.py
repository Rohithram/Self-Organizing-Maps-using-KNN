

import numpy as np
import pandas as pd
import psycopg2
import reader_writer.reader as reader
import reader_writer.checker as checker
import datetime as dt

class Data_Writer():
    def __init__(self,anomaly_detector):
        self.anomaly_detector = anomaly_detector
               
        
    def write(self):
        raise NotImplementedError    

class Anomaly_Detector():
    def __init__(self,algo_name,data,anom_indexes,is_train=False,):
        self.algo_name = algo_name
        self.istrainable = is_train
        self.data = data
        self.anom_indexes = anom_indexes

    
class Postgres_Writer():
    def __init__(self,anomaly_detector,db_credentials,sql_query_args,table_name,window_size=10):
        
#         super(Postgres_Writer,self).__init__(anomaly_detector)
        self.anomaly_detector = anomaly_detector
        self.db_credentials = db_credentials
        self.sql_query_args = sql_query_args
        self.table_name = table_name
        self.window_size = window_size
        print("Postgres writer initialised \n")

        
    def write_to_db(self,col_names,col_vals):

    #     print("\n Changepoint info : \n {}".format(col_vals))
        col_vals1 = [[str(val) if(type(val)!=str) else "'{}'".format(val) for val in row] for row in col_vals]
        joined_col_vals = ["({})".format(','.join(map(str,val))) for val in col_vals1]
        fmt_col_vals = (','.join(joined_col_vals))
        insert_query = """ INSERT INTO {} ({}) VALUES{};""".format(self.table_name,col_names,fmt_col_vals)
        
        status = 0
        try:
            conn = psycopg2.connect(**self.db_credentials)
            cur = conn.cursor()
            cur.execute(insert_query)
            conn.commit()
            print('\n Successfully written into database\n')

        except psycopg2.DatabaseError as error:
            status = 1
            print("Database error : {}".format(error))
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

        return status
        
    def ts_to_unix(self,t):
        return int((t - dt.datetime(1970, 1, 1)).total_seconds()*1000)
    
    def map_outputs_and_write(self):
         
        original_data = self.anomaly_detector.data
        sql_query_args = self.sql_query_args
        window = self.window_size
        col_names_list = list(sql_query_args.keys())
        col_names = ','.join(col_names_list)
        col_vals = []
        table_name = self.table_name
        
        if(self.anomaly_detector.anom_indexes is not None):
            anom_indexes = self.anomaly_detector.anom_indexes

            for i in anom_indexes:

                sql_query_args['parameter_list'] = '[{}]'.format(original_data.columns[self.anomaly_detector.data_col_index])
                sql_query_args['event_name'] = original_data.columns[0]+self.anomaly_detector.algo_code+'_anomaly'
                sql_query_args['event_source'] = self.anomaly_detector.algo_name
                
                if(original_data.index.values.dtype!='int64'):
                    time_series = original_data.index[i-window:i+window]
                    print(time_series)
                    sql_query_args['event_timestamp'] =  str(original_data.index[i])
                    sql_query_args['event_timestamp_epoch'] = str(self.ts_to_unix(original_data.index[i]))
                else:
                    time_series = (pd.to_datetime(original_data.index[i-window:i+window],unit='ms',utc=True))
                    sql_query_args['event_timestamp'] =  str(pd.to_datetime(original_data.index[i],unit='ms',utc=True))
                    sql_query_args['event_timestamp_epoch'] = str((original_data.index[i]))

                time_around_anoms = ["''{}''".format((t)) for t in time_series]
                data_around_anoms = {'timestamp':time_around_anoms,
                                    'value':(list(original_data.iloc[i-window:i+window,0].values))}

                pts_around_anoms = ''

                for key,val in data_around_anoms.items():
                    pts_around_anoms += "{}:{},".format(key,val)

                sql_query_args['event_context_info'] = "{}".format("{"+pts_around_anoms.strip(',')+"}")

                col_vals.append(list(sql_query_args.values()))

            self.write_to_db(col_names,col_vals)
class Data_reader():
    
    def __init__(self,reader_kwargs):
        self.reader_kwargs = reader_kwargs
        print("Data reader initialised \n")
        
    def read(self):
        
        response_dict=reader.reader_api(**self.reader_kwargs)
        print(response_dict)
        print("Getting the dataset from the reader....\n")
        entire_data = self.parse_dict_to_dataframe(response_dict)
        print(entire_data.head())

        print(entire_data.dtypes)
        print(entire_data.shape)
    #     entire_data = entire_data[np.isfinite(entire_data[entire_data.columns].values)]
        return entire_data
    
    def parse_dict_to_dataframe(self,response_dict):
        entire_data_set = []

        for data_per_asset in response_dict['body']:
            assetno = data_per_asset['assetno']
            for data_per_metric in data_per_asset['readings']:
                data = pd.DataFrame(data_per_metric['datapoints'],columns=['timestamp',data_per_metric['name']])
                data.index = data['timestamp']
                del data['timestamp']
                data['assetno']=assetno
                entire_data_set.append(data)
                print(data)
        return pd.concat(entire_data_set,ignore_index=True)