
import pandas as pd
import numpy as np
import datetime
db_name = "Cerebra"
table_name = "public.log_asset_timeline"
assetno = '1'
name_of_algo = "bayesian_change_point_detection"
anomaly_code = "_bcp"
param = ['FE-001.DRIVEENERGY']
metric_name  = str(param[0])
event_name   = metric_name+anomaly_code+'_anomaly'
event_context_info = []
anom_time = ''
anom_time_epoch = ''
writer_kwargs = { 
                  'operating_unit_serial_number':assetno,
                  'event_type':'Symptom',
                  'event_sub_type':'Anomaly',
                  'event_source':name_of_algo,
                  'event_name':event_name,
                  'event_context_info':event_context_info,
                  'event_state':'TRUE',
                  'parameter_list':'[{}]'.format(metric_name),
                  'event_timestamp':anom_time,
                  'event_timestamp_epoch':anom_time_epoch,
                  'created_date':str(pd.to_datetime(datetime.datetime.now(),utc=True)),
                  'event_occurence_flag':1}