
import pandas as pd
import numpy as np
import datetime
table_name = "public.log_asset_timeline"
writer_kwargs = { 
                  'operating_unit_serial_number':'',
                  'event_type':'Symptom',
                  'event_sub_type':'Anomaly',
                  'event_source':'',
                  'event_name':'',
                  'event_context_info':[],
                  'event_state':'TRUE',
                  'parameter_list':'[{}]'.format(''),
                  'event_timestamp':'',
                  'event_timestamp_epoch':'',
                  'created_date':str(pd.to_datetime(datetime.datetime.now(),utc=True)),
                  'event_occurence_flag':1}