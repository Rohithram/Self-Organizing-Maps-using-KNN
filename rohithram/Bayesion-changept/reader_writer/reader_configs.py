
assetno = ['1']
opentsdb_url = '52.173.76.89:4242'
src_type     =  'opentsdb'
param = ['FE-001.DRIVEENERGY']
reader_kwargs={
    'assetno':'AssetNo={}'.format(','.join(assetno)),
    'from_timestamp':1520402214,
    'to_timestamp':1520407294,
    'con':opentsdb_url,
    'para_list':param,
    'source_type':src_type,
    'table_name':'',
    'qry_str':'',
    'impute_fill_method':0.5,
    'down_sampling_method':'',
    'down_sampling_window':'',
    'freq':None,
    'resample_fill_method':None,
    'to_resample':None,
    'to_impute':None
}