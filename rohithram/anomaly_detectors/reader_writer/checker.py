def check_global_args(assetno, from_timestamp, to_timestamp, down_sampling_window,
                      down_sampling_method, con, freq, resample_fill_method,
                      impute_fill_method, to_resample, to_impute, para_list, source_type, table_name, qry_str):
    import re

    if (str(type(assetno)).find("list") == -1):
        return ('Asset entries must be in a list!')
    elif (len(assetno) == 0):
        return ('atleast one asset entry must exist!')
    elif from_timestamp is None:
        return ('from_timestamp is required!')
    elif to_timestamp is None:
        return ('to_timestamp is required!')
    elif ((type(from_timestamp) != int) | (type(to_timestamp) != int)):
        return ('from_timestamp and to_timestamp have to be single epoch timestamps, in milliseconds!')
    elif (from_timestamp >= to_timestamp):
        return ('from_timestamp has to be less than to_timestamp!')
    elif (str(down_sampling_method).find(",") != -1):
        return ('down_sampling_method must be a single value!')
    elif (str(down_sampling_window).find(",") != -1):
        return ('down_sampling_window must be a single value!')
    elif (str(type(para_list)).find("list") == -1):
        return ('parameter entries must be in a list!')
    elif (len(para_list) == 0):
        return ('atleast one parameter entry must exist!')
    elif (type(table_name) != str):
        return ('table name must be a string!')
    elif (type(qry_str) != str):
        return ('query must be a string!')

    if type(source_type) != str:
        return ('Invalid source type!')
    elif source_type.lower() not in ['postgres', 'csv', 'opentsdb']:
        """,'opentsdb']:"""
        return ('Invalid source type!')

    if ((source_type == 'postgres') & (len(table_name) == 0) & (len(qry_str) == 0)):
        return ('Either table name or query is required!')

    if freq is not None:
        if (type(freq) == str):
            if (bool(re.search(r'^[\d]*[LSTHDMAlsthdma]{1,1}$', freq)) == False):
                return ('Invalid resampling frequency!')
        else:
            return ('Invalid resampling frequency!')

    if ((to_resample is not None) & (isinstance(to_resample, bool) is False)):
        return ('to_resample must be a boolean value!')
    elif to_resample is None:
        to_resample = False
    elif to_resample is False:
        resample_fill_method = None
    else:
        if freq is None:
            return ('Invalid resampling frequency!')
        if resample_fill_method is not None:
            if (type(resample_fill_method) == str):
                if resample_fill_method.lower() not in ['forward', 'backward']:
                    return ('Invalid resample fill method!')
                else:
                    resample_fill_method = resample_fill_method.lower()
            elif ((str(resample_fill_method).isdigit() is False) & (type(resample_fill_method) != float)):
                return ('Invalid resample fill number!')

    if ((to_impute is not None) & (isinstance(to_impute, bool) is False)):
        return ('to_impute must be a boolean value!')
    elif to_impute is None:
        to_impute = False
    elif to_impute is False:
        impute_fill_method = None
    else:
        if impute_fill_method is not None:
            if (type(impute_fill_method) == str):
                if impute_fill_method.lower() not in ['forward', 'backward', 'drop']:
                    return ('Invalid impute fill method!')
                else:
                    impute_fill_method = impute_fill_method.lower()
            elif ((str(impute_fill_method).isdigit() is False) & (type(impute_fill_method) != float)):
                return ('Invalid impute fill number!')

    if (str(type(assetno)).find('list') == -1):
        return ('Invalid assetno!')

    if (down_sampling_method is not None):
        if (down_sampling_method.lower() not in ['sum', 'avg', 'max', 'min']):
            return ('invalid down_sampling_method!')
        if (down_sampling_window is None):
            return ('invalid down sampling window!')
        else:
            # 'y','M','d','h','m','s','ms'
            if (bool(re.search(r'^[\d]*[LSTHDMAlsthdma]{1,1}$', down_sampling_window)) == False):
                return ('invalid down sampling window!')
    elif (down_sampling_window is not None):
        return ('invalid down_sampling_method!')

    if con is None:
        return ('Connection details are required!')
    elif type(con) != str:
        return ('Connection details are invalid!')

    if down_sampling_method is None:
        down_sampling_method = ''
        down_sampling_window = ''

    if (source_type.lower() == 'opentsdb'):
        if ((down_sampling_method != '') & (down_sampling_window != '')):

            def ds_unit(tunit):
                switcher = {
                    'l': 'ms',
                    's': 's',
                    't': 'm',
                    'h': 'h',
                    'd': 'd',
                    'm': 'n',
                    'a': 'y'
                }
                return switcher.get(tunit)

            ds_win = str(filter(str.isdigit, str(down_sampling_window)))
            if len(ds_win) == 0:
                ds_win = str(1)
            window_level = down_sampling_window.replace(filter(str.isdigit, str(down_sampling_window)), '')
            tunit = ds_unit(window_level.lower())
            down_sampling_window = str(ds_win) + tunit

            down_sampling_window = str(down_sampling_window) + '-'
            down_sampling_method = str(down_sampling_method) + '-'

    msg = {}
    msg['Downsampling Method'] = down_sampling_method
    msg['Downsampling Window'] = down_sampling_window
    msg['Resample Fill Method'] = resample_fill_method
    msg['Impute Fill Method'] = impute_fill_method
    msg['Resample'] = to_resample
    msg['Impute'] = to_impute
    msg['source_type'] = source_type.lower()
    return (msg)
