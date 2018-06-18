import datetime as dt
import sys

import numpy as np
import pandas as pd
import psycopg2

import reader_writer.checker as chk
import reader_writer.db_properties as db_props


def opentsdb_reader(para_list, assetno, from_timestamp, to_timestamp, down_sampling_method, down_sampling_window,
                    con):
    import requests, json
    finalDF = pd.DataFrame()
    for parameter_name in para_list:
        url = 'http://' + con + '/api/query?start=' + str(from_timestamp) + '&end=' + str(
            to_timestamp) + '&ms=true&m=max:' + down_sampling_window + down_sampling_method + 'none:' + parameter_name + '{AssetNo=' + assetno + '}'

        import sys
        try:
            res = requests.get(url)
        except:
            return (str(sys.exc_info()))
        opentsdbdata = json.loads(res.text)
        if len(opentsdbdata) > 0:
            midDF = pd.DataFrame()
            for i in range(len(opentsdbdata)):
                tempDF = pd.DataFrame()
                try:
                    tempDF[['timestamp', parameter_name]] = pd.DataFrame(list(opentsdbdata[i]['dps'].items()),
                                                                         columns=['timestamp', 'value'])
                except:
                    return (str(opentsdbdata['error']['message']))

                tempDF['assetno'] = (opentsdbdata[i].get('tags')).get('AssetNo')
                frames = [midDF, tempDF]
                midDF = pd.concat(frames)
            midDF = midDF.sort_values(['assetno', 'timestamp'], ascending=[True, True]).reset_index(drop=True)
            midDF = midDF.drop_duplicates(keep='first').reset_index(drop=True)
        else:
            return ('Empty DataFrame!')
        finalDF = finalDF.append(midDF)
    finalDF = finalDF.groupby('timestamp').max()
    finalDF = finalDF.reset_index(drop=False)
    return finalDF


def build_query(asset, from_timestamp, to_timestamp, para_list, table_name, down_sampling_method, down_sampling_window):
    qry_str = """SELECT CASE WHEN LENGTH("timestamp"::text)=10 THEN "timestamp"*1000 ELSE "timestamp" END AS "timestamp",assetno,""" + ",".join(
        map(str, para_list)) + " FROM " + table_name

    asset_list = "'" + ((",".join(map(str, asset))).replace(",", "','")) + "'"

    if ((down_sampling_method == '') & (down_sampling_window == '')):
        qry_str = """SELECT * FROM(""" + qry_str + """ WHERE assetno IN(""" + asset_list + """))res WHERE "timestamp" BETWEEN """ + str(
            from_timestamp) + """ AND """ + str(to_timestamp) + """ ORDER BY assetno,timestamp"""
    else:
        qry_str = """SELECT TO_TIMESTAMP("timestamp"::double precision / 1000)::timestamp with time zone AT TIME ZONE 'UTC' AS timestamp_human ,* FROM(""" + qry_str + " WHERE assetno IN(""" + asset_list + """))res WHERE timestamp BETWEEN """ + str(
            from_timestamp) + """ AND """ + str(to_timestamp)

        def ds_unit(tunit):
            switcher = {
                'a': 'year',
                'm': 'month',
                'd': 'day',
                'h': 'hour',
                't': 'minute',
                's': 'second',
                'l': 'millisecond',
            }
            return switcher.get(tunit)

        def ds_unit_upper(tunit):
            switcher = {
                'year': 'year',
                'month': 'year',
                'day': 'month',
                'hour': 'day',
                'minute': 'hour',
                'second': 'minute',
                'millisecond': 'second',
            }
            return switcher.get(tunit)

        ds_win = str(filter(str.isdigit, str(down_sampling_window)))
        if len(ds_win) == 0:
            ds_win = str(1)
        window_level = down_sampling_window.replace(filter(str.isdigit, str(down_sampling_window)), '')
        tunit = ds_unit(window_level.lower())
        tunit_upper = ds_unit_upper(tunit)

        qry_str = """(SELECT DATE_TRUNC('""" + tunit_upper + """',timestamp_human) AS t_up,
            TRUNC(EXTRACT(""" + tunit + """ FROM timestamp_human)/""" + ds_win + """) AS t,DATE_TRUNC('""" + tunit + """',timestamp_human) AS t_ds,* FROM(""" + qry_str + """)res1 ORDER BY to_timestamp("timestamp"))res2"""

        temp_str = """SELECT MAX(assetno) as assetno,CASE WHEN LENGTH(MAX(EXTRACT(epoch from t_ds))::text)=10 THEN MAX(EXTRACT(epoch from t_ds))*1000 END AS "timestamp","""
        for para in para_list:
            temp_str = temp_str + down_sampling_method.upper() + """(""" + para + """) as """ + para + ""","""
        temp_str = temp_str[:-1]
        qry_str = temp_str + """ FROM""" + qry_str + """ GROUP BY assetno,t_up,t ORDER by assetno,timestamp"""

    return (qry_str)


def postgres_reader(qry_str):
    try:
        conn = psycopg2.connect(
            """dbname='""" + db_props.db_connection["dbname"] + """' user='""" + db_props.db_connection[
                "user"] + """' password='""" + db_props.db_connection["password"] + """' host='""" +
            db_props.db_connection[
                "host"] + """' port='""" + db_props.db_connection["port"] + """'""")
        df = pd.read_sql_query(qry_str, con=conn)
    except:
        return (str(sys.exc_info()))
    return (df)


def csv_reader(con):
    import sys
    try:
        df = pd.read_csv(con, ",")
    except:
        return (str(sys.exc_info()))
    if (df.empty):
        return ('Empty DataFrame!')
    return (df)


def filter_csv_dataframe(df, para_list, assetno, from_timestamp, to_timestamp):
    import sys
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.ix[df['timestamp'].apply(int).apply(str).apply(len) == 10, 'timestamp'] = df['timestamp'] * 1e3

    df_para_list = list(set(df.columns.values))
    if ((set(para_list) <= set(df_para_list)) is False):
        return ('Invalid parameter_name!')
    else:
        try:
            df = df[df['assetno'].isin(assetno)].reset_index(drop=True)
            temp = []
            temp.append('assetno')
            temp.append('timestamp')
            temp = temp + para_list
            df = (df[((df.timestamp >= from_timestamp) & (df.timestamp <= to_timestamp))])[temp].reset_index(drop=True)
        except:
            return (str(sys.exc_info()))
    if (df.empty):
        return ('Empty DataFrame!')
    df = df.sort_values(by=['assetno', 'timestamp'], ascending=[True, True]).reset_index(drop=True)
    return (df)


def dataframe_downsampling(df, down_sampling_method, down_sampling_window):
    df = df_unix_to_ts(df)
    df.index = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

    import sys
    try:
        if down_sampling_method == 'sum':
            tempdf = df.groupby(['assetno'], as_index=True).resample(down_sampling_window.lower()).agg(np.sum)
        elif down_sampling_method == 'avg':
            tempdf = df.groupby(['assetno'], as_index=True).resample(down_sampling_window.lower()).agg(np.mean)
        elif down_sampling_method == 'min':
            tempdf = df.groupby(['assetno'], as_index=True).resample(down_sampling_window.lower()).agg(np.min)
        elif down_sampling_method == 'max':
            tempdf = df.groupby(['assetno'], as_index=True).resample(down_sampling_window.lower()).agg(np.max)
        else:
            return ('Invalid down_sampling_method!')
    except:
        return (str(sys.exc_info()))

    if (tempdf.empty):
        return ('Empty DataFrame!')

    df = tempdf.reset_index(['timestamp', 'assetno'], drop=True).dropna().reset_index(drop=True)
    df = df_ts_to_unix(df)
    return (df)


def resample_dataframe(df, freq):
    df = df_unix_to_ts(df)
    df.index = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    import sys
    try:
        df = df.groupby(['assetno'], as_index=True).resample(freq).mean().reset_index()
    except:
        return (str(sys.exc_info()))
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df = df_ts_to_unix(df)
    return (df)


def fill_dataframe(df, fill_method):
    import re
    if fill_method is not None:
        if (fill_method == 'forward'):
            df = df.groupby(['assetno'], as_index=True).ffill().bfill()
        elif (fill_method == 'backward'):
            df = df.groupby(['assetno'], as_index=True).bfill().ffill()
        elif (fill_method == 'drop'):
            df = df.dropna()
        elif bool(re.search('^[\d]*[.]?[\d]+$', str(fill_method))) is True:
            # if fill method is neither of the above, this holds the float/int value that is passed like 0 or 1.5 etc
            df = df.fillna(fill_method)
    df = df.reset_index(drop=True)
    return (df)


def df_ts_to_unix(df):
    df['timestamp'] = ((df['timestamp'] - dt.datetime(1970, 1, 1)).dt.total_seconds() * 1000).astype(long)
    return (df)


def df_unix_to_ts(df):
    df.timestamp = pd.to_datetime(df['timestamp'], unit='ms')
    return (df)


def json_body_format(df_grouped, para_list):
    dict_output = {'assetno': df_grouped['assetno'].iloc[0]}
    dict_output['readings'] = []
    for col in para_list:
        dict_output['readings'].append({'name': col,
                                        'datapoints': df_grouped.apply(lambda row: (row['timestamp'], row[col]),
                                                                       1).tolist()})
    return dict_output


def json_output_format(df, para_list, assetno):
    df[para_list] = df[para_list].astype('float')
    data_dict = {}
    data_dict['header'] = {
        'parameter_count': len(para_list),
        'asset_count': len(assetno),
        'data_count': df.shape[0]
    }

    data_dict['body'] = df.groupby('assetno', as_index=False).apply(json_body_format, para_list=para_list).tolist()
    return data_dict


def reader_api(assetno, from_timestamp, to_timestamp, down_sampling_method, down_sampling_window, con,
               freq, resample_fill_method, impute_fill_method, to_resample, to_impute, para_list, table_name,
               source_type, qry_str):
    # check parameters first
    msg = chk.check_global_args(assetno, from_timestamp, to_timestamp, down_sampling_window,
                                down_sampling_method, con, freq, resample_fill_method,
                                impute_fill_method, to_resample, to_impute, para_list, source_type, table_name, qry_str)
    if type(msg) != str:
        if (source_type == 'opentsdb'):
            df = opentsdb_reader(para_list, ",".join(assetno), from_timestamp, to_timestamp, msg['Downsampling Method'],
                                 msg['Downsampling Window'], con)

        elif (source_type == 'csv'):
            df = csv_reader(con)
            if (type(df) != str):
                df = filter_csv_dataframe(df, para_list, assetno, from_timestamp, to_timestamp)
                if (type(df) != str):
                    if ((msg['Downsampling Method'] != '') & (msg['Downsampling Window'] != '')):
                        df = dataframe_downsampling(df, msg['Downsampling Method'], msg['Downsampling Window'])
        else:
            if len(qry_str) == 0:
                qry_str = build_query(assetno, from_timestamp, to_timestamp, para_list, table_name,
                                      msg['Downsampling Method'], msg['Downsampling Window'])
            df = postgres_reader(qry_str)

        if ((type(df) != str)):
            if to_resample is True:
                if to_impute is True:
                    df = fill_dataframe(df, msg['Impute Fill Method'])
                    df = resample_dataframe(df, freq)
                    if type(df) != str:
                        df = fill_dataframe(df, msg['Resample Fill Method'])
                else:
                    df = resample_dataframe(df, freq)
                    if type(df) != str:
                        df = fill_dataframe(df, msg['Resample Fill Method'])
            else:
                if to_impute is True:
                    df = fill_dataframe(df, msg['Impute Fill Method'])
                else:
                    df = fill_dataframe(df, None)

            if ((type(df) != str)):
                if (df.empty):
                    return ('Empty DataFrame!')
                df['timestamp'] = df.ix[:, 'timestamp'].apply(long)

            data_dict = json_output_format(df, para_list, assetno)
            return data_dict
        return df
    return msg
