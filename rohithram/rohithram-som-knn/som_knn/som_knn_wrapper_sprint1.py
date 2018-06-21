

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':True,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':True,
            'test_frac':0.2
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':True,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':True,
            'test_frac':0.2
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',qry_str='',
         impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':True,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':True,
            'test_frac':0.2
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',qry_str='',
         impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':True,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':True,
            'test_frac':0.2
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',qry_str='',
         impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':True,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':True,
            'test_frac':0.2
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',qry_str='',
         impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant:None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,metric_names=cols,model_input_args,training_args)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno,model_input_args,training_args,metric_names=cols)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                                      )
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data=torch.from_numpy(data_per_asset[1:].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(
                        data=torch.from_numpy(data_per_asset[data_per_asset.columns[1:]].values),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    anomaly_detector = som_detector.Som_Detector(data = (data_per_asset),
                                                                 assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']

def main(assetno,from_timestamp,to_timestamp,con,para_list,source_type='opentsdb',table_name='',
        qry_str='',impute_fill_method='forward',down_sampling_method=None,down_sampling_window=None,freq=None,
        resample_fill_method=None,to_resample=None,to_impute=None,
        network_shape=None,input_feature_size=None,time_constant=None,minNumPerBmu=2,no_of_neighbours=10,init_radius=0.4,
        init_learning_rate=0.01,N=100,diff_order=1,is_train=True,epochs=4,batch_size=4,to_plot=True,test_frac=0.2,anom_thres=7):

        '''
        Wrapper function which should be called inorder to run the anomaly detection, it has four parts :
        *reader           - Class Data_reader defined in data_handler.py which takes in reader args and parses json 
                            and gives dataframes
        *preprocessor     - preprocessors are defined in preprocessors.py, which takes in data and gives out processed 
                            data
        *anomaly detector - Class Bayesian_Changept_Detector defined in bayesian_changept_detector.py, which takes in
                            data and algorithm parameters as argument and returns anomaly indexes and data.        
        *writer           - Class Postgres_Writer defined in data_handler.py which takes in anomaly detector object and
                            and sql_queries , db_properties and table name as args and gives out response code.
        
        Arguments :
        It takes reader args as of now to get the dataset and algo related arguments
        Note:
        To run this, import this python file as module and call this function with required args and it will detect
        anomalies and writes to the local database.
        This algorithm is univariate, so each metric per asset is processed individually
        '''
        
        #reader arguments
        reader_kwargs={
            'assetno':assetno,
            'from_timestamp':from_timestamp,
            'to_timestamp':to_timestamp,
            'con':con,
            'para_list':para_list,
            'source_type':source_type,
            'table_name':table_name,
            'qry_str':qry_str,
            'impute_fill_method':impute_fill_method,
            'down_sampling_method':down_sampling_method,
            'down_sampling_window':down_sampling_window,
            'freq':freq,
            'resample_fill_method':resample_fill_method,
            'to_resample':to_resample,
            'to_impute':to_impute
        }
        
        #algorithm arguments

        model_input_args = {
            'som_shape':network_shape,
            'input_feature_size':None,
            'time_constant':None,
            'minNumPerBmu':minNumPerBmu,
            'no_of_neighbors':no_of_neighbours,
            'initial_radius':init_radius,
            'initial_learning_rate':init_learning_rate,
            'n_iterations':None,
            'N':N,    
            'diff_order':diff_order
        }
        
        #Training arguments
        training_args = {
            'is_train':is_train,
            'epochs':epochs,
            'batch_size':batch_size,
            'to_plot':to_plot,
            'test_frac':test_frac
        }
        
        #merging all algo arguments for params checking
        algo_kwargs = {**model_input_args,**training_args}
        
                    
        try: 
            '''
            #reseting the error_codes to avoid overwritting
            #error_codes is a python file imported as error_codes which has error_codes dictionary mapping 
            #for different kinds errors and reset function to reset them.
            '''
            
            error_codes.reset()
            # type_checker is python file which has Type_checker class which checks given parameter types
            checker = type_checker.Type_checker(kwargs=algo_kwargs,ideal_args_type=ideal_kwargs_type)
            # res is None when no error raised, otherwise it stores the appropriate error message
            res = checker.params_checker()
            if(res!=None):
                return res
            
            # instanstiating the reader class with reader arguments
            data_reader = Data_reader(reader_kwargs=reader_kwargs)
            #getting list of dataframes per asset if not empty
            #otherwise gives string 'Empty Dataframe'
            entire_data = data_reader.read()
            
            writer_data = []
            anomaly_detectors = []
            
            if((len(entire_data)!=0 and entire_data!=None and type(entire_data)!=dict)):

                '''
                looping over the data per assets and inside that looping over metrics per asset
                * Instantiates anomaly detector class with algo args and metric index to detect on
                * Stores the anomaly indexes and anomaly detector object to bulk write to db at once
                '''

                for i,data_per_asset in enumerate(entire_data):
                    assetno = reader_kwargs['assetno'][i]
                    data_per_asset[data_per_asset.columns[1:]] = normalise_standardise(data_per_asset[data_per_asset.columns[1:]]
                                                                 )
                    
                    print("Data of Asset no: {} \n {}\n".format(assetno,data_per_asset.head()))
                    cols = list(data_per_asset.columns[1:])
                    
                    anomaly_detector = som_detector.Som_Detector(data = data_per_asset,                                                            assetno=assetno,model_input_args=model_input_args,
                                                                 training_args=training_args,metric_names=cols,
                                                                anom_thres=anom_thres)
                    anom_indexes = anomaly_detector.detect_anomalies()
                    anomaly_detectors.append(anomaly_detector)
                    
                    
#                     for data_col in range(1,len(data_per_asset.columns[1:])+1):
#                         algo_kwargs['data_col_index'] = data_col
#                         print("\nAnomaly detection for AssetNo : {} , Metric : {}\n ".format(assetno,data_per_asset.columns[data_col]))
#                         anomaly_detector = Bayesian_Changept_Detector(data_per_asset,assetno=assetno,**algo_kwargs)
#                         data,anom_indexes = anomaly_detector.detect_anomalies()
                    
                    
                
                '''
                Instantiates writer class to write into local database with arguments given below
                Used for Bulk writing
                '''
                sql_query_args = write_args.writer_kwargs
                table_name = write_args.table_name
                window_size = 10
                    
                writer = Postgres_Writer(anomaly_detectors,db_credentials=db_props.db_connection,sql_query_args=sql_query_args,
                                        table_name=table_name,window_size=window_size)

                #called for mapping args before writing into db
                res = writer.map_outputs_and_write()
                return res
            else:
                '''
                Data empty error
                '''
                return error_codes.error_codes['data_missing']
        except Exception as e:
            '''
            unknown exceptions are caught here and traceback used to know the source of the error
            '''
            traceback.print_exc()
            error_codes.error_codes['unknown']['message']=e
            return error_codes.error_codes['unknown']