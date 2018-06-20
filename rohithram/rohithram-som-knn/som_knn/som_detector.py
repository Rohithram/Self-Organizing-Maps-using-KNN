

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing modules to run the algo

class Som_Detector():
    def __init__(self,data,assetno,is_train=True,model_input_args,training_args):
        
        '''
        Class which is used to find Changepoints in the dataset with given algorithm parameters.
        It has all methods related to finding anomalies to plotting those anomalies and returns the
        data being analysed and anomaly indexes.
        Arguments :
        data -> dataframe which has one or two more metric columnwise
        assetno -> assetno of the dataset
        is_train -> By Default is False , as no training required for this algo
        data_col_index -> column index of the metric to find changepoints on
        pthres -> Default value :0.5 , (float) it is threshold after which a changepoint is flagged as on anomaly
        mean_runlen -> (int) By default 100, It is the average gap between two changepoints , this comes from 
                       nitty gritty math of exponential distributions
        Nw (samples to wait) -> (int) By default 10 is being used for optimal performance. It is the samples after which
                                we start assigning probailities for it to be a changepoint.
        to_plot -> True if you want to plot anomalies
        '''
        
        
        self.algo_name = 'Self Organizing Map AD'
        self.algo_code = 'som'
        self.algo_type = 'multivariate'
        self.istrain = is_train
        self.data = data
        self.data_col_index = data_col_index
        self.metric_name = list(data.columns[1:])
        self.assetno = assetno
        self.anom_indexes = None

    def detect_anomalies(self):
        
        '''
        Detects anomalies and returns data and anomaly indexes
        '''
        data = self.data
        print("Shape of the dataset : ")
        print(data.shape)
        if(self.istrain):
            net,data_set,train_data,test_data = som_detector.train_som(data,self.model_input_args,self.training_args)
            anom_indexes = som_detector.test(net,test_data,anom_thres=anom_thres,diff_order=diff_order)
        else:
            anom_indexes = som_detector.test(net,test_data,anom_thres=anom_thres,diff_order=diff_order)

        self.anom_indexes = anom_indexes
          
        print("\n No of Anomalies detected = %g"%(len(anom_indexes)))

        return anom_indexes

class Som_Detector():
    def __init__(self,data,assetno,model_input_args,training_args):
        
        '''
        Class which is used to find Changepoints in the dataset with given algorithm parameters.
        It has all methods related to finding anomalies to plotting those anomalies and returns the
        data being analysed and anomaly indexes.
        Arguments :
        data -> dataframe which has one or two more metric columnwise
        assetno -> assetno of the dataset
        is_train -> By Default is False , as no training required for this algo
        data_col_index -> column index of the metric to find changepoints on
        pthres -> Default value :0.5 , (float) it is threshold after which a changepoint is flagged as on anomaly
        mean_runlen -> (int) By default 100, It is the average gap between two changepoints , this comes from 
                       nitty gritty math of exponential distributions
        Nw (samples to wait) -> (int) By default 10 is being used for optimal performance. It is the samples after which
                                we start assigning probailities for it to be a changepoint.
        to_plot -> True if you want to plot anomalies
        '''
        
        
        self.algo_name = 'Self Organizing Map AD'
        self.algo_code = 'som'
        self.algo_type = 'multivariate'
        self.istrain = training_args['is_train']
        self.data = data
        self.data_col_index = data_col_index
        self.metric_name = list(data.columns[1:])
        self.assetno = assetno
        self.anom_indexes = None

    def detect_anomalies(self):
        
        '''
        Detects anomalies and returns data and anomaly indexes
        '''
        data = self.data
        print("Shape of the dataset : ")
        print(data.shape)
        if(self.istrain):
            net,data_set,train_data,test_data = som.train_som(data,self.model_input_args,self.training_args)
            anom_indexes = som.test(net,test_data,anom_thres=anom_thres,diff_order=diff_order)
        else:
            anom_indexes = som.test(net,test_data,anom_thres=anom_thres,diff_order=diff_order)

        self.anom_indexes = anom_indexes
          
        print("\n No of Anomalies detected = %g"%(len(anom_indexes)))

        return anom_indexes