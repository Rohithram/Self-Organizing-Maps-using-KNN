

import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

#torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#importing sklearn libraries
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import datetime as dt
import time
import os
import pickle 

from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

#importing error_codes
from anomaly_detectors.utils import error_codes as error_codes
from anomaly_detectors.som_knn_detector import som_knn_module
import traceback
import warnings
from anomaly_detectors.utils.preprocessors import *

warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 12, 9
rcParams[ 'axes.grid']=True


def save_model(model,metric_names,filename='som_trained_model',target_dir="../../Anomaly_Detection_Models/Machine_Learning_Models"):
    
    try:
        time_now = ts_to_unix(pd.to_datetime(dt.datetime.now()))
        metric_names = [''.join(e for e in metric if e.isalnum()) for metric in metric_names]
        filename = filename+'_{}_{}'.format('_'.join(metric_names),str(time_now))

        filepath = os.path.join(target_dir,filename)
        

        filehandler = open(filepath, 'wb')
        pickle.dump(model, filehandler)
        print("\nSaved model : {} in {},\nLast Checkpointed at: {}\n".format(filename,target_dir,time_now))
        return filepath,time_now
    
    except Exception as e:
        traceback.print_exc()
        print("Error occured while saving model\n")
        error_codes.error_codes['unknown']['message']=e
        return error_codes.error_codes['unknown']
    
def load_model(filepath):
    filehandler = open(filepath, 'rb')
    return pickle.load(filehandler)

class Som_Detector():
    def __init__(self,data,assetno,metric_names,model_input_args,training_args,eval_args):
        
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
        
        if(training_args is not None):
            self.istrain = training_args['is_train']
        else:
            self.istrain = False
            
        self.data = data
        self.metric_name = metric_names
        self.assetno = assetno
        self.anom_indexes = None
        self.data_col_index = None
        self.model_input_args = model_input_args
        self.training_args = training_args
        self.eval_args = eval_args

    def detect_anomalies(self):
        
        '''
        Detects anomalies and returns data and anomaly indexes
        '''
        
        data = torch.from_numpy(self.data[self.data.columns[1:]].values)
        
        
        print("Shape of the Entire dataset : {}\n".format(data.shape))
#         print(data.shape)

        if(self.istrain):
            data_set,train_data,test_data = process_data(data=data,test_frac=self.training_args['test_frac'],
                                                        to_plot=self.training_args['to_plot'])
            entire_data = data_set[:,].numpy()

            diff_order = self.model_input_args['diff_order']
            net = train_som(train_data,self.model_input_args,self.training_args)
            model_path = save_model(net,metric_names = self.metric_name)
            
            if(type(model_path)!=str and type(model_path)==dict):
                #meaning some unknown error happened while saving the model
                return model_path
            
            #Validation on entire data after training on the train data
            anom_indexes = test(net,entire_data,to_plot=self.training_args['to_plot'])
            self.anom_indexes = anom_indexes

        else:
            model_path = self.eval_args['model_path']
            anom_thres = self.eval_args['anom_thres']
            eval_net = load_model(model_path)
            anom_indexes = test(eval_net,data.numpy(),anom_thres=anom_thres,to_plot=self.eval_args['to_plot'])
            self.anom_indexes = anom_indexes

            print("\n No of Anomalies detected = %g"%(len(anom_indexes)))

          

        return anom_indexes


class TimeSeries_Dataset(Dataset):
    """Time series dataset."""

    def __init__(self,data,data_col_index=0):
        """
        Args:
            data: input data after all preprocessing done
            Trains the model on this Tensor object data
        """
        
        self.data = data

#         print("Overview of dataset : \n {} \n".format(self.data.head()))
            
        print(self.data.dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]        
        
        return (sample)

def split_the_data(data,test_frac=0.0):
    '''
    Splitting the data into train and test with default ratio = 0.1
    Splits the data in orderly manner not random
    '''
    if(test_frac is not None):
        train_data = data[0:int(np.ceil((1-test_frac)*data[:,].shape[0])),:]
        test_data = data[-int(np.ceil(test_frac*data[:,].shape[0])):]
    else:
        train_data = data
        test_data = torch.empty()
    return train_data,test_data

def plot_dataset(data):
    fig = plt.figure()
    print("Dataset has {} rows {} columns".format(data[:,].shape[0],data[:,].shape[1]))

    for i in range(data[:,].shape[-1]):
        plt.plot(data[:,i])
        plt.title("Plot of the dataset column wise")
    plt.show()   

def process_data(data,test_frac,to_plot=True):
    data_set = TimeSeries_Dataset(data)
    train_data,test_data = split_the_data(data_set,test_frac=test_frac)
    print("Shape of Training dataset :{} and Test dataset :{}\n".format(train_data.shape,test_data.shape))
    if(to_plot):
        plot_dataset(train_data.numpy())
    return data_set,train_data,test_data

def network_dimensions(train_data,N=100):
    
    approx_network_size = 5*np.sqrt(N)
    train_df = pd.DataFrame(train_data.numpy())
    cov_train_df = train_df.cov()
    cov_data = cov_train_df.values
    
    if(train_data.shape[-1]<=1):
        x = sp.linalg.eigh(cov_data,eigvals_only=True)[-1:]
        ratio = np.ceil(x)    
    else:
        x,y = sp.linalg.eigh(cov_data,eigvals_only=True)[-2:]
        ratio = np.ceil(x)/np.ceil(y)
        
    row_dim = int(np.ceil(np.sqrt(approx_network_size*ratio)))
    col_dim = int(np.ceil(approx_network_size/row_dim))
    return row_dim,col_dim

def test(net,evaluateData,anom_thres=3,to_plot=True):
        
        original_data = evaluateData
        no_cols = original_data.shape[-1]
        diff_order = net.diff_order
        print("Input data's shape: {}".format(original_data.shape))
        res_evaluateData = np.diff(evaluateData,n=diff_order,axis=0).reshape(-1,no_cols)
        print("Differenced data shape {}".format(res_evaluateData.shape))
        
#         res_evaluateData.reshape(-1,original_data.shape[-1])
        # Fit the anomaly detector and apply to the evaluateData data
        anomaly_metrics = net.evaluate(res_evaluateData) # Evaluate on the evaluateData data
        print(anomaly_metrics.shape)
        anomaly_metrics = anomaly_metrics/np.linalg.norm(anomaly_metrics)
#         k=anom_thres
        thres = anom_thres*(1/np.sqrt(len(anomaly_metrics)))
#         thres = np.mean(anomaly_metrics)+k*np.std(anomaly_metrics)
        selector = anomaly_metrics > thres
        anom_indexes = np.arange(len(res_evaluateData))[selector]
#         anom_indexes = anom_indexes+diff_order
#         anom_indexes = np.arange(original_data.shape[0]-diff_order)[selector]
        
        
        if(to_plot):
            '''
            # We make a density plot and a histogram showing the distrbution
            # of the number of points mapped to a BMU
            '''
            figa = plt.figure(figsize=(20,10))
            plt.subplot(121)
            density = gaussian_kde(anomaly_metrics)
            xs = np.linspace(0,5,200)
            plt.plot(xs,density(xs))
            plt.title("Distribution of dataset")

            plt.subplot(122)
            plt.hist(net.bmu_counts)
            plt.title("Histogram of Bmu counts")
            plt.show();
            
            fig = plt.figure(figsize=(20,10))
            plt.plot(anomaly_metrics)
            plt.title("Anomaly score")
            plt.axhline(y=thres,color='r',label="Threshold")
            plt.legend()
            plt.show();

            if(diff_order!=0):
                fig2 = plt.figure(figsize=(20,10))
                plt.plot(res_evaluateData[:,])
                plt.title("Dataset after differencing marked with anomalies")
#                 plt.scatter(x=anom_indexes,y=res_evaluateData[anom_indexes,0],color='r')
                [plt.axvline(x=ind,color='r') for ind in anom_indexes]
                plt.show();

                fig3 = plt.figure(figsize=(20,10))
                plt.plot(original_data)
                plt.title("Exact Dataset with detectedanomalies")
#                 plt.scatter(x=anom_indexes,y=original_data[anom_indexes,0],color='r')
                [plt.axvline(x=ind,color='r') for ind in anom_indexes]

                plt.show();

            else:
                fig3 = plt.figure(figsize=(20,10))
                plt.plot(original_data)
                plt.title("Exact Dataset with detectedanomalies")
#                 plt.scatter(x=anom_indexes,y=original_data[anom_indexes,0],color='r')
                [plt.axvline(x=ind,color='r') for ind in anom_indexes]

                plt.show();
        
        
#         print("Anomaly indexes : {}".format(anom_indexes))

        no_anoms_detected = (list(selector).count(True))
        print("No of anomalies detected : {}, Fraction of data detected as anomaly : {}".
              format(no_anoms_detected,no_anoms_detected/(evaluateData.shape[0])))
        return anom_indexes


def train_loop(net,train_loader,epochs):
    curr_batch_iter = 0
    for epoch in range(epochs):
        print("Epoch : {} completed \n Max Bmu index : {}".
              format(epoch,np.unravel_index(torch.argmax(net.bmu_counts),net.bmu_counts.shape)))

    for i,x_batch in enumerate(train_loader):
            curr_batch_iter += 1
            net = net.fit(x_batch,curr_batch_iter)
    
    print("\n Training successfully completed \n")
    return net


def check_default_args(inp_kwargs,def_kwargs):
    for key in inp_kwargs:
        for def_key in def_kwargs:
            if(inp_kwargs[def_key]==None):
                inp_kwargs[def_key]=def_kwargs[def_key]
    return inp_kwargs

def train_som(train_data,model_input_args,training_args):
    
    def_kwargs = {}
    
    epochs = training_args['epochs']
    batch_size = training_args['batch_size']
    n_iterations = int(epochs*(len(train_data)/batch_size))

    def_kwargs['som_shape'] = network_dimensions(train_data,model_input_args['N'])
    row_dim,col_dim = def_kwargs['som_shape']

    def_kwargs['initial_radius'] = max(row_dim,col_dim)/2
    def_kwargs['time_constant'] = n_iterations/np.log(def_kwargs['initial_radius'])
    
    model_kwargs = check_default_args(model_input_args,def_kwargs)
    model_input_args.update(model_kwargs)
    
        
#     actual_network_size = row_dim*col_dim
    
    
    
    # initial neighbourhood radius
#     init_radius = model_input_args['initial_radius']
        
    # initial learning rate
#     init_learning_rate = model_input_args['initial_learning_rate']
    
    # radius decay parameter
#     time_constant = model_input_args['time_constant']
    
    model_input_args['input_feature_size'] = train_data.shape[-1]
    model_input_args['n_iterations'] = n_iterations
    
    
    print("Network dimensions are {} x {} \n".format(row_dim,col_dim))
    diff_order = model_input_args['diff_order']
    del model_input_args['N']
    net = som_knn_module.Som_net(**model_input_args)

    
    res_train_data = (np.diff(train_data.numpy(),n=diff_order,axis=0).reshape(-1,train_data.numpy().shape[-1]))
    print("\nShape of differenced Training data : {}\n".format(res_train_data.shape))
    train_data_diff = torch.from_numpy(res_train_data)

    train_loader = torch.utils.data.DataLoader(train_data_diff, batch_size=batch_size,shuffle=True)
    
    net = train_loop(net=net,epochs=epochs,train_loader=train_loader)
    
    return net