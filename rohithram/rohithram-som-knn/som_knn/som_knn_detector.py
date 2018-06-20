

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


from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

#importing error_codes
import error_codes as error_codes
import som_knn_module

import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 12, 9
rcParams[ 'axes.grid']=True

class Som_Detector():
    def __init__(self,data,assetno,metric_names,model_input_args,training_args,anom_thres=7):
        
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
        self.metric_name = metric_names
        self.assetno = assetno
        self.anom_indexes = None
        self.model_input_args = model_input_args
        self.training_args = training_args
        self.anom_thres = anom_thres

    def detect_anomalies(self):
        
        '''
        Detects anomalies and returns data and anomaly indexes
        '''
        data = self.data
        anom_thres = self.anom_thres
        diff_order = self.model_input_args['diff_order']
        
        print("Shape of the dataset : ")
        print(data.shape)
        if(self.istrain):
            net,data_set,train_data,test_data = train_som(data,self.model_input_args,self.training_args)
            anom_indexes = test(net,data_set[:,].numpy(),anom_thres=anom_thres,diff_order=diff_order)
        else:
            anom_indexes = test(net,test_data,anom_thres=anom_thres,diff_order=diff_order)

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

def split_the_data(data,test_frac=0.1):
    '''
    Splitting the data into train and test with default ratio = 0.1
    Splits the data in orderly manner not random
    '''
    train_data = data[0:int(np.ceil((1-test_frac)*data[:,].shape[0])),:]
    test_data = data[-int(np.ceil(test_frac*data[:,].shape[0])):]
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

def test(net,evaluateData,anom_thres=7,diff_order=1,to_plot=True):
        
        original_data = evaluateData
        res_evaluateData = np.diff(evaluateData.reshape(-1),n=diff_order).reshape(-1,1)
        
        # Fit the anomaly detector and apply to the evaluateData data
        anomaly_metrics = net.evaluate(res_evaluateData) # Evaluate on the evaluateData data
        
        anomaly_metrics = anomaly_metrics/np.linalg.norm(anomaly_metrics)
#         k=anom_thres
        thres = anom_thres*(1/np.sqrt(len(anomaly_metrics)))
#         thres = np.mean(anomaly_metrics)+k*np.std(anomaly_metrics)
        selector = anomaly_metrics > thres
#         anom_indexes = np.arange(len(res_evaluateData))[selector]
        anom_indexes = np.arange(len(original_data)-diff_order)[selector]
        
        
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
                plt.scatter(x=anom_indexes,y=res_evaluateData[anom_indexes],color='r')
                plt.show();

                fig3 = plt.figure(figsize=(20,10))
                plt.plot(original_data)
                plt.title("Exact Dataset with detectedanomalies")
                plt.scatter(x=anom_indexes,y=original_data[anom_indexes],color='r')
                plt.show();

            else:
                fig3 = plt.figure(figsize=(20,10))
                plt.plot(original_data)
                plt.title("Exact Dataset with detectedanomalies")
                plt.scatter(x=anom_indexes,y=original_data[anom_indexes],color='r')
                plt.show();
        
        
#         print("Anomaly indexes : {}".format(anom_indexes))

        no_anoms_detected = (list(selector).count(True))
        print("No of anomalies detected : {}, Fraction of data detected as anomaly : {}".
              format(no_anoms_detected,no_anoms_detected/(evaluateData.shape[0])))
        return anom_indexes


def train(net,train_loader,epochs):
    curr_batch_iter = 0
    for epoch in range(epochs):
        print("Epoch : {} completed \n Max Bmu index : {}".
              format(epoch,np.unravel_index(torch.argmax(net.bmu_counts),net.bmu_counts.shape)))

    for i,x_batch in enumerate(train_loader):
            curr_batch_iter += 1
            net = net.fit(x_batch,curr_batch_iter)
    
    print("\n Training successfully completed \n")
    return net

def train_som(data,model_input_args,training_args):
    
    data_set,train_data,test_data = process_data(data=data,test_frac=training_args['test_frac'])
    
    if(model_input_args['som_shape']!=None):
        row_dim,col_dim = network_dimensions(train_data,model_input_args['N'])
    else:
        row_dim,col_dim = model_input_args['som_shape']
        
    actual_network_size = row_dim*col_dim
    
    epochs = training_args['epochs']
    batch_size = training_args['batch_size']
    
    # initial neighbourhood radius
    if(model_input_args['initial_radius']==None):
        init_radius = max(row_dim,col_dim)/2
    else:
        init_radius = model_input_args['initial_radius']
        
    # initial learning rate
    init_learning_rate = model_input_args['initial_learning_rate']
    # radius decay parameter
    n_iterations = int(epochs*(len(train_data)/batch_size))
    time_constant = n_iterations/np.log(init_radius)
    
    model_input_args['som_shape'] = (row_dim,col_dim)
    model_input_args['input_feature_size'] = train_data.shape[-1]
    model_input_args['initial_radius'] = init_radius
    model_input_args['initial_learning_rate'] = init_learning_rate
    model_input_args['time_constant'] = time_constant
    model_input_args['n_iterations'] = n_iterations
    
    
    print("Network dimensions are {} x {} \n".format(row_dim,col_dim))
    diff_order = model_input_args['diff_order']
    del model_input_args['diff_order']
    del model_input_args['N']
    net = som_knn_module.Som_net(**model_input_args)

    
    res_series = (np.diff(train_data.numpy().reshape(-1),n=diff_order))
    
    train_data_diff = torch.from_numpy(res_series)

    train_loader = torch.utils.data.DataLoader(train_data_diff, batch_size=batch_size,shuffle=True)
    
#     if(training_args['freeze']==False):
    net = train(net=net,epochs=epochs,train_loader=train_loader)
    
    return net,data_set,train_data,test_data