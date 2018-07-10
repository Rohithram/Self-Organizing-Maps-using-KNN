

import numpy as np
import pandas as pd
import json
import scipy as sp

#torch libraries
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import datetime as dt
import time
import os
import pickle 

from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

#importing dependencies
from anomaly_detectors.utils.error_codes import error_codes
from anomaly_detectors.som_knn_detector import som_knn_module
from anomaly_detectors.utils.preprocessors import *

import traceback
import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 12, 9
rcParams[ 'axes.grid']=True


def save_model(model,metric_names,assetno,filename='som_trained_model',
               target_dir="../../Anomaly_Detection_Models/Machine_Learning_Models"):
    '''
    Function to save the model in the given relative path using pickle
    Arguments:
    Required Params:
        model : MODEL's class object which contains all model related info like weights and architecture
        metric_names :  list of metric names to form the filename for saving the model
        filename : Default -> 'som_trained_model' (It's main part of the filename)
        target_dir: Give relative path to the target directory
    '''
    
    error_codes1  = error_codes()
    try:
        time_now = ts_to_unix(pd.to_datetime(dt.datetime.now()))
        metric_names = [''.join(e for e in metric if e.isalnum()) for metric in metric_names]
        
        # Creating the filename with metricnames and assetno and current time
        filename = filename+'_{}_{}'.format('_'.join(metric_names),str(assetno),str(time_now))
        
        filepath = os.path.join(target_dir,filename)
        
        if(len(filepath)>100):
            filepath = filepath[:100]

        filehandler = open(filepath, 'wb')
        pickle.dump(model, filehandler)
        print("\nSaved model : {} in {},\nLast Checkpointed at: {}\n".format(filename,target_dir,time_now))
        return filepath
    
    except Exception as e:
        traceback.print_exc()
        print("Error occured while saving model\n")
        error_codes1['unknown']['message']=e
        return error_codes1['unknown']
    
def load_model(filepath):
    '''
    Load the model from the given relative filepath
    '''
    filehandler = open(filepath, 'rb')
    return pickle.load(filehandler)

class Som_Detector():
    def __init__(self,data,model_input_args,training_args,eval_args):
        
        '''
        Class which is used to find Changepoints in the dataset with given algorithm parameters.
        It has all methods related to finding anomalies to plotting those anomalies and returns the
        data being analysed and anomaly indexes.
        Arguments :
        data -> dataframe which has one or two more metric columnwise per asset
        model_input_args: dictionary of model related arguments 
        '''
        
        
        self.algo_name = 'Self Organizing Map AD'
        self.algo_code = 'som'
        self.algo_type = 'multivariate'
        
        #training args is set to None incase of evaluation mode
        if(training_args is not None):
            self.istrain = training_args['is_train']
        else:
            self.istrain = False
            
        self.data = data
        self.metric_name = list(data[data.columns[1:]])
        self.assetno = list(pd.unique(data['assetno']))[0]
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

        if(self.istrain):
            data_set,train_data,test_data = process_data(data=data,test_frac=self.training_args['test_frac'],
                                                        to_plot=self.training_args['to_plot'])
            entire_data = data_set[:,].numpy()

            diff_order = self.model_input_args['diff_order']
            net = create_cum_train_som(train_data,self.model_input_args,self.training_args)
            model_path = save_model(net,metric_names = self.metric_name,assetno=self.assetno)
            
            return model_path
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
    '''
    To visualise the multivariate data
    '''
    fig = plt.figure()
    print("Dataset has {} rows {} columns".format(data[:,].shape[0],data[:,].shape[1]))

    for i in range(data[:,].shape[-1]):
        plt.plot(data[:,i])
        plt.title("Plot of the dataset column wise")
    plt.show()   

def process_data(data,test_frac,to_plot=True):
    '''
    Function to convert the raw dataset into Timeseries class object
    Then splits the data into train and test with test_frac arg
    Returns : timeseries dataset, train_data and test_data
    '''
    data_set = TimeSeries_Dataset(data)
    train_data,test_data = split_the_data(data_set,test_frac=test_frac)
    print("Shape of Training dataset :{} and Test dataset :{}\n".format(train_data.shape,test_data.shape))
    if(to_plot):
        plot_dataset(train_data.numpy())
    return data_set,train_data,test_data

def network_dimensions(train_data,N=100):
    '''
    Calculates the 2D network shape
    The neurons are organized in a 2-dimensional map. The
    ratio of the side lengths of the map is approximately the
    ratio of the two largest eigenvalues of the training data
    covariance matrix. 
    Arguments: 
    training data - Tensor object
    N  - no of observations it's little abstract to decide this value. So its default set to 100
    Returns: 
    Dimensions of the 2D network
    '''
    
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

def test(model,evaluateData,anom_thres=3,to_plot=True):
        
        
        '''
        Function to detect anomalies (Evaluating from  the saved model)
        Arguments: 
        model -> Its instance of SOM_MODULE class imported from som_knn_module.py which is typically saved model loaded
                from user given model path
        evaluateData -> Data to detect anomalies
        anom_thres   -> Default:3, Takes only integer
        to_plot      -> Default: True, Give False (bool) to not plot the plots
        
        Returns : anomaly indexes
        '''
        
        original_data = evaluateData
        no_cols = original_data.shape[-1]
        diff_order = model.diff_order
        print("Input data's shape: {}".format(original_data.shape))
        
        # Differencing done on the data if diff_order is non zero, default diff_order=0
        res_evaluateData = np.diff(evaluateData,n=diff_order,axis=0).reshape(-1,no_cols)
        print("Differenced data shape {}".format(res_evaluateData.shape))
        
        #evaluate and get anomaly scores
        anomaly_metrics = model.evaluate(res_evaluateData)
        print(anomaly_metrics.shape)
        anomaly_metrics = anomaly_metrics/np.linalg.norm(anomaly_metrics)
        
        #Normalising the anomaly scores by l2 norm
        thres = anom_thres*(1/np.sqrt(len(anomaly_metrics)))
        
        #finding indexes where anomaly scores are greater than threshold
        selector = anomaly_metrics > thres
        
        # getting the anomaly indexes
        anom_indexes = np.arange(len(res_evaluateData))[selector]
        
        
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
            plt.hist(model.bmu_counts)
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
        
    
        no_anoms_detected = (list(selector).count(True))
        print("\nNo of anomalies detected : {}, Fraction of data detected as anomaly : {}".
              format(no_anoms_detected,no_anoms_detected/(evaluateData.shape[0])))
        return anom_indexes


def train_loop(model,train_loader,epochs):
    '''
    Function to train the model.
    Arguments:
    model: instance of the som_module class imported from som_knn_module.py
    train_loader : Its data loader using pytorch which gives out batch of inputs
    epochs: No of epochs to train the model
    
    Returns : Trained model
    '''
    curr_batch_iter = 0
    for epoch in range(epochs):
        print("Epoch : {} completed\n".format(epoch))

    for i,x_batch in enumerate(train_loader):
            curr_batch_iter += 1
            model = model.fit(x_batch,curr_batch_iter)
    
    print("\n Training successfully completed \n")
    return model


def check_default_args(inp_kwargs,def_kwargs):
    '''
    Checking Default arguments
    Arguments :
    inp_kwargs : input kwargs given by the user
    def_kwargs : default kwargs calculated from logic
    
    So it overwrites the input args if its not none 
    Returns: 
    Updated inp_kwargs
    '''
    for key in inp_kwargs:
        for def_key in def_kwargs:
            if(inp_kwargs[def_key]==None):
                inp_kwargs[def_key]=def_kwargs[def_key]
    return inp_kwargs

def create_cum_train_som(train_data,model_input_args,training_args):
    
    '''
    Function to create the model and train the som_knn model
    Arguments :
    
    train_data : training data
    model_input_args : dictionary of model_input_args
    training_args : dictionary of training_args
    
    Returns:
    Created and trained model 
    '''
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
    
    model_input_args['input_feature_size'] = train_data.shape[-1]
    model_input_args['n_iterations'] = n_iterations
    
    print("Network dimensions are {} x {} \n".format(row_dim,col_dim))
    diff_order = model_input_args['diff_order']
    del model_input_args['N']
    model = som_knn_module.Som_model(**model_input_args)

    res_train_data = (np.diff(train_data.numpy(),n=diff_order,axis=0).reshape(-1,train_data.numpy().shape[-1]))
    print("\nShape of differenced Training data : {}\n".format(res_train_data.shape))
    train_data_diff = torch.from_numpy(res_train_data)

    train_loader = torch.utils.data.DataLoader(train_data_diff, batch_size=batch_size,shuffle=True)
    
    model = train_loop(model=model,epochs=epochs,train_loader=train_loader)
    
    return model