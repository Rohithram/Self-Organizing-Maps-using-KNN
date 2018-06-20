
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

#torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
from sklearn.neighbors import NearestNeighbors

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import scipy as sp

import matplotlib.pyplot as plt
from sklearn import preprocessing 
from matplotlib.pylab import rcParams
import datetime as dt
import argparse
import time
from seasonal import fit_seasons,adjust_seasons
from scipy.signal import detrend

from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

#Importing reader and checker python files as modules
import reader as reader
import checker as checker
import writer_configs as write_args
import psycopg2

import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 12, 9
rcParams[ 'axes.grid']=True
plt.ion()   # interactive mode
class Som_net():
    
    def __init__(self,som_shape,input_feature_size,time_constant,n_iterations,
                 minNumPerBmu=1,no_of_neighbors=3,initial_radius=1,initial_learning_rate=0.4):
        super(Som_net, self).__init__()
        
        self.shape = som_shape
        self.weight_dim = self.shape.__len__()
        self.feature_size = input_feature_size
        self.time_constant = time_constant
        self.initial_radius = initial_radius
        self.initial_learning_rate = initial_learning_rate
        self.weights = torch.rand((*self.shape,self.feature_size),dtype=torch.float64)
        self.bmu_counts = torch.zeros((*self.shape),dtype=torch.int32)
        self.n_iterations = n_iterations
        self.neuron_locations = self.neuron_locations(*self.shape)
        self.minNumPerBmu = minNumPerBmu
        self.no_of_neighbors = no_of_neighbors
#         self.allowed_nodes = torch.from_numpy(np.array())
        
    def findBMU(self,x_batch):
        """
         Find the best matching unit for a specific batch of samples
        :param x_batch: The data points for which the best matching unit should be found.
        :type x_batch: numpy.ndarray
        :return: numpy.ndarray with index
        """
        
        batch_size = len(x_batch)
        bmu_indexes = np.zeros((batch_size,2),dtype=int)
#         print(bmu_indexes[0].shape)
        for i in range(batch_size):
#             print(np.square(self.weights.numpy()-x_batch[i].numpy()).sum(axis=-1))
            bmu_dists = np.square(self.weights.numpy()-x_batch[i].numpy()).sum(axis=-1)
            arg_min_ind = np.argmin(bmu_dists)
#             print(arg_min_ind)
            bmu_indexes[i] = np.unravel_index(arg_min_ind,bmu_dists.shape)
#             print(bmu_indexes[i])
#             print(self.bmu_counts[])
            
            self.bmu_counts[np.unravel_index(arg_min_ind,bmu_dists.shape)] +=1

                           
#         print("Bmu indexes shape {}".format(bmu_indexes.shape))
        return bmu_indexes
    
    
    def fit(self,train_batch,curr_batch_iter):
        
        """Train the SOM to a specific dataset.
        :param train_batch: The complete training dataset
        :type train_batch: 2d ndarray
        :param num_iterations: The number of iterations used for training
        :type num_iterations: int
        :return: a reference to the object
        """
        
        bmu_indexes = self.findBMU(train_batch)
        
        curr_iter = np.array([curr_batch_iter+i for i in range(len(train_batch))])
        # Update the parameters to let them decay to 0
               
        r_batch = self.decay_radius((curr_iter))
        l_batch = self.decay_learning_rate(curr_iter)
        
    
        self.update_weights(train_batch, bmu_indexes, r_batch, l_batch)
        self.allowed_nodes = self.weights[self.bmu_counts >= self.minNumPerBmu]

        return self

    def evaluate(self,evaluationData):
        """
        This function maps the evaluation data to the previously fitted network. It calculates the anomaly measure
        based on the distance between the observation and the K-NN nodes of this observation.
        :param evaluationData: Numpy array of the data to be evaluated
        :return: 1D-array with for each observation an anomaly measure
        """
        try:
            self.allowed_nodes
            assert self.allowed_nodes.shape[0] > 1
        except NameError:
            raise Exception("Make sure the method fit is called before evaluating data.")
        except AssertionError:
            raise Exception("There are no nodes satisfying the minimum criterium, algorithm cannot proceed.")
        else:
            classifier = NearestNeighbors(n_neighbors=self.no_of_neighbors)
            classifier.fit(self.allowed_nodes)
            dist, _ = classifier.kneighbors(evaluationData)
        return dist.mean(axis=1)
    
    
    def neuron_locations(self,m,n):
        r0 = np.arange(m) # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
        r1 = np.arange(n)
        out = np.empty((m,n,2),dtype=int)
        out[:,:,0] = r0[:,None]
        out[:,:,1] = r1
        return out

    def decay_radius(self,i):
        return self.initial_radius * np.exp(-(1*i)/self.time_constant)

    
    def decay_learning_rate(self, i):
        return self.initial_learning_rate * np.exp(-(1*i)/self.n_iterations)

    
    def calculate_influence(self,distance, radius,index):
        return np.exp(-distance / (2* (radius**2)))

    
    def update_weights(self,train_batch, bmu_indexes,radius,learning_speed):
        
        '''
        # now we know the BMU, update its weight vector to move closer to input
        # and move its neighbours in 2-D space closer
        # by a factor proportional to their 2-D distance from the BMU
        '''
        
        batch_size = len(train_batch)
        
        #print("BMU Index for batch {}".format(bmu_idx))
        network_indexes = self.neuron_locations
        network_indexes = np.stack([network_indexes for i in range(batch_size)],axis=0)
#         print("Network shape {}".format(network_indexes.shape))

        bmu_indexes = bmu_indexes.reshape(batch_size,1,1,2)
        learning_speed = learning_speed.reshape(batch_size,1,1,1)
        
        w_dists_coordinates = (np.square(network_indexes - bmu_indexes))
        
        w_dists = (w_dists_coordinates[:,:,:,0]+w_dists_coordinates[:,:,:,1])

#         print("Shape of distance of neurons from bmu {}".format(w_dists.shape))

        bool_index = np.array([(w_dists[i]<=r2) for i,r2 in enumerate(radius**2)])
        
        influence = np.array([self.calculate_influence(w_dists[i][bool_index[i]],radius[i],bool_index[i]) 
                              for i in range((batch_size))])
        
        influence_neurons = np.zeros(w_dists.shape)
#         print(influence.shape)
        influence = influence.reshape(-1,)
#         print(influence)
        try:
            influence_neurons[bool_index] = influence
            influential_neurons = np.stack([influence_neurons for i in range(self.feature_size)],axis=-1)

    #         print("Influtential neurons shape {}".format(influential_neurons.shape))

            learningMatrix = np.array([-torch.add(self.weights,-1*(train_batch[k])).numpy() for k in range(batch_size)])

    #         print("Learning matrix shape {}".format(learningMatrix.shape))

    #         scaledLearningMatrix = np.zeros((batch_size,*weights.shape))
            scaledLearningMatrix = learning_speed * (influential_neurons * learningMatrix)

    #         print("Scaled lmatrix {}".format(scaledLearningMatrix.shape))

            [torch.add(self.weights,torch.from_numpy(scaledLearningMatrix)[k],out=self.weights) for k in range(batch_size)]
        except:
            return
        return