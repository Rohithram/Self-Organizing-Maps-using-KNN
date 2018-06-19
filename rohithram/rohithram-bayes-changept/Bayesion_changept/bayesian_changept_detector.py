

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing modules to run the algo
import cProfile
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import matplotlib.cm as cm

class Bayesian_Changept_Detector():
    def __init__(self,data,assetno,is_train=False,data_col_index=0,pthres=0.5,mean_runlen = 100,Nw=10):
        
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
        '''
        
        
        self.algo_name = 'bayesian_change_point_detection'
        self.algo_code = 'bcp'
        self.algo_type = 'univariate'
        self.istrainable = is_train
        self.data = data
        self.data_col_index = data_col_index
        self.metric_name = data.columns[data_col_index]
        self.assetno = assetno
        self.pthres = pthres
        self.mean_runlen = mean_runlen
        self.Nw = Nw


    def detect_anomalies(self):
        
        '''
        Detects anomalies and returns data and anomaly indexes
        '''
        data = self.data
        print("Shape of the dataset : ")
        print(data.shape)
        print("Overview of first five rows of dataset : ")
    #     print(data.head())
        ncol = self.data_col_index
        
#         ax = data[data.columns[ncol]].plot.hist(figsize=(9,7),bins=100)
#         ax.set_title("Histogram of Dataset")

        R,maxes = self.findonchangepoint(data[data.columns[ncol]].values)
        anom_indexes = self.plotonchangepoints(R,maxes)
        self.anom_indexes = anom_indexes
        return data,anom_indexes
    

    def findonchangepoint(self,data):
        '''
        finds the changepoints and returns the run lenth probability matrix and indexes of maximum run lengths
        probability
        '''
        R, maxes = oncd.online_changepoint_detection(data, partial(oncd.constant_hazard,self.mean_runlen),
                                                     oncd.StudentT(0.1, .01, 1, 0))
        return R,maxes
    

    def findthreshold(self,data):
        
        '''
        finds inversion points where probability is greater than mean
        Returns -> list of inversion points
        '''
        mu = np.mean(data)
        sigma = np.mean(data)
        inv_pt = []
        for i in range(len(data)-1):
            if((data[i+1]>mu and data[i]<=mu) or (data[i+1]<mu and data[i]>=mu)):
                inv_pt.append(i)

        return inv_pt    
    

    def plotonchangepoints(self,R,maxes,nrow=None):
        '''
        plots the original data and anomaly indexes as vertical line
        and plots run length distribution and probability score for each possible run length
        '''
        fig,(ax1,ax3) = plt.subplots(2,figsize=[18, 16])
        ncol = self.data_col_index
        data = self.data
        Nw = self.Nw
        pthres = self.pthres
        
        ltext = 'Column : '+str(ncol+1)+' data with threshold probab = '+ str(pthres)

        ax1.set_title(data.columns[ncol])

        cp_probs = np.array(R[Nw,Nw:-1][1:-2])

        inversion_pts = self.findthreshold(cp_probs)

        max_indexes = []
        for i in range(len(inversion_pts)-1):
            max_indexes.append(inversion_pts[i]+np.argmax(cp_probs[inversion_pts[i]:inversion_pts[i+1]+1]))

        cp_mapped_probs = pd.Series(cp_probs[max_indexes],index=max_indexes)
        anom_indexes = cp_mapped_probs.index[(np.where(cp_mapped_probs.values>pthres)[0])]

        if(nrow==None):
            ax1.plot(data.values[:,ncol],label=ltext)
        else:
            ax1.plot(data.values[:nrow,ncol],label=ltext)

        ax1.legend()


        for a in anom_indexes:
            if(a):
                ax1.axvline(x=a,color='r')

#         sparsity = 5  # only plot every fifth data for faster display
#         ax2.pcolor(np.array(range(0, len(R[:,0]), sparsity)), 
#                   np.array(range(0, len(R[:,0]), sparsity)), 
#                   -np.log(R[0:-1:sparsity, 0:-1:sparsity]), 
#                   cmap=cm.Greys, vmin=0, vmax=30,label="Distribution of Run length")
#         ax2.legend()

        ax3.plot(cp_probs)

        ax3.set_title('Change points with Probability')

        plt.show()
        print("\n No of Anomalies detected = %g"%(len(anom_indexes)))

        return anom_indexes