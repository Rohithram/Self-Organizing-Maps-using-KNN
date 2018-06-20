
import pandas as pd
import numpy as np
import os
# importing the bayesian changepoint main python file to detect changepoints
import bayeschangept_sprint1 as bayeschangept
import warnings
warnings.filterwarnings('ignore')

assetno = ['1']
con = '52.173.76.89:4242'
src_type =  'opentsdb'
param = ['FE-001.DRIVEENERGY']
from_timestamp = 1520402214
to_timestamp = 1520407294
kwargs1 = kwargs()
for key in keys:
    kwargs1[key]=''
    res = bayeschangept.call(**kwargs1)
    print(res)

kwargs1 = kwargs()
for key in keys:
    del kwargs1[key]
    res = bayeschangept.call(**kwargs1)
    print(res)
kwargs1 = kwargs()
val = ['2',4.5,'def']
for i,key in enumerate(keys):
    kwargs1[key]=val[i]
    res = bayeschangept.call(**kwargs1)
    print(res)
kwargs1 = get_csv_kwargs()
pthreses = [0.5,0.0,1.0]
for i,pthres in enumerate(pthreses):
    kwargs1['thres_prob']=pthres
    res = bayeschangept.call(**kwargs1)
    print(res)