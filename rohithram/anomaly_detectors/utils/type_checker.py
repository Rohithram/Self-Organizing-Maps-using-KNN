
# importing required modules 
# error_codes is python file which has errorcodes dictionary
import numpy as np
from anomaly_detectors.utils import error_codes as error_codes

class Type_checker():
    
    def __init__(self,kwargs,ideal_args_type):
        
        '''
        Class Type_checker is used to check parameter mismatches and takes any set of argument dictionary and
        corresponding argument types
        '''
        self.ideal_args_type = ideal_args_type
        self.kwargs = kwargs
        
    def params_checker(self):
        
        '''
        Function to check the parameters
        and returns the corresponding error message when mismatch encountered
        It also checks for probability threshold between 0 and 1
        '''
        error_codes.reset()
        kwargs = self.kwargs
        algo_params_type = self.ideal_args_type
        for key in kwargs:
            try:
                if(key=='pthres'):
                    if(type(kwargs[key])==int or type(kwargs['pthres'])==float):
                        if(kwargs[key]<1 and kwargs['pthres']>0):
                            continue
                        else:
                            error_codes.error_codes['param']['data']['argument']='pthres'
                            error_codes.error_codes['param']['data']['value']=kwargs['pthres']
                            error_codes.error_codes['param']['message']='probability must be between 0 and 1 and it must be of type int or float'
                            return error_codes.error_codes['param']
                        
            except:
                pass
            
            if(kwargs[key]!=None and type(kwargs[key])!=(algo_params_type[key])):
                error_codes.error_codes['param']['data']['argument']=key
                error_codes.error_codes['param']['data']['value']=kwargs[key]
                error_codes.error_codes['param']['message']='should be of type {}'.format((algo_params_type[key]))
                return error_codes.error_codes['param']
            
            else:
                continue