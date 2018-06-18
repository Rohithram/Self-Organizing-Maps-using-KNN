 
import numpy as np
import error_codes as error_codes

class Type_checker():
    
    def __init__(self,kwargs,ideal_args_type):
        self.ideal_args_type = ideal_args_type
        self.kwargs = kwargs
        
    def params_checker(self):
        error_codes.reset()
        kwargs = self.kwargs
        algo_params_type = self.ideal_args_type
        for key in kwargs:
            if(type(kwargs[key])!=(algo_params_type[key])):
                error_codes.error_codes['param']['data']['argument']=key
                error_codes.error_codes['param']['data']['value']=kwargs[key]
                error_codes.error_codes['param']['message']='should be of type {}'.format((algo_params_type[key]))
                return error_codes.error_codes['param']
            else:
                continue