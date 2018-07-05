

from anomaly_detectors.utils.error_codes import error_codes
from anomaly_detectors.reader_writer import reader_new as reader
import traceback
import json



def read(reader_kwargs):
        '''
        Function to read the data using reader api, and parses the json to list of dataframes per asset
        '''
#         response_json=reader.reader_api(**self.reader_kwargs)
#         response_dict = json.loads(response_json)
#         print(response_dict)
#         response_dict=reader.reader_api(**reader_kwargs)
    
        '''
        To do when new reader works with csv file
        '''
        response_json = ''
        try:
            response_json=reader.reader_api(**reader_kwargs)
#             print("\nResponse from reader: \n{}\n".format(response_json))
#             response_dict = json.loads(response_json)
            return response_json
        except Exception as e:
            traceback.print_exc()
            return str(response_json)


#         '''
#         To read from old reader file
#         '''
#         response_dict=reader.reader_api(**reader_kwargs)