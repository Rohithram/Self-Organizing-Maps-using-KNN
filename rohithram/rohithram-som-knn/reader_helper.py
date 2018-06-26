

import error_codes
import reader



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
#         try:
#             response_json=reader.reader_api(**self.reader_kwargs)
#             response_dict = json.loads(response_json)
#         except Exception as e:
#             error_codes.error_codes['data_missing']['message'] = response_json
#             return error_codes.error_codes['data_missing']


        '''
        To read from old reader file
        '''
        response_dict=reader.reader_api(**reader_kwargs)

        return response_dict