
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import datetime as dt

import time
# Importing reader and checker python files as modules
import reader_writer.db_properties as db_props
import reader_writer.writer_configs as write_args
import psycopg2

from utils.preprocessors import *
from utils.data_handler import *
from utils.bayesian_changept_detector import *
import json
from pandas.io.json import json_normalize

import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 12, 9
rcParams[ 'axes.grid']=True
input_kwargs = {
    'src_type'       :src_type,
    'filepath'      :filepath,
    'date_col'      :date_col,
    'pthres'        :pthres,
    'time_format'   :time_format,
    'weekly_data'   :weekly_data,
    'mean_runlen'   :mean_runlen,
    'Nw'            :Nw
}
input_kwargs['src_type'] = 1
original_data,anom_indexes = analyse_detectchangepts(**input_kwargs)