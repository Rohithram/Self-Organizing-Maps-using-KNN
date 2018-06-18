

import bayeschangept_sprint1 as bayeschangept
import warnings
warnings.filterwarnings('ignore')

assetno = ['1']
con = '52.173.76.89:4242'
src_type =  'opentsdb'
param = ['FE-001.DRIVEENERGY']
from_timestamp = 1520402214
to_timestamp = 1520407294

res = bayeschangept.call(**kwargs)
print(res)
# del kwargs['thres_prob']
# del kwargs['samples_to_wait']
res = bayeschangept.call(**kwargs)
print(res)
del kwargs['thres_prob']
del kwargs['samples_to_wait']
res = bayeschangept.call(**kwargs)
print(res)
# del kwargs['thres_prob']
del kwargs['samples_to_wait']
res = bayeschangept.call(**kwargs)
print(res)

res = bayeschangept.call(**kwargs)
print(res)

res = bayeschangept.call(**kwargs())
print(res)
# del kwargs['thres_prob']
del kwargs['samples_to_wait']
res = bayeschangept.call(**kwargs)
print(res)
# del kwargs['thres_prob']
del kwargs()['samples_to_wait']
res = bayeschangept.call(**kwargs)
print(res)
# del kwargs['thres_prob']
del kwargs()['samples_to_wait']
res = bayeschangept.call(**kwargs())
print(res)
del kwargs()['thres_prob']
del kwargs()['samples_to_wait']
del kwargs()['mean_runlen']
res = bayeschangept.call(**kwargs())
print(res)
del kwargs()['thres_prob']
del kwargs()['samples_to_wait']
del kwargs()['expected_run_length']
res = bayeschangept.call(**kwargs())
print(res)
del kwargs()['thres_prob']
del kwargs()['samples_to_wait']
del kwargs()['expected_run_length']
res = bayeschangept.call(**kwargs())
print(res)
kwargs = kwargs()
del kwargs['thres_prob']
del kwargs['samples_to_wait']
del kwargs['expected_run_length']
res = bayeschangept.call(**kwargs())
print(res)
kwargs = kwargs()
del kwargs['thres_prob']
del kwargs['samples_to_wait']
del kwargs['expected_run_length']
res = bayeschangept.call(**kwargs())
print(res)
kwargs1 = kwargs()
del kwargs1['thres_prob']
del kwargs1['samples_to_wait']
del kwargs1['expected_run_length']
res = bayeschangept.call(**kwargs())
print(res)
kwargs1 = kwargs()
del kwargs1['thres_prob']
del kwargs1['samples_to_wait']
del kwargs1['expected_run_length']
res = bayeschangept.call(**kwargs1
print(res)
kwargs1 = kwargs()
del kwargs1['thres_prob']
del kwargs1['samples_to_wait']
del kwargs1['expected_run_length']
res = bayeschangept.call(**kwargs1)
print(res)

res = bayeschangept.call(**kwargs())
print(res)
kwargs1 = kwargs()
del kwargs1['thres_prob']
del kwargs1['samples_to_wait']
del kwargs1['expected_run_length']
res = bayeschangept.call(**kwargs1)
print(res)