from setuptools import setup,find_packages

setup(name='anomaly_detectors',
      version='v.01',
      description='Anomaly Detection algorithms like som knn and bayes changepoint',
      author='Rohithram R',
      author_email='rohithram.r@flutura.com',
      url='',
      packages = find_packages(),
      requires=['scipy', 'numpy','pandas','bayesian_changepoint_detection']
     )
