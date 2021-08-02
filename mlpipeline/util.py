import importlib
import os
import pandas as pd
import logging

def func_from_string(key):
    _mod = key.rsplit('.',1)[0]
    _fun = key.rsplit('.',1)[1]
    mod = importlib.import_module(_mod)
    return getattr(mod, _fun)

class Util():
    def __init__(
        self,
        config = None
    ):
        self.config = config
    
    def extract_best_model(self,ds='test'):
        modelpath = f"{self.config.data.outpath}/models"
        metricpath = f"{self.config.data.outpath}/metrics"
        results=pd.DataFrame()
        for file in os.listdir(metricpath):
            _r = pd.read_csv(f"{metricpath}/{file}",header=[0],index_col=[0,1,2],parse_dates=False)
            results = pd.concat([results,_r],axis=0)
            

        results.index.set_names(['model','iteration','metric'],inplace=True)
        
        results = results.sort_index().reset_index()
        results['model_fname'] = [f"{modelpath}{m}_{i}.dat" for m,i in zip(results.model,results.iteration)]
        results = results.set_index('metric')

        d = {}
        for metric in dir(self.config.metrics):
            _results = results.loc[metric]
            d[metric] = _results.sort_values(by=ds,ascending=False)
        return d