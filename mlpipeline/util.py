import importlib
import os
import pandas as pd
import logging

def func_from_string(key):
    _mod = key.rsplit('.',1)[0]
    _fun = key.rsplit('.',1)[1]
    mod = importlib.import_module(_mod)
    return getattr(mod, _fun)

def extract_best_model(config,metric='accuracy_score',ds='valid'):
    modelpath = config['mpath']+'models/'
    metricpath = config['mpath']+'metrics/'
    results=pd.DataFrame()
    for file in os.listdir(metricpath):
        #print(file)
        _r = pd.read_csv(f"{metricpath}{file}",header=[0,1,2],index_col=[0],parse_dates=True)
        results = pd.concat([results,_r.stack(level=0)],axis=0)

        results.index.set_names(['model', 'begin'],inplace=True)

        res = results[(ds,metric)].reset_index()

    ilst = []
    for group in res.groupby('model'):
        #display(group[1])
        imax = group[1][(ds,metric)].idxmax()
        #print(imax)
        ilst.append(imax)

    bestmodel = res.loc[ilst]
    bestmodel.columns = ['model','begin',metric]
    bestmodel['model_fname'] = [f"{modelpath}{m}_{pd.Timestamp(b).strftime('%Y%m%d')}.dat" for m,b in zip(bestmodel.model,bestmodel.begin)]

    return bestmodel.sort_values(by=metric,ascending=False)