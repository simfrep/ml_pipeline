import importlib
import os
import pandas as pd
import logging

def func_from_string(key):
    _mod = key.rsplit('.',1)[0]
    _fun = key.rsplit('.',1)[1]
    mod = importlib.import_module(_mod)
    return getattr(mod, _fun)


def get_feat(config):
    data = get_data(config)
    bads = config['bads']
    tgt = config['target']
    
    collst =list(data.columns)
    feat = sorted(list(set(collst)-set(bads)-set(tgt)))
    return feat


def get_data(config):
    datafile = config['dpath']+config['dvarsel']
    if os.path.isfile(datafile) & config['use_binning']:
        logging.info(f"Preprocessed Datafile found {datafile}")
    elif config['use_binning'] == False:
        datafile = config['dpath']+config['data']
        logging.info(f"Use Binning is False. Use {datafile}")    
    else:
        datafile = config['dpath']+config['data']
        logging.info(f"No Preprocessed Datafile found. Default to {datafile}")

    return pd.read_parquet(datafile)


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