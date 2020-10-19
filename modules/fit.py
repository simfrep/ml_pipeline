import pandas as pd
import numpy as np
import os
import scorecardpy as sc
import joblib
import logging
from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.model_selection import ParameterGrid

from pathlib import Path

import random
import importlib


def model_config(m_dict=None):
    
    models = {}
    for key in m_dict.keys():

        para_lst = list(ParameterGrid(m_dict[key]))

        _mod = key.rsplit('.',1)[0]
        _fun = key.rsplit('.',1)[1]

        mod = importlib.import_module(_mod)
        method_to_call = getattr(mod, _fun)
        
        _models = {
            f"{_fun}_{'_'.join([f'{x}{y}' for (x,y) in list(p.items())])}":
                {'model':method_to_call(**p)} 
            for p in para_lst  
            }
        models.update(_models)

    return models


def bin_datasets(
    data,
    bins,
    tgt,
    begin_training = pd.Timestamp("2015-01-01 00:00"),
    end_training = pd.Timestamp("2019-12-31 23:00"),
    begin_valid = pd.Timestamp("2020-01-01 00:00"),
    end_valid = pd.Timestamp("2020-03-31 23:00"),
    begin_test = pd.Timestamp("2020-04-01 00:00"),
    end_test = pd.Timestamp("2020-06-15 23:00")
):
    """
        Creates and returns all necessary Datasets for normal modellung and DMatrix elements for XGB
    """   
    
    # Always ensure DF are ordered
    binned_feat = list(bins.keys())
    binned_feat.sort()
    woe_feat = [x+'_woe' for x in binned_feat]    
    
    dataset = sc.woebin_ply(data.loc[begin_training:end_training,binned_feat+[tgt]].dropna(), bins)
    valiset = sc.woebin_ply(data.loc[begin_valid:end_valid,binned_feat+[tgt]].dropna(), bins)
    testset = sc.woebin_ply(data.loc[begin_test:end_test,binned_feat+[tgt]].dropna(), bins)

    print(dataset.index.min(),dataset.index.max())
    print(valiset.index.min(),valiset.index.max())
    print(testset.index.min(),testset.index.max())

    X_train = dataset.drop(columns=[tgt])
    X_valid = valiset.drop(columns=[tgt])
    X_test  = testset.drop(columns=[tgt])

    y_train = dataset[tgt].values
    y_valid = valiset[tgt].values
    y_test  = testset[tgt].values

    return y_train,y_valid,y_test,X_train[woe_feat],X_valid[woe_feat],X_test[woe_feat],dataset,valiset,testset


def mltrain_loop(
    models,
    data,
    bins,
    tgt,
    offset_lst=[0,12,24,36],
    begin_training = pd.Timestamp("2015-01-01 00:00"),
    end_training = pd.Timestamp("2019-12-31 23:00"),
    begin_valid = pd.Timestamp("2020-01-01 00:00"),
    end_valid = pd.Timestamp("2020-03-31 23:00"),
    begin_test = pd.Timestamp("2020-04-01 00:00"),
    end_test = pd.Timestamp("2020-06-15 23:00"),
    refit_models=False
):
    
    mlist = list(models.keys())   
    random.shuffle(mlist)
    
    logging.info(f"Fitting {len(mlist)} models: {mlist}")
    # Always ensure DF are ordered
    binned_feat = list(bins.keys())
    binned_feat.sort()
    woe_feat = [x+'_woe' for x in binned_feat]
           
    for offset in offset_lst:

        _begin_training = begin_training+pd.DateOffset(months=offset)
        y_train,y_valid,y_test,X_train,X_valid,X_test ,dataset,valiset,testset = \
            bin_datasets(
                data,
                bins,
                tgt,
                begin_training = _begin_training,
                end_training = end_training,
                begin_valid = begin_valid,
                end_valid = end_valid,
                begin_test = begin_test,
                end_test = end_test
                )       
        
        i=1
        for m in mlist:

            # Specify Output filenames
            Path('models/').mkdir(parents=True, exist_ok=True)
            model_fname = "models/{}_{}.dat".format(m,_begin_training.strftime("%Y%m%d"))
            Path('metrics/').mkdir(parents=True, exist_ok=True)
            metric_fname = "metrics/{}_{}.csv".format(m,_begin_training.strftime("%Y%m%d"))

            logging.info(f"Model {m}: Starting")

            if os.path.isfile(model_fname):
                logging.info(f"Model {m}: already exists.")

                if refit_models == False:
                    logging.info(f"Model {m}: No refitting. Finished {i}/{len(mlist)}")
                    i = i+1
                    continue
                logging.info(f"Model {m}: Refit model")
            
            logging.info(f"Model {m}: Fitting started")

            model = models[m]['model']
            if 'fit_params' in models[m].keys():
                fit_params = models[m]['fit_params']
                fit_params['eval_set'] = [(X_valid[woe_feat],y_valid)]
                model.fit(X_train[woe_feat],y_train,**fit_params)
            else:
                model.fit(X_train[woe_feat],y_train)

            
            joblib.dump(model, model_fname)
            logging.info(f"Model {m}: Fitting ended")
            
            
            results = pd.DataFrame(columns=pd.MultiIndex.from_product([[_begin_training],['train', 'valid','test'], ['accuracy_score','roc_auc_score']]))

            results.loc[m,(_begin_training,'train','accuracy_score')]   = accuracy_score(y_train,model.predict(X_train[woe_feat]))
            results.loc[m,(_begin_training,'train','roc_auc_score')]   = roc_auc_score(y_train,model.predict(X_train[woe_feat]))
            
            results.loc[m,(_begin_training,'valid','accuracy_score')]   = accuracy_score(y_valid,model.predict(X_valid[woe_feat]))
            results.loc[m,(_begin_training,'valid','roc_auc_score')]   = roc_auc_score(y_valid,model.predict(X_valid[woe_feat]))

            results.loc[m,(_begin_training,'test','accuracy_score')]   = accuracy_score(y_test,model.predict(X_test[woe_feat]))
            results.loc[m,(_begin_training,'test','roc_auc_score')]   = roc_auc_score(y_test,model.predict(X_test[woe_feat]))

            results.to_csv(metric_fname)
            logging.info(f"Model {m}: Metrics Calculated")

            logging.info(f"Model {m}: Finished {i}/{len(mlist)}")
            i=i+1
    


def extract_best_model(results, ds = 'valid',metric='accuracy_score'):

    r2 = results.loc[:,(slice(None),ds,metric)].unstack()

    bestmodel = pd.DataFrame()
    for m in set(r2.index.get_level_values(0)):
        bestmodel_begin = None
        bestmodel_acc = None
        for i,x in pd.DataFrame(r2.loc[m]).reset_index().iterrows():
            if bestmodel_begin is None:
                bestmodel_begin = x['level_2']
                bestmodel_acc = x[0]
            if x[0] > bestmodel_acc:
                bestmodel_begin = x['level_2']
                bestmodel_acc = x[0]

        bestmodel.loc[m,'begin'] = bestmodel_begin
        bestmodel.loc[m,metric] = bestmodel_acc
        #print(m,bestmodel_begin,bestmodel_acc)
    #bestmodel

    _r3 = bestmodel.reset_index().sort_values(by=metric, ascending=False)

    for i,x in _r3.iterrows():
        _r3.loc[i,'model_fname'] = "models/{}_{}.dat".format(x['index'],x['begin'].strftime("%Y%m%d"))
    return _r3
