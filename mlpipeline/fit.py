import pandas as pd
import numpy as np
import os
import scorecardpy as sc
import joblib
import logging
from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from pathlib import Path


import random
import importlib

from .util import func_from_string, get_feat, get_data

class Fitting():

    def __init__(
        self,
        config = None
    ):
        self.config = config


    def model_config(self):
        m_dict=self.config['models']
        if 'preprocessing' in self.config.keys():
            preprocessing = self.config['preprocessing']

        models = {}

        if self.config['use_binning'] in [False, 'Both']:
            for trans in preprocessing.keys():
                trans_func = func_from_string(trans)
                for key in m_dict.keys():
                    para_lst = list(ParameterGrid(m_dict[key]))
                    model_func = func_from_string(key)

                    if trans_func.__name__ == 'BinningProcess':
                        _models = {
                            f"{trans_func.__name__}_{model_func.__name__}_{'_'.join([f'{x}{y}' for (x,y) in list(p.items())])}":
                                {'model':Pipeline([('transformation',trans_func(get_feat(self.config))),('model',model_func(**p))])} 
                            for p in para_lst  
                            }
                    else:                    
                        _models = {
                            f"{trans_func.__name__}_{model_func.__name__}_{'_'.join([f'{x}{y}' for (x,y) in list(p.items())])}":
                                {'model':Pipeline([('transformation',trans_func()),('model',model_func(**p))])} 
                            for p in para_lst  
                            }
                    models.update(_models)
        
        if self.config['use_binning'] in [True, 'Both']:
            for key in m_dict.keys():
                para_lst = list(ParameterGrid(m_dict[key]))
                model_func = func_from_string(key)
                _models = {
                    f"woe_{model_func.__name__}_{'_'.join([f'{x}{y}' for (x,y) in list(p.items())])}":
                        {'model':Pipeline([('model',model_func(**p))])} 
                    for p in para_lst  
                    }
                models.update(_models)


        return models

    def split_datasets(
        self,
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

        data = get_data(self.config)
        use_binning = self.config['use_binning']
        bads = self.config['bads']
        tgt = self.config['target']
        
        collst =list(data.columns)
        feat = sorted(list(set(collst)-set(bads)-set(tgt)))

        if use_binning:       
            bfile = self.config['dpath']+self.config['dbins']
            bins = joblib.load(bfile)

            # Always ensure DF are ordered
            binned_feat = list(bins.keys())
            binned_feat.sort()
            feat = [x+'_woe' for x in binned_feat]    
            
            dataset = sc.woebin_ply(data.loc[begin_training:end_training,binned_feat+[tgt]].dropna(), bins)
            valiset = sc.woebin_ply(data.loc[begin_valid:end_valid,binned_feat+[tgt]].dropna(), bins)
            testset = sc.woebin_ply(data.loc[begin_test:end_test,binned_feat+[tgt]].dropna(), bins)

        else:
            dataset = data.loc[begin_training:end_training,feat+[tgt]].dropna()
            valiset = data.loc[begin_valid:end_valid,feat+[tgt]].dropna()
            testset = data.loc[begin_test:end_test,feat+[tgt]].dropna()

        X_train = dataset.drop(columns=[tgt])[feat]
        X_valid = valiset.drop(columns=[tgt])[feat]
        X_test  = testset.drop(columns=[tgt])[feat]
        y_train = dataset[tgt].values
        y_valid = valiset[tgt].values
        y_test  = testset[tgt].values

        logging.info(f"Training Dataset {dataset.index.min()} {dataset.index.max()}")
        logging.info(f"Validation Dataset {valiset.index.min()} {valiset.index.max()}")
        logging.info(f"Test Dataset {testset.index.min()} {testset.index.max()}")

        return y_train,y_valid,y_test,X_train,X_valid,X_test,dataset,valiset,testset

    def mltrain_loop(
        self,
        refit_models=False,
        use_binning=False,
        **kwargs
    ):

        if self.config is None:
            logging.info("No self.config file provided")
        else:
            logging.info("self.config file provided")

            models = self.model_config()
            data = get_data(self.config)
            
            cfg = self.config['begin_training']
            logging.debug(f"Config begin_training: {cfg} of type {type(cfg)}")
            if type(cfg) == list:
                training_begins = [pd.Timestamp(x) for x in cfg]
            else:
                offset_lst= self.config['offset_lst']
                offset_res= self.config['offset_res']
                training_begins = [pd.Timestamp(cfg)+pd.DateOffset(**{offset_res: offset}) for offset in offset_lst]
            logging.debug(f"List of Training Begins: {training_begins}")

            end_training    = pd.Timestamp(self.config['end_training'])
            begin_valid     = pd.Timestamp(self.config['begin_valid'])
            end_valid       = pd.Timestamp(self.config['end_valid'])
            begin_test      = pd.Timestamp(self.config['begin_test'])
            end_test        = pd.Timestamp(self.config['end_test'])     
            
        mlist = list(models.keys())   
        random.shuffle(mlist)
        
        logging.info(f"Fitting {len(mlist)} models for {len(training_begins)} training begin. Total: {len(mlist)*len(training_begins)}")
        logging.debug(f"List of models {mlist}")
        
        cnt_begins = 1
        for begin_training in training_begins:

            y_train,y_valid,y_test,X_train,X_valid,X_test ,dataset,valiset,testset = \
                self.split_datasets(
                    begin_training = begin_training,
                    end_training = end_training,
                    begin_valid = begin_valid,
                    end_valid = end_valid,
                    begin_test = begin_test,
                    end_test = end_test
                    )       
            
            cnt_models=1
            for m in mlist:

                # Specify Output filenames
                Path('models/').mkdir(parents=True, exist_ok=True)
                model_fname = "models/{}_{}.dat".format(m,begin_training.strftime("%Y%m%d"))
                Path('metrics/').mkdir(parents=True, exist_ok=True)
                metric_fname = "metrics/{}_{}.csv".format(m,begin_training.strftime("%Y%m%d"))

                logging.debug(f"Model {m}: Starting")

                if os.path.isfile(model_fname):
                    if refit_models == False:
                        logging.info(f"{cnt_begins}/{len(training_begins)} {cnt_models}/{len(mlist)} Completed Model {m} already exists. No refitting.")
                        cnt_models = cnt_models+1
                        if cnt_models == len(mlist):
                            cnt_begins = cnt_begins + 1
                        continue
                    logging.debug(f"Model {m}: Refit model")
                
                logging.debug(f"Model {m}: Fitting started")

                model = models[m]['model']

                if 'fit_params' in models[m].keys():
                    fit_params = models[m]['fit_params']
                    fit_params['eval_set'] = [(X_valid,y_valid)]
                    model.fit(X_train,y_train,**fit_params)
                else:
                    model.fit(X_train,y_train)
                
                joblib.dump(model, model_fname)
                logging.debug(f"Model {m}: Fitting ended")
                
                
                results = pd.DataFrame(columns=pd.MultiIndex.from_product([[begin_training],['train', 'valid','test'], ['accuracy_score','roc_auc_score']]))

                results.loc[m,(begin_training,'train','accuracy_score')]   = accuracy_score(y_train,model.predict(X_train))
                results.loc[m,(begin_training,'train','roc_auc_score')]   = roc_auc_score(y_train,model.predict(X_train))
                
                results.loc[m,(begin_training,'valid','accuracy_score')]   = accuracy_score(y_valid,model.predict(X_valid))
                results.loc[m,(begin_training,'valid','roc_auc_score')]   = roc_auc_score(y_valid,model.predict(X_valid))

                results.loc[m,(begin_training,'test','accuracy_score')]   = accuracy_score(y_test,model.predict(X_test))
                results.loc[m,(begin_training,'test','roc_auc_score')]   = roc_auc_score(y_test,model.predict(X_test))

                results.to_csv(metric_fname)
                logging.debug(f"Model {m}: Metrics Calculated")

                logging.info(f"{cnt_begins}/{len(training_begins)} {cnt_models}/{len(mlist)} Completed Model {m}")
                cnt_models=cnt_models+1
            cnt_begins = cnt_begins + 1
