import pandas as pd
import os
import dill
import logging
from sklearn.metrics import accuracy_score,roc_auc_score,mean_squared_error

from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from pathlib import Path


import random
import importlib

from .util import func_from_string, get_feat, get_data

from concurrent.futures import ProcessPoolExecutor, as_completed

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
        
        return models

    def split_datasets_byts(
        self,
        begin_training = pd.Timestamp("2015-01-01 00:00"),
        begin_valid = pd.Timestamp("2020-01-01 00:00"),
        begin_test = pd.Timestamp("2020-04-01 00:00"),
    ):
        """
            Creates and returns all necessary Datasets for normal modellung and DMatrix elements for XGB
        """   

        data = get_data(self.config)
        bads = self.config.bads
        tgt = self.config.target
        
        collst =list(data.columns)
        feat = sorted(list(set(collst)-set(bads)-set(tgt)))

        dataset = data.loc[begin_training:begin_valid,feat+[tgt]].dropna()
        valiset = data.loc[begin_valid:begin_test,feat+[tgt]].dropna()
        testset = data.loc[begin_test:,feat+[tgt]].dropna()
        X_train = dataset.drop(columns=[tgt])[feat]
        X_valid = valiset.drop(columns=[tgt])[feat]
        X_test  = testset.drop(columns=[tgt])[feat]
        y_train = dataset[tgt].values
        y_valid = valiset[tgt].values
        y_test  = testset[tgt].values
        logging.debug(f"Training Dataset {dataset.index.min()} {dataset.index.max()}")
        logging.debug(f"Validation Dataset {valiset.index.min()} {valiset.index.max()}")
        logging.debug(f"Test Dataset {testset.index.min()} {testset.index.max()}")

        return y_train,y_valid,y_test,X_train,X_valid,X_test,dataset,valiset,testset

    def split_datasets_byratio(
        self,split :list= [0.75,0.15,0.1]
    ):        
        data = get_data(self.config)
        bads = self.config.bads
        tgt = self.config.target
        
        collst =list(data.columns)
        feat = sorted(list(set(collst)-set(bads)-set(tgt)))

        logging.info(f"Using Split: {split}")
        train_ratio = split[0]
        validation_ratio = split[1]
        test_ratio = split[2]
        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        X_train, _X, y_train, _y = train_test_split(data[feat], data[tgt], test_size=1 - train_ratio)
        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        X_valid, X_test, y_valid, y_test = train_test_split(_X, _y, test_size=test_ratio/(test_ratio + validation_ratio)) 
        dataset = pd.concat([X_train,y_train],axis=1)
        valiset = pd.concat([X_valid,y_valid],axis=1)
        testset = pd.concat([X_test,y_test],axis=1)
        logging.debug(f"Training Dataset {dataset.shape}")
        logging.debug(f"Validation Dataset {valiset.shape}")
        logging.debug(f"Test Dataset {testset.shape}")

        return y_train,y_valid,y_test,X_train,X_valid,X_test,dataset,valiset,testset

    def mltrain_loop(
        self,
        refit_models=False,
        **kwargs
    ):

        models = self.model_config()
        mlist = list(models.keys())   
        random.shuffle(mlist)

        data = get_data(self.config)

        logging.debug(self.config)
        logging.debug(self.config.keys())


        if 'split' in dir(self.config):
            split = self.config.split
            if type(split) == list:
                y_train,y_valid,y_test,X_train,X_valid,X_test ,dataset,valiset,testset = \
                self.split_datasets_byratio(split)
                # Create pools for parallel execution
                pool = ProcessPoolExecutor(self.config['parallel_threads'])

                # This list contains all functions that are executed
                futures = []
                cnt_begins = 1
                cnt_totals = 0
                cnt_models = 0
                trainingiterations = 1
                total_models = len(mlist)*trainingiterations
                for x in mlist:
                    futures.append(pool.submit(self.mfit, x,models,"RatioSplit",X_train,y_train,X_valid,y_valid,X_test,y_test))

                # This loop ensures that we wait until all models are fitted
                for x in as_completed(futures):
                    cnt_models += 1
                    cnt_totals += 1
                    logging.debug(x.result())
                    logging.debug(f"{cnt_begins}/{trainingiterations} {cnt_models}/{len(mlist)} Completed Models")
                    if (cnt_models % 10 == 0) or (cnt_models == len(mlist)):
                        logging.info(f"Finished {cnt_totals} of {total_models} model fittings (Training Begin {cnt_begins}/{trainingiterations} Model {cnt_models}/{len(mlist)})")
            
            elif 'time' in dir(split):
                time = split.time
                begin = split.time.begin_training
                logging.debug(f"Config begin_training: {begin} of type {type(begin)}")
                if type(begin) == list:
                    training_begins = [pd.Timestamp(x) for x in begin]
                else:
                    offset_lst= time.offset_lst
                    offset_res= time.offset_res
                    training_begins = [pd.Timestamp(begin)+pd.DateOffset(**{offset_res: offset}) for offset in offset_lst]
                logging.debug(f"List of Training Begins: {training_begins}")

                begin_valid     = pd.Timestamp(time.begin_valid)
                begin_test      = pd.Timestamp(time.begin_test)

                trainingiterations = len(training_begins)
                total_models = len(mlist)*trainingiterations
                logging.info(f"Fitting {len(mlist)} models for {trainingiterations} training datasets. Total: {total_models}")
                logging.debug(f"List of models {mlist}")
                cnt_begins = 1
                cnt_totals = 0
                for begin_training in training_begins:           
                    y_train,y_valid,y_test,X_train,X_valid,X_test ,dataset,valiset,testset = \
                        self.split_datasets_byts(
                            begin_training = begin_training,
                            begin_valid = begin_valid,
                            begin_test = begin_test,
                            )

                    trainingid = begin_training.strftime("%Y%m%d")
                    # Create pools for parallel execution
                    pool = ProcessPoolExecutor(self.config['parallel_threads'])

                    # This list contains all functions that are executed
                    futures = []
                    cnt_models = 0
                    
                    for x in mlist:
                        futures.append(pool.submit(self.mfit, x,models,trainingid,X_train,y_train,X_valid,y_valid,X_test,y_test))

                    # This loop ensures that we wait until all models are fitted
                    for x in as_completed(futures):
                        cnt_models += 1
                        cnt_totals += 1
                        logging.debug(x.result())
                        logging.debug(f"{cnt_begins}/{len(training_begins)} {cnt_models}/{len(mlist)} Completed Models")
                        if (cnt_models % 10 == 0) or (cnt_models == len(mlist)):
                            logging.info(f"Finished {cnt_totals} of {total_models} model fittings (Training Begin {cnt_begins}/{trainingiterations} Model {cnt_models}/{len(mlist)})")
                    
                    cnt_begins += 1
            else:
                raise ValueError("self.config.split must be either list or split by times")        
        else:
            logging.info("No splits defined use default split") 
        logging.info(f"Finished Model Fitting.")

    def mfit(self,m,models,trainingid,X_train,y_train,X_valid,y_valid,X_test,y_test):

        # Specify Output filenames
        modelpath = self.config['mpath']+'models/'
        metricpath = self.config['mpath']+'metrics/'
        Path(modelpath).mkdir(parents=True, exist_ok=True)
        Path(metricpath).mkdir(parents=True, exist_ok=True)
        model_fname = f"{modelpath}{m}_{trainingid}.dat"
        metric_fname = f"{metricpath}{m}_{trainingid}.csv"

        logging.debug(f"Model {m}: Starting")

        if os.path.isfile(model_fname):
            if self.config['refit_models'] == False:
                return f"No refitting. Model {m} already exists. "
            logging.debug(f"Model {m}: Refit model")
        
        logging.debug(f"Model {m}: Fitting started")

        model = models[m]['model']

        if 'fit_params' in models[m].keys():
            fit_params = models[m]['fit_params']
            fit_params['eval_set'] = [(X_valid,y_valid)]
            model.fit(X_train,y_train,**fit_params)
        else:
            model.fit(X_train,y_train)
        
        with open(model_fname, "wb") as dill_file:
            dill.dump(model, dill_file)
        logging.debug(f"Model {m}: Fitting ended")
        
        if self.config['modeltype'] == 'regression':
            results = pd.DataFrame(columns=pd.MultiIndex.from_product([[trainingid],['train', 'valid','test'], ['mean_squared_error']]))

            results.loc[m,(trainingid,'train','mean_squared_error')]   = mean_squared_error(y_train,model.predict(X_train))
            results.loc[m,(trainingid,'valid','mean_squared_error')]   = mean_squared_error(y_valid,model.predict(X_valid))
            results.loc[m,(trainingid,'test','mean_squared_error')]   = mean_squared_error(y_test,model.predict(X_test))

            results.to_csv(metric_fname)
        else:
            results = pd.DataFrame(columns=pd.MultiIndex.from_product([[trainingid],['train', 'valid','test'], ['accuracy_score','roc_auc_score']]))

            results.loc[m,(trainingid,'train','accuracy_score')]   = accuracy_score(y_train,model.predict(X_train))
            results.loc[m,(trainingid,'train','roc_auc_score')]   = roc_auc_score(y_train,model.predict(X_train))

            results.loc[m,(trainingid,'valid','accuracy_score')]   = accuracy_score(y_valid,model.predict(X_valid))
            results.loc[m,(trainingid,'valid','roc_auc_score')]   = roc_auc_score(y_valid,model.predict(X_valid))

            results.loc[m,(trainingid,'test','accuracy_score')]   = accuracy_score(y_test,model.predict(X_test))
            results.loc[m,(trainingid,'test','roc_auc_score')]   = roc_auc_score(y_test,model.predict(X_test))

            results.to_csv(metric_fname)
        logging.debug(f"Model {m}: Metrics Calculated")

        return f"Completed Model {m}"        