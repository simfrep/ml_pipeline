import pandas as pd
import os
import dill
import logging
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random

from .util import func_from_string


class Fitting:
    def __init__(self, config=None):
        self.config = config

    def model_config(self):
        m_dict = self.config["models"]
        if "preprocessing" in self.config.keys():
            prep = self.config["preprocessing"]

        models = {}

        for trans in prep.keys():
            trans_args = tuple()
            trans_kwargs = {}
            tf_has_args = False
            tf_has_kwargs = False

            logging.info(f"{trans}")
            transFunc = func_from_string(trans)

            if prep[trans] is not None:
                tf_has_args = "args" in prep[trans].keys()
                tf_has_kwargs = "kwargs" in prep[trans].keys()

            if tf_has_args:
                trans_args = tuple([getattr(self, x) for x in prep[trans].args])
                logging.info(f"trans_args: {trans_args}")

            if tf_has_kwargs:
                trans_kwargs = prep[trans].kwargs
                logging.info(f"trans_kwargs: {trans_kwargs}")

            for model in m_dict.keys():
                para_lst = list(ParameterGrid(m_dict[model]))
                modelFunc = func_from_string(model)

                mname = f"{transFunc.__name__}_{modelFunc.__name__}"
                _models = {
                    f"{mname}_{'_'.join([f'{x}{y}' for (x,y) in list(p.items())])}": {
                        "model": Pipeline([
                            ("transformation",transFunc(*trans_args, **trans_kwargs),),
                            ("model", modelFunc(**p)),
                        ])
                    }
                    for p in para_lst
                }
                models.update(_models)

        return models

    def split_datasets_byts(
        self,
        begin_training=pd.Timestamp("2015-01-01 00:00"),
        begin_valid=pd.Timestamp("2020-01-01 00:00"),
        begin_test=pd.Timestamp("2020-04-01 00:00"),
    ):
        """
        Creates and returns all necessary Datasets for normal modellung and DMatrix elements for XGB
        """

        collst = self.feat + [self.target]
        d = {}
        d["dataset"] = self.data.loc[begin_training:begin_valid, collst].dropna()
        d["valiset"] = self.data.loc[begin_valid:begin_test, collst].dropna()
        d["testset"] = self.data.loc[begin_test:, collst].dropna()
        d["X_train"] = d["dataset"][self.feat]
        d["X_valid"] = d["valiset"][self.feat]
        d["X_test"] = d["testset"][self.feat]
        d["y_train"] = d["dataset"][self.target].values
        d["y_valid"] = d["valiset"][self.target].values
        d["y_test"] = d["testset"][self.target].values

        logging.debug(
            f'Training Dataset {d["dataset"].index.min()} {d["dataset"].index.max()}'
        )
        logging.debug(
            f'Validation Dataset {d["valiset"].index.min()} {d["valiset"].index.max()}'
        )
        logging.debug(
            f'Test Dataset {d["testset"].index.min()} {d["testset"].index.max()}'
        )

        for key, value in d.items():
            setattr(self, key, value)

    def split_datasets_byratio(self, split: list = [0.75, 0.15, 0.1]):
        d = {}
        logging.info(f"Using Split: {split}")
        train_ratio = split[0]
        validation_ratio = split[1]
        test_ratio = split[2]
        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        d["X_train"], _X, d["y_train"], _y = train_test_split(
            self.data[self.feat], self.data[self.target], test_size=1 - train_ratio
        )
        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        d["X_valid"], d["X_test"], d["y_valid"], d["y_test"] = train_test_split(
            _X, _y, test_size=test_ratio / (test_ratio + validation_ratio)
        )
        d["dataset"] = pd.concat([d["X_train"], d["y_train"]], axis=1)
        d["valiset"] = pd.concat([d["X_valid"], d["y_valid"]], axis=1)
        d["testset"] = pd.concat([d["X_test"], d["y_test"]], axis=1)
        logging.debug(f'Training Dataset {d["dataset"].shape}')
        logging.debug(f'Validation Dataset {d["valiset"].shape}')
        logging.debug(f'Test Dataset {d["testset"] .shape}')

        for key, value in d.items():
            setattr(self, key, value)

    def mltrain_loop(self, **kwargs):

        logging.debug(self.config)
        logging.debug(self.config.keys())

        if "split" in dir(self.config):
            split = self.config.split
            if type(split) == list:
                self.split_datasets_byratio(split)

                models = self.model_config()
                mlist = list(models.keys())
                random.shuffle(mlist)

                # Create pools for parallel execution
                pool = ProcessPoolExecutor(self.config.runtime.parallel_threads)

                # This list contains all functions that are executed
                futures = []
                cnt_begins = 1
                cnt_totals = 0
                cnt_models = 0
                trainingiterations = 1
                total_models = len(mlist) * trainingiterations
                for x in mlist:
                    futures.append(
                        pool.submit(self.mfit, x, models, "RatioSplit")
                    )

                # This loop ensures that we wait until all models are fitted
                for x in as_completed(futures):
                    cnt_models += 1
                    cnt_totals += 1
                    logging.debug(x.result())
                    logging.debug(
                        f"{cnt_begins}/{trainingiterations} {cnt_models}/{len(mlist)} Completed Models"
                    )
                    if (cnt_models % 10 == 0) or (cnt_models == len(mlist)):
                        logging.info(
                            f"Finished {cnt_totals} of {total_models} model fittings (Training Begin {cnt_begins}/{trainingiterations} Model {cnt_models}/{len(mlist)})"
                        )

            elif "time" in dir(split):
                time = split.time
                begin = split.time.begin_training
                logging.debug(f"Config begin_training: {begin} of type {type(begin)}")
                if type(begin) == list:
                    training_begins = [pd.Timestamp(x) for x in begin]
                else:
                    offset_lst = time.offset_lst
                    offset_res = time.offset_res
                    training_begins = [
                        pd.Timestamp(begin) + pd.DateOffset(**{offset_res: offset})
                        for offset in offset_lst
                    ]
                logging.debug(f"List of Training Begins: {training_begins}")

                begin_valid = pd.Timestamp(time.begin_valid)
                begin_test = pd.Timestamp(time.begin_test)

                trainingiterations = len(training_begins)

                cnt_begins = 1
                cnt_totals = 0
                for begin_training in training_begins:
                    self.split_datasets_byts(
                        begin_training=begin_training,
                        begin_valid=begin_valid,
                        begin_test=begin_test,
                    )
                    models = self.model_config()
                    mlist = list(models.keys())
                    random.shuffle(mlist)

                    total_models = len(mlist) * trainingiterations
                    logging.info(
                        f"Fitting {len(mlist)} models for {trainingiterations} training datasets. Total: {total_models}"
                    )
                    logging.debug(f"List of models {mlist}")

                    trainingid = begin_training.strftime("%Y%m%d")
                    # Create pools for parallel execution
                    pool = ProcessPoolExecutor(self.config.runtime.parallel_threads)

                    # This list contains all functions that are executed
                    futures = []
                    cnt_models = 0

                    for x in mlist:
                        futures.append(
                            pool.submit(self.mfit, x, models, trainingid)
                        )

                    # This loop ensures that we wait until all models are fitted
                    for x in as_completed(futures):
                        cnt_models += 1
                        cnt_totals += 1
                        logging.debug(x.result())
                        logging.debug(
                            f"{cnt_begins}/{len(training_begins)} {cnt_models}/{len(mlist)} Completed Models"
                        )
                        if (cnt_models % 10 == 0) or (cnt_models == len(mlist)):
                            logging.info(
                                f"Finished {cnt_totals} of {total_models} model fittings (Training Begin {cnt_begins}/{trainingiterations} Model {cnt_models}/{len(mlist)})"
                            )

                    cnt_begins += 1
            else:
                raise ValueError(
                    "self.config.split must be either list or split by times"
                )
        else:
            logging.info("No splits defined use default split")
        logging.info(f"Finished Model Fitting.")

    def mfit(self, m, models, trainingid):

        # Specify Output filenames
        outpath = self.config.data.outpath
        modelpath = f"{outpath}/models"
        metricpath = f"{outpath}/metrics"
        Path(modelpath).mkdir(parents=True, exist_ok=True)
        Path(metricpath).mkdir(parents=True, exist_ok=True)
        model_fname = f"{modelpath}/{m}_{trainingid}.dat"
        metric_fname = f"{metricpath}/{m}_{trainingid}.csv"

        logging.debug(f"Model {m}: Starting")

        if os.path.isfile(model_fname):
            if self.config.runtime.refit == False:
                return f"No refitting. Model {m} already exists. "
            logging.debug(f"Model {m}: Refit model")

        logging.debug(f"Model {m}: Fitting started")

        model = models[m]["model"]

        if "fit_params" in models[m].keys():
            fit_params = models[m]["fit_params"]
            fit_params["eval_set"] = [(self.X_valid,self.y_valid)]
            model.fit(self.X_train, self.y_train, **fit_params)
        else:
            model.fit(self.X_train, self.y_train)

        with open(model_fname, "wb") as dill_file:
            dill.dump(model, dill_file)
        logging.debug(f"Model {m}: Fitting ended")

        results = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [[m], [trainingid], list(self.config.metrics.keys())]
            ),
            columns=["train", "valid", "test"],
        )
        for metric in self.config.metrics.keys():
            metricFunc = func_from_string(metric)

            for ds in ["train", "valid", "test"]:
                idx = (m, trainingid, metric)
                y = getattr(self,f"y_{ds}")
                y_pred = model.predict(getattr(self,f"X_{ds}"))
                results.loc[idx, ds] = metricFunc(y, y_pred)

            results.to_csv(metric_fname)

        logging.debug(f"Model {m}: Metrics Calculated")

        return f"Completed Model {m}"
