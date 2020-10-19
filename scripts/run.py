import yaml
import sys 
import os
import logging
import joblib
import pandas as pd

logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s",level=logging.INFO)

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'modules'))

# Import custom modules
from binning import iv_varsel, woe_bins
from fit import model_config, mltrain_loop

# Load Config
c = yaml.load(open('conf.yaml','r'),Loader=yaml.FullLoader)

data = pd.read_parquet(c['data'])

rerun_varsel = False
rerun_binning = False
refit_models = False

if os.path.isfile(c['dpath']+c['dvarsel']):
    logging.info('Variable Selection dataset already exists')
    if rerun_varsel:
        logging.info('Variable Selection redone')
        iv_varsel(
            data,
            c['bads'],
            c['target'],
            c['dpath'],
            c['dvarsel'],
        )
else:
    logging.info('Variable Selection dataset not found. Start Variable Selection')
    iv_varsel(
        data,
        c['bads'],
        c['target'],
        c['dpath'],
        c['dvarsel'],
    )


if os.path.isfile(c['dpath']+c['dbins']):
    logging.info('Woe Binning dataset already exists')
    if rerun_binning:
        logging.info('Woe Binning redone')
        data = pd.read_parquet(c['dpath']+c['dvarsel'])
        woe_bins(
            data,
            c['target'],
            c['bads'],
            c['dpath'],
            c['dbins'],
            c['ppath'],
        )
else:
    logging.info('Woe Binning dataset not found. Start Woe Binning')
    data = pd.read_parquet(c['dpath']+c['dvarsel'])
    woe_bins(
        data,
        c['target'],
        c['bads'],
        c['dpath'],
        c['dbins'],
        c['ppath'],
    )

# C
models = model_config(m_dict=c['models'])

data = pd.read_parquet(c['dpath']+c['dvarsel'])
bfile = c['dpath']+c['dbins']
bins = joblib.load(bfile)

print(len(models))

mltrain_loop(
    models,
    data,
    bins,
    c['target'],
    offset_lst=[0],
    begin_training  = pd.Timestamp(c['begin_training']),
    end_training    = pd.Timestamp(c['end_training']),
    begin_valid     = pd.Timestamp(c['begin_valid']),
    end_valid       = pd.Timestamp(c['end_valid']),
    begin_test      = pd.Timestamp(c['begin_test']),
    end_test        = pd.Timestamp(c['end_test']),
    refit_models=refit_models
    )