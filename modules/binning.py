import scorecardpy as sc
import pandas as pd
import joblib
from pathlib import Path
import os
import logging

#logging.basicConfig(level=logging.DEBUG)

def iv_varsel(data,bads,tgt,dpath,dvarsel):

    collst =list(data.columns)
    #logging.debug(collst)
    feat = sorted(list(set(collst)-set(bads)-set(tgt)))
    #logging.debug(feat)
    data = data[feat+[tgt]]
    logging.debug(data[tgt].value_counts())
    data = sc.var_filter(data, y= tgt)
    
    # Ensure Output path exists. If not create folder
    Path(dpath).mkdir(parents=True, exist_ok=True)

    outfile = dpath+dvarsel
    if os.path.isfile(dpath+dvarsel):
        logging.info('File already exists. Add timestamp suffix')
        _suffix = pd.Timestamp('now').strftime('%Y%m%d_%H:%m:%S')
        outfile = dpath+dvarsel+_suffix
    data.to_parquet(outfile)

def woe_bins(data, tgt, bads, dpath, dbins, ppath):
    
    collst =list(data.columns)
    feat = sorted(list(set(collst)-set(bads)))

    data = data[feat+[tgt]]
    bins = sc.woebin(data, y=tgt)
    joblib.dump(bins, dpath+dbins)

    # Create and save plots
    plotlist = sc.woebin_plot(bins)

    # Create df to rank variables by IV
    iv_rank = pd.DataFrame()
    for key in bins.keys():
        iv_rank.loc[key,'iv']=bins[key]['bin_iv'].sum()
    iv_rank['iv_rank'] = iv_rank['iv'].rank(ascending=False)

    # Ensure Output path exists. If not create folder
    Path(ppath).mkdir(parents=True, exist_ok=True)
    # Save Plots with rank in filename
    for i,x in iv_rank.iterrows():
        rk = int(x['iv_rank'])
        #print(f'{rk:03}')
        plotlist[i].savefig(ppath+f'{rk:03}'+'_'+(i)+'.png')

