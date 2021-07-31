import scorecardpy as sc
import pandas as pd
import dill
from pathlib import Path
import os
import logging

#logging.basicConfig(level=logging.DEBUG)

class Binning():
    
    def __init__(
        self,
        config
    ):
        self.config = config

    def iv_varsel(self, **kwargs):

        if self.config is None:
            logging.info("No Config file provided")
        else:
            logging.info("Config file provided")
            datafile = self.config['dpath']+self.config['data']
            data = pd.read_parquet(datafile)
            bads = self.config['bads']
            tgt = self.config['target']
            dpath = self.config['dpath']
            dvarsel = self.config['dvarsel']

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

    def woe_bins(self, **kwargs):

        if self.config is None:
            logging.info("No Config file provided")
        else:
            logging.info("Config file provided")

            datafile = self.config['dpath']+self.config['dvarsel']
            if os.path.isfile(datafile):
                logging.info(f"Preprocessed Datafile found {datafile}")
                
            else:
                datafile = self.config['dpath']+self.config['data']
                logging.info(f"No Preprocessed Datafile found. Default to {datafile}")

            data = pd.read_parquet(datafile)
            bads = self.config['bads']
            tgt = self.config['target']
            dpath = self.config['dpath']
            dbins = self.config['dbins']
            ppath = self.config['mpath']+self.config['ppath']
        
        collst =list(data.columns)
        feat = sorted(list(set(collst)-set(bads)))

        data = data[feat+[tgt]]
        bins = sc.woebin(data, y=tgt)
        with open(dpath+dbins, "wb") as dill_file:
            dill.dump(bins, dill_file)

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

