import pandas as pd
from munch import munchify

from .fit import Fitting
from .util import Util

class MLPipeline(Fitting, Util):

    def __init__(
        self,
        config = None
    ):
        self.config = munchify(config)

        self.target = self.config.variables.target
        self.bads = self.config.variables.bads
        self.target = self.config.variables.target
        self.data = self.get_data()
        self.feat = self.get_feat()

    def get_feat(self):      
        collst =list(self.data.columns)
        feat = sorted(list(set(collst)-set(self.bads)-set(self.target)))
        return feat

    def get_data(self):
        datafile = f"{self.config.data.inppath}/{self.config.data.inpfile}"
        return pd.read_parquet(datafile)