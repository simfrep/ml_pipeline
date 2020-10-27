from .binning import Binning
from .fit import Fitting

class MLPipeline(Binning, Fitting):

    def __init__(
        self,
        config = None
    ):
        self.config = config
