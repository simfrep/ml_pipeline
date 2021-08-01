from .fit import Fitting
from munch import munchify

class MLPipeline(Fitting):

    def __init__(
        self,
        config = None
    ):
        self.config = munchify(config)
