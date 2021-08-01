from .fit import Fitting

class MLPipeline(Fitting):

    def __init__(
        self,
        config = None
    ):
        self.config = config
