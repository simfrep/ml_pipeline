import scorecardpy as sc
from sklearn.base import BaseEstimator, TransformerMixin

#Custom Transformer that extracts columns passed as argument to its constructor 
class WoeBinning( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, X, y ):
        df = X.copy()
        df['target'] = y
        self.bins = sc.woebin(df, y='target')
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X):
        woe = sc.woebin_ply(X, self.bins)
        return woe