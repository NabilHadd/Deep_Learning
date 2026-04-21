from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd



#Creamos nuestras propias clases para usar dentro de pipeline.
from sklearn.base import BaseEstimator, TransformerMixin


#metodo para dropear columnas arbitrareamente
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)


#metodo para dropear columnas con mala correlación.
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        corr = X.corr().abs()

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.to_drop_, errors='ignore')



#Asumiremos que las variables estan relacionadas con la salida debido al contexto del estudio
#en cambio no asumiremos que estan relacionadas entre si (filtro de pearson)
preprocessors = {
    'pearson_chi2': {
        'pipeline': Pipeline([
            ('dropper', ColumnDropper()),
            ('pearson_drop', CorrelationDropper(threshold=0.8)),
        ]),
        'params': {
            'dropper__columns': [['GDS_R1','GDS_R2','GDS_R3','GDS_R4','GDS_R5'], ['GDS','GDS_R1','GDS_R3','GDS_R4','GDS_R5']]
        }
    },
    'pca':{
        'pipeline': Pipeline([
            ('dropper', ColumnDropper()),
            ('standar_scaler', StandardScaler()),
            ('pca', PCA())
        ]),
        'params': {
            'dropper__columns': [['GDS_R1','GDS_R2','GDS_R3','GDS_R4','GDS_R5'], ['GDS','GDS_R1','GDS_R3','GDS_R4','GDS_R5']],
            'pca__n_components': [10, 12]
        }   
    }
}
