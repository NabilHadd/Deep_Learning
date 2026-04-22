from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd



#Creamos nuestras propias clases para usar dentro de pipeline.
from sklearn.base import BaseEstimator, TransformerMixin


#metodo para dropear columnas con mala correlación.
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        corr = X.corr().abs()

        triu_mask = np.triu(np.ones(corr.shape), k=1).astype(bool) #creamos una mascara trinagular superior
        upper = corr.where(triu_mask) #se aplica
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)] #guarda las columnas que no cumplen con el threshold
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.to_drop_, errors='ignore')



#Asumiremos que las variables estan relacionadas con la salida debido al contexto del estudio
#en cambio no asumiremos que estan relacionadas entre si (filtro de pearson)
preprocessors = {
    'pearson': {
        'pipeline': Pipeline([
            ('pearson_drop', CorrelationDropper()),
        ]),
        'params': {
            'pearson_drop__threshold': uniform(0.5, 0.45) #minimo threshold 0.6 maximo 0.95
        }
    },
    'pca':{
        'pipeline': Pipeline([
            ('standard_scaler', StandardScaler()),
            ('pca', PCA())
        ]),
        'params': {
            'pca__n_components': randint(7, 15) #minimo la mitad de las componentes, max el total
        }   
    }
}
