from scipy.stats import randint, uniform
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import RANDOM_SEED





#podemos pasarle la funcion de activacion
#podemos pasarle un dropout
#podemos pasarle learning rate
#podemos pasarle el batch size
#y tambien threshold para la prediccion


#Se crea nuestra red poco profunda con una sola capa oculta
class MyNN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
        activation
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        return self.fc2(x)



#Creamos un clasificador sklearn que utiliza nuestra red neuronal poco profunda
#como modelo
class ShallowMultiLabelNet(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        hidden_dim=32,
        dropout=0.2,
        learning_rate=0.001,
        epochs=100,
        activation=nn.ReLU,
        batch_size=32,
        random_state=RANDOM_SEED,
        weight_decay=0.0001,
        threshold=0.5
    ):

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.batch_size = batch_size
        self.random_state = random_state
        self.threshold = threshold
        self.weight_decay = weight_decay
        self.rng_ = None

    def batch_loader(self, X, Y):

        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        self.rng_.shuffle(indices)

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], Y[batch_indices]





#Definimos un metodo auxiliar que entrena el modelo por una epoca
#Utilizando el metodo auxiliar batch_loader
    def train_one_epoch(self, loader, optimizer, criterion, device="cpu"):
      self.model_.train()

      for xb, yb in loader:
        xb, yb = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)
        optimizer.zero_grad()
        logits = self.model_(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()





    def fit(self, X, y):
        self.input_dim_ = X.shape[1]
        self.output_dim_ = y.shape[1]
        self.classes_ = np.arange(self.output_dim_)

        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.rng_ = np.random.default_rng(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.input_dim_ = X.shape[1]
        self.output_dim_ = y.shape[1]

        self.model_ = MyNN(
            input_dim=self.input_dim_,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim_,
            dropout=self.dropout,
            activation=self.activation
        )

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        


        for _ in range(self.epochs):
          #Debido a que es un generador, debemos recrearlo dentro del loop
          loader = self.batch_loader(X, y)

          self.train_one_epoch(
            loader= loader,
            optimizer= optimizer,
            criterion= criterion
           )

        return self


    def predict_proba(self, X):
        check_is_fitted(self, "model_")

        X = np.asarray(X, dtype=np.float32)

        X_tensor = torch.from_numpy(X)

        self.model_.eval()

        with torch.no_grad():

            logits = self.model_(X_tensor)

            probs = torch.sigmoid(logits)

        return probs.numpy()



    def predict(self, X):

        probs = self.predict_proba(X)

        return (probs >= self.threshold).astype(int)


nn_config = {
    'classifier': ShallowMultiLabelNet(random_state=RANDOM_SEED),
    'params': {
        'hidden_dim': randint(16, 64),
        'epochs': randint(50, 200),
        'activation': [
            torch.nn.ReLU,
            torch.nn.LeakyReLU
        ],
        'batch_size': randint(16, 32),
        'learning_rate': uniform(0.001, 0.01),
        'dropout': uniform(0.1, 0.5),
        'threshold': uniform(0.3, 0.7),
        'weight_decay': uniform(0, 0.0001)
    }
}