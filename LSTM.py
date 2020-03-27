import click
import pandas as pd 
import numpy as np 
import keras
import datetime as dt
from matplotlib import pyplot as plt

def features_merging(features):
    tab_merged = []
    for i in range(len(features[0])):
        tab_merged.append([features[j][i] for j in range(len(features))])
    return tab_merged

class Model():
    def __init__(self, preprocessor, features, goal):
        self.preprocessor = preprocessor
        self.df_train = preprocessor.df_train
        self.df_test = preprocessor.df_test
        self.features = features
        self.goal = goal

        self.model = keras.models.Sequential()
        self.input_layer = None
        self.layers = []
        self.y_pred = None
        self.scaler = None
        
        self.X_train = np.asarray(self.df_train[self.features].values.tolist())
        self.y_train = np.asarray(self.df_train[self.goal].values.tolist())
        self.X_test = np.asarray(self.df_test[self.features].values.tolist())
        self.y_test = np.asarray(self.df_test[self.goal].values.tolist())
        
    def features_selection(self):
        
        self.df_train['Features'] = self.df_train[self.features].apply(lambda x: 
            features_merging(x), axis=1)
        self.df_test['Features'] = self.df_test[self.features].apply(lambda x: 
            features_merging(x), axis=1)
            
        self.X_train = np.asarray(self.df_train['Features'].values.tolist())
        self.y_train = np.asarray(self.df_train[self.goal].values.tolist())
        self.X_test = np.asarray(self.df_test['Features'].values.tolist())
        self.y_test = np.asarray(self.df_test[self.goal].values.tolist())

    def LSTM(self, N_n):
        
        return keras.layers.LSTM(N_n)

    def dense(self, N_n):
        
        return keras.layers.Dense(N_n)

    def add_layer(self, layer):
        
        self.layers.append(layer)

    def remove_layer(self, n_layers=1):
        
        if len(self.layers) > 0:
            if n_layers > len(self.layers):
                return 'Error: you are trying to remove layers than there are !'
            else:
                self.layers = self.layers[:-n_layers]

        else:
            return 'Error: no layer to remove'

    def training(self, n_epochs=100, loss='mse', 
    optimizer=keras.optimizers.RMSprop(0.001), metrics=['mae', 'mse']):
        
        if len(self.layers) == 0:
            return 'Error: you are trying to train an empty network'
        else:
            for layer in self.layers:
                self.model.add(layer)

            self.model.compile(loss=loss, optimizer=optimizer, 
            metrics=metrics)
            
            self.model.fit(self.X_train, self.y_train, epochs=n_epochs)
            
    def predict(self, X):
        
        return self.model.predict(X)

    def validation(self):
        
        self.y_pred = self.predict(self.X_test).flatten()
        
        fig,ax = plt.subplots(figsize=(25, 10))
        
        ax.scatter(x=self.y_test, y=self.y_pred)
        ax.set_xtitle('Real labels')
        ax.set_ytitle('Predictions')
        
    