import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.layers import Input, Dense, Dropout
from keras.models import Model





class MCDropoutREGWrapper():
    def __init__(self, model, epochs = 50, mcd_runs = 50, debug = False):

        # self.model = model_form
        # self.model.compile(optimizer='adam', loss='mae')

        self.model = model[0]
        self.init_weights = model[1]

        self.model.compile(optimizer='adam', loss='mae')
        self.model.set_weights(self.init_weights)
        #print(self.model.get_weights())

        self.scaler = StandardScaler()
        self.epochs = epochs
        self.mcd_runs = mcd_runs

        if debug:
            self.epochs = 3
            self.mcd_runs = 2
        
    def fit(self, X_train, y_train):

        X_train = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train, y_train, epochs=self.epochs, verbose=0) #epochs = 50
        
    def refit(self, X_refit, y_refit):
        X_refit = self.scaler.transform(X_refit)

        self.model.compile(optimizer='adam', loss='mae')
        self.model.set_weights(self.init_weights)
        self.model.fit(X_refit, y_refit, epochs=self.epochs, verbose=0)
    
    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        
        y_pred = []
        predictions = []

        #print('MCD: {} runs'.format(self.mcd_runs))
        for _ in range(self.mcd_runs):
            pred = self.model.predict(X_test)
            predictions.append(pred)

        mean = np.mean(predictions, axis=0)
        var = np.var(predictions, axis=0)
        y_pred.append(np.hstack(mean))
        uncertainty_list = np.hstack(var)

        return y_pred, uncertainty_list
