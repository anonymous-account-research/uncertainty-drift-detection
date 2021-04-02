import pandas as pd
import numpy as np
from scipy.stats import entropy

from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical


from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
from sklearn.preprocessing import StandardScaler


class MCDropoutCLFWrapper():
    def __init__(self, model, targets = 2, epochs = 50, mcd_runs = 25, scale_data = False, debug=False):


        self.model = model[0]
        self.init_weights = model[1]

        self.targets = targets
        if self.targets > 2:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            print('{} classes'.format(targets))
        else:
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
            print('2 classes')
        self.model.set_weights(self.init_weights)

        self.scaler = StandardScaler()
        self.scale_data = scale_data
        self.epochs = epochs
        self.mcd_runs = mcd_runs

        if debug:
            self.epochs = 3
            self.mcd_runs = 2
            print('Epochs: ', 3)

        
    def fit(self, X_train, y_train):
        if self.scale_data == True:
            X_train = self.scaler.fit_transform(X_train)
            #print(X_train)

        y_train_cat = to_categorical(y_train, num_classes=self.targets)
        self.model.fit(x=X_train, y=y_train_cat, epochs=self.epochs, verbose=0)
        
    def refit(self, X_retrain, y_retrain):
        if self.scale_data == True:
            X_retrain = self.scaler.fit_transform(X_retrain)

        y_retrain_cat = to_categorical(y_retrain, num_classes=self.targets)

        if self.targets > 2:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        else:
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.set_weights(self.init_weights)

        self.model.fit(x=X_retrain, y=y_retrain_cat, epochs=self.epochs, verbose=0)
        
        
    # def predict(self, X_test):
    #
    #     y_pred = []
    #     uncertainty_list = []
    #
    #     for i in range(len(X_test)):
    #         predictions = []
    #         for runs in range(25):
    #             prediction = self.model.predict(X_test[i].reshape(1,-1))
    #             predictions.append(prediction)
    #
    #         predictions_class_probability = np.mean(predictions, axis=0)
    #         entro = entropy(predictions_class_probability.tolist()[0], base=2)
    #         #entro = 0.25 - np.array(predictions_class_probability).var()
    #         y_pred.append(np.argmax(predictions_class_probability))
    #         uncertainty_list.append(entro)
    #
    #
    #     return y_pred, uncertainty_list
    
    def predict(self, X_test):

        if self.scale_data == True:
            X_test = self.scaler.fit_transform(X_test)

        y_pred = []
        predictions = []

        for runs in range(self.mcd_runs): #25
            prediction = self.model.predict(X_test)
            predictions.append(prediction)

        predictions_class_probability = np.mean(predictions, axis=0)
        entro = entropy(predictions_class_probability, base=2, axis = 1)
        # entro = 0.25 - np.array(predictions_class_probability).var()
        y_pred.append(np.argmax(predictions_class_probability, axis= 1))
        uncertainty_list = entro.tolist()

        return predictions_class_probability, uncertainty_list,

    
