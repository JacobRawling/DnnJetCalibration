"""
A set of models aimed at fully calibrating the four vectors of jets. 
"""
import keras 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import ELU
from keras.optimizers import RMSprop
import time
import numpy as np

def feedforward_model():
    """
    Creates a simple feed power NN that takes a four vector as an input and 
    returns a calibrated 4 vector 
    """
    model = Sequential()
    model.add(Dense(30, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4))
    model.add(ELU(4))
    return model 
