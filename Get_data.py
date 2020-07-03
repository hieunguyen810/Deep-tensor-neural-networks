import numpy as np
import pandas as pd
from sklearn import preprocessing
import feature_selection
def get_data(filename, label):
    data = pd.read_csv(filename)
    a = np.arange(54, 321)
    b = np.arange(321, 588)
    X1 = data.iloc[5:85, a].values
    Z = data.iloc[200, a].values
    X2 = data.iloc[5:85, b].values
    K = data.iloc[200, b].values
    y = np.full((1, 80), label)
    X_1 = Z/X1
    X_2 = K/X2
    X = np.vstack([X_1, X_2])
    X = np.reshape(X, [80, 2, 267])
    return X, y
def get_mix_data(filename):
    a = np.arange(54, 321)
    b = np.arange(321, 588)
    data = pd.read_csv(filename)
    X1 = data.iloc[11:21, a].values
    Z = data.iloc[300, a].values
    X2 = data.iloc[11:21, b].values
    K = data.iloc[300, b].values
    X_1 = Z/X1
    X_2 = K/X2
    X = np.vstack([X_1, X_2])
    X = np.reshape(X, [10, 2, 267])
    return X
def get_full_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[1:81, 54:588].values
    y = np.full((1, 80), label)
    Z = data.iloc[200, 54:588].values
    X = Z/X
    return X, y
def get_full_mix_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[1:81, 54:588].values
    y = np.full((1, 80), label)
    Z = data.iloc[320, 54:588].values
    X = Z/X
    return X, y
def get_data_Vodka(filename, label):
    data = pd.read_csv(filename)
    a, b = feature_selection.feature_importance()
    X_1 = data.iloc[1:81,a].values
    Z = data.iloc[260,a].values
    X_2 = data.iloc[1:81, b].values
    K = data.iloc[260, b].values
    y = np.full((1, 80), label)
    X_1 = Z/X_1
    X_2 = K/X_2
    X = np.vstack([X_1, X_2])
    X = np.reshape(X, [80, 2, 267])
    return X, y

