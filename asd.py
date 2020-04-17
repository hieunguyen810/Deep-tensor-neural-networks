import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import os.path
from os import path
min_max_scaler = preprocessing.MinMaxScaler()
def get_data3(filename, label):
    file = pd.read_csv(filename)
    a=file.shape[0]
    row=1
    X=file.iloc[0, 1:].values
    y=label
    #t = X[0]/2
    for i in enumerate(file.itertuples()):    
        u = file.iloc[row, 1:].values 
        row+=1
        #if u[0]<t:
        X=np.vstack([X, u])
        y=np.append(y,label)
        if row==50:
            break
    X = X/X[0]
    X = min_max_scaler.fit_transform(X)
    return X, y
def get_data(filename, label):
    file = pd.read_csv(filename)
    a=file.shape[0]
    row=1
    X=file.iloc[0, 1:].values
    y=label
    #t = X[0]/2
    for i in enumerate(file.itertuples()):    
        u = file.iloc[row, 1:].values 
        row+=1
        #if u[0]<t:
        X=np.vstack([X, u])
        y=np.append(y,label)
        if row==300:
            break
    X = X/X[0]
    X = min_max_scaler.fit_transform(X)
    return X, y
def get_data2(filename, label):
    file = pd.read_csv(filename)
    a = file.shape[0]
    X1=file.iloc[0, 1:].values
    X2=file.iloc[1, 1:].values
    #t = X1[0]/2
    t = np.var(X1)
    y = label
    y = np.append(y, label)
    row=2
    for i in enumerate(file.itertuples()):
        u = file.iloc[row, 1:].values
        row+=1
        if np.var(u)>t:
            X1 = np.vstack([X1, u])
            y = np.append(y, label)
        else:
            X2 = np.vstack([X2, u])
            y = np.append(y, label)
        if row==300:
            break
    X1 = X1/X1[0]
    X1 = min_max_scaler.fit_transform(X1)
    X2 = X2/X2[0]
    X2 = min_max_scaler.fit_transform(X2)
    X = np.vstack([X1, X2])
    return X, y
X_1, y_1=get_data('Dataset/nep10.csv', 16.67)
X_2, y_2=get_data('Dataset/nep20.csv', 33.33)
X_3, y_3=get_data('Dataset/nep30.csv', 50)
X_4, y_4=get_data('Dataset/nep40.csv', 66.67)
X_5, y_5=get_data('Dataset/nep50.csv', 83.33)
X_6, y_6=get_data('Dataset/nep60.csv', 100)
X = np.vstack([X_1, X_2, X_3, X_4, X_5, X_6]) 
y = np.append(y_1, y_2)
y = np.append(y, y_3)
y = np.append(y, y_4)
y = np.append(y, y_5)
y = np.append(y, y_6)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(1800,1))
#X = scalarX.transform(X)
y = scalarY.transform(y.reshape(1800,1))
X = np.reshape(X, [900, 2, 1892])
y = np.reshape(y, [900, 2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

test, label = get_data3('Dataset/HH2.csv', 17 )
scalarX.fit(test)
test = scalarX.transform(test)
scalarY.fit(label.reshape(50,1))
label = scalarY.transform(label.reshape(50,1))
test = np.reshape(test, [25, 2, 1892])
label = np.reshape(label, [25, 2])
print(label)
num_epochs = 10
inputs = tf.keras.Input(shape=(2, 1892))
#x0 = tf.keras.layers.Dropout(rate=0.5, noise_shape=None, seed=None, name=None)(inputs)
x1 = tf.keras.layers.Dense(400, activation=tf.nn.sigmoid)(inputs)
split0, split1 = tf.split(x1, num_or_size_splits=2, axis=1)
h1 = tf.keras.layers.Dense(25, activation=tf.nn.sigmoid)(split0)
h2 = tf.keras.layers.Dense(25, activation=tf.nn.sigmoid)(split1)
h1 = tf.transpose(h1)
v = tf.tensordot(h1, h2, axes=[[1],[1]])
x2 = tf.keras.layers.Dense(400, activation=tf.nn.sigmoid)(v)
#x3 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)(x2)
x4 = tf.keras.layers.Dense(400, activation=tf.nn.sigmoid)(x2)
outputs = tf.keras.layers.Dense(1)(x4)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#if path.exists("model.h5"):
#    model.load_weights("model.h5")
#else:
model.fit(X_train, y_train, batch_size=30, epochs=num_epochs, verbose=1)
#    model.save("model.h5")
    
eval = model.evaluate(X_test, y_test, verbose=1)
print("\nEvaluation on test data: \nloss = %0.4f \
  accuracy = %0.2f%%" % (eval[0], eval[1]*100) )
pre = model.predict(test)
#print(pre.shape)
#print(label)
pre = np.reshape(pre, [15625, 1])
#label = np.reshape(label, [, 1])
print(scalarY.inverse_transform(pre))
#print(scalarY.inverse_transform(label))
#pred = model.predict(X_test)
#print(scalarY.inverse_transform(pred))
#print(scalarY.inverse_transform(y_test))
#print(np.sqrt(mean_squared_error(y_test,pred)))




