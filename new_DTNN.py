import numpy as np
import pandas as pd
import tensorflow as tf
import os.path
from os import path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
def get_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[1:101, 54:587].values
    y = np.full((1, 100), label)
    Z = data.iloc[200, 54:587].values
    X = Z/X
    return X, y
def get_data2(filename, label):
    data = pd.read_csv(filename)
    X1 = data.iloc[1:101, 54:320].values
    Z = data.iloc[200, 54:320].values
    X2 = data.iloc[1:101, 321:587].values
    Z = data.iloc[200, 321:587].values
    y = np.full((1, 200), label)
    X1 = Z/X1
    X2 = Z/X2
    X = np.vstack([X1, X2])
    return X, y
def get_mix_data2(filename, label):
    data = pd.read_csv(filename)
    X1 = data.iloc[1:5, 54:320].values
    Z = data.iloc[330, 54:320].values
    X2 = data.iloc[1:5, 321:587].values
    Z = data.iloc[330, 321:587].values
    y = np.full((1, 8), label)
    X1 = Z/X1
    X2 = Z/X2
    X = np.vstack([X1, X2])
    return X, y
def get_mix_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[1:5, 54:320].values
    y = np.full((1, 4), label)
    Z = data.iloc[330, 321:587].values
    X = Z/X
    return X, y
def main(argv = None):
    X_1, y_1 = get_data2('Dataset/nep10.csv', 16.67)
    X_2, y_2 = get_data2('Dataset/nep20.csv', 33.33)
    X_3, y_3 = get_data2('Dataset/nep30.csv', 50)
    X_4, y_4 = get_data2('Dataset/nep40.csv', 66.67)
    X_5, y_5 = get_data2('Dataset/nep50.csv', 83.33)
    X_6, y_6 = get_data2('Dataset/nep60.csv', 100)
    X = np.vstack([X_1, X_2, X_3, X_4, X_5, X_6])
    y = np.append(y_1, y_2)
    y = np.append(y, y_3)
    y = np.append(y, y_4)
    y = np.append(y, y_5)
    y = np.append(y, y_6)
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    scalar = MinMaxScaler()
    y = scalar.fit_transform(y.reshape(1200, 1))
    test, label = get_mix_data2('Dataset/HH2.csv', 30)
    test = np.asarray(test).astype(np.float32)
    X = np.reshape(X, [600, 2, 266])
    y = np.reshape(y, [600, 2])
    test = np.reshape(test, [4, 2, 266])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    num_epochs = 1000
    inputs = tf.keras.Input(shape = (2, 266))
    L1 = tf.keras.layers.Dense(512, activation = tf.nn.sigmoid)(inputs)
    split0, split1 = tf.split(L1, num_or_size_splits=2, axis=1)
    h1 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)(split0)
    h1 = tf.keras.layers.Dense(3, activation='sigmoid')(h1)
    h2 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)(split1)
    h2 = tf.keras.layers.Dense(3, activation='sigmoid')(h2)
    v = tf.linalg.cross(h1, h2)
    L2 = tf.keras.layers.Dense(128, activation = tf.nn.sigmoid)(v)
    L3 = tf.keras.layers.Dense(64, activation = tf.nn.sigmoid)(L2)
    outputs = tf.keras.layers.Dense(1)(L3)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    print(model.summary())
    model.compile(loss = 'mean_absolute_error', optimizer = 'Adam', metrics = ['mae', 'mse'])
    if path.exists("new_model.h5"):
        h = model.load_weights("new_model.h5")
    else:
        h = model.fit(X_train, y_train, validation_data =(X_val, y_val), batch_size=50, epochs=num_epochs, verbose=1)
        model.save("new_model.h5")
    eval = model.evaluate(X_test, y_test, verbose = 1)
    print("\nEvaluation on test data: \nloss = %0.4f " % (eval[0]))
    pre = model.predict(test)
    pre = np.reshape(pre, [1, 4])
    print(scalar.inverse_transform(pre))
    #print("\nTest loss: %0.4f" % mean_squared_error(pre, label))
    epochs_range = range(num_epochs)
    plt.plot(epochs_range, h.history['loss'], label='Test Loss')
    plt.plot(epochs_range, h.history['val_loss'], label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.show()
if __name__ == '__main__':
    tf.compat.v1.app.run()   
