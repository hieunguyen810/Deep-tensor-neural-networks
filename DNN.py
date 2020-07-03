import numpy as np
import pandas as pd
import tensorflow as tf
import os.path
from os import path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import feature_selection
def get_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[5:85,54:588].values
    y = np.full((1, 80), label)
    Z = data.iloc[200,54:588].values
    X = Z/X
    return X, y
def get_mix_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[1:11,54:588].values
    Z1 = data.iloc[330,54:588].values
    Z2 = data.iloc[300,54:588].values
    X = (Z1+Z2)/(2*X)
    return X
def main(argv = None):
    '''
    data = pd.read_csv('Dataset/Vodka15.csv')
    _, a = feature_selection.feature_importance()
    X_1 = data.iloc[1:81, a].values
    y_1 = np.full((1, 80), 25)
    Z_1 = data.iloc[200, a].values
    X_1 = Z_1/X_1
    '''
    X_1, y_1 = get_data('Dataset/nep10.csv', 16.67)
    X_2, y_2 = get_data('Dataset/nep20.csv', 33.33)
    X_3, y_3 = get_data('Dataset/nep30.csv', 50)
    X_4, y_4 = get_data('Dataset/nep40.csv', 66.67)
    X_5, y_5 = get_data('Dataset/nep50.csv', 83.33)
    X_6, y_6 = get_data('Dataset/nep60.csv', 100)
    X = np.vstack([X_1, X_2, X_3, X_4, X_5, X_6])
    y = np.append(y_1, y_2)
    y = np.append(y, y_3)
    y = np.append(y, y_4)
    y = np.append(y, y_5)
    y = np.append(y, y_6)
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    test = get_mix_data('Dataset/HH1.csv')
    test = np.asarray(test).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    num_epochs = 8000
    inputs = tf.keras.Input(shape = (534, ))
    L1 = tf.keras.layers.Dense(47, activation = tf.nn.sigmoid)(inputs)
    L2 = tf.keras.layers.Dense(10, activation = tf.nn.relu)(L1)
    outputs = tf.keras.layers.Dense(1)(L2)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    print(model.summary())
    optimizers = tf.keras.optimizers.Adam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(loss = 'mean_absolute_error', optimizer = optimizers, metrics = ['mae', 'mse'])
    if path.exists("m1.h5"):
        h = model.load_weights("m1.h5")
    else:
        h = model.fit(X_train, y_train, validation_data =(X_val, y_val), batch_size=30, epochs=num_epochs, verbose=1)
        model.save("m1.h5")
    eval = model.evaluate(X_test, y_test, verbose = 1)
    print("\nEvaluation on test data: \nloss = %0.4f " % (eval[0]))
    pred = model.predict(test)
    print(pred)
    epochs_range = range(num_epochs)
    plt.plot(epochs_range, h.history['loss'], label='Test Loss')
    plt.plot(epochs_range, h.history['val_loss'], label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.show()
if __name__ == '__main__':
    tf.compat.v1.app.run()   
