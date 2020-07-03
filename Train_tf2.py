import tensorflow as tf
import os.path
from os import path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
def train(X_train, X_test, y_train, y_test, X_val, y_val, test):
    num_epochs = 6000
    inputs = tf.keras.Input(shape = (2, 267))
    L1 = tf.keras.layers.Dense(32, activation = 'sigmoid')(inputs)
    split0, split1 = tf.split(L1, num_or_size_splits=2, axis = 1)
    h1 = tf.keras.layers.Dense(3, activation='sigmoid')(split0)
    h2 = tf.keras.layers.Dense(3, activation='sigmoid')(split1)
    v = tf.linalg.cross(h1, h2)
    L2 = tf.keras.layers.Dense(13, activation = 'sigmoid')(v)
    L3 = tf.keras.layers.Dense(10, activation = 'relu')(L2)
    outputs = tf.keras.layers.Dense(1)(L3)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    print(model.summary())
    callback = tf.keras.callbacks.TensorBoard(log_dir=".\logs")
    model.compile(loss = "mean_absolute_error", optimizer = "Adam", metrics = ['mae', 'mse'])
    if path.exists("vodka.h5"):
        h = model.load_weights("vodka.h5")
    else:
        h = model.fit(X_train, y_train, validation_data =(X_val, y_val), batch_size=32, epochs=num_epochs, verbose=1, callbacks = [callback])
        model.save("vodka.h5")
    eval = model.evaluate(X_test, y_test, verbose = 1)
    print("\nEvaluation on test data: \nloss = %0.4f " % (eval[0]))
    pre = model.predict(test)
    print(pre)
    return h.history['loss'], h.history['val_loss']
