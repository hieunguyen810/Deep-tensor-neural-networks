import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import Get_data
import Show_result  
import Train_tf2
def main(argv = None):   
    X_1, y_1=Get_data.get_data('Dataset/nep10.csv', 16.67)
    X_2, y_2=Get_data.get_data('Dataset/nep20.csv', 33.33)
    X_3, y_3=Get_data.get_data('Dataset/nep30.csv', 50)
    X_4, y_4=Get_data.get_data('Dataset/nep40.csv', 66.67)
    X_5, y_5=Get_data.get_data('Dataset/nep50.csv', 83.33)
    X_6, y_6=Get_data.get_data('Dataset/nep60.csv', 100)
    X = np.vstack([X_1, X_2, X_3, X_4, X_5, X_6])
    y = np.append(y_1, y_2)
    y = np.append(y, y_3)
    y = np.append(y, y_4)
    y = np.append(y, y_5)
    y = np.append(y, y_6)
    X = np.asarray(X).astype(float)
    y = np.asarray(y).astype(float)
    test, label = Get_data.get_data('Dataset/nep10.csv', 1)
    test = np.asarray(test).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    loss, val_loss = Train_tf2.train(X_train, X_test, y_train, y_test, X_val, y_val, test)
    num_epochs = 6000
    Show_result.show_result_tf2(loss, val_loss, num_epochs)
if __name__ == '__main__':
    tf.compat.v1.app.run()











