from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
def get_data(filename, label):
    file = pd.read_csv(filename)
    a=file.shape[0]
    row=1
    X=file.iloc[0, 1:].values
    y=label
    t = X[0]/2
    for i in enumerate(file.itertuples()):    
        u = file.iloc[row, 1:].values 
        row+=1
        if u[0]>t:
             X=np.vstack([X, u])
             y=np.append(y,label)
        if row==a:
            break
    return X, y
X_1, y_1=get_data('Dataset/HH1.csv', 1)
X_2, y_2=get_data('Dataset/HH2.csv', 2)
X_3, y_3=get_data('Dataset/HH3.csv', 3)
X_4, y_4=get_data('Dataset/HH4.csv', 4)
X_5, y_5=get_data('Dataset/nep10.csv', 5)
X_6, y_6=get_data('Dataset/nep20.csv', 6)
X_7, y_7=get_data('Dataset/nep30.csv', 7)
X_8, y_8=get_data('Dataset/nep40.csv', 8)
X_9, y_9=get_data('Dataset/nep50.csv', 9)
X_10, y_10=get_data('Dataset/nep60.csv', 10)
X_11, y_11=get_data('Dataset/Vodka15.csv', 11)
X_12, y_12=get_data('Dataset/Vodka20.csv', 12)
X_13, y_13=get_data('Dataset/Vodka30.csv', 13)
X_14, y_14=get_data('Dataset/Vodka40.csv', 14)
X_15, y_15=get_data('Dataset/Vodka50.csv', 15)
X_16, y_16=get_data('Dataset/Vodka60.csv', 16)
X=np.vstack([X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14, X_15, X_16])
X = preprocessing.scale(X)
y = np.append(y_1, y_2)
y = np.append(y, y_3)
y = np.append(y, y_4)
y = np.append(y, y_5)
y = np.append(y, y_6)
y = np.append(y, y_7)
y = np.append(y, y_8)
y = np.append(y, y_9)
y = np.append(y, y_10)
y = np.append(y, y_11)
y = np.append(y, y_12)
y = np.append(y, y_13)
y = np.append(y, y_14)
y = np.append(y, y_15)
y = np.append(y, y_16)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
import keras as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
s = tf.compat.v1.InteractiveSession()
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

np.random.seed(1)
model = K.models.Sequential()
model.add(K.layers.Dense(units=1000, input_dim=1892,
  activation='relu'))
model.add(K.layers.Dense(units=1000,
  activation='relu'))
model.add(K.layers.Dense(units=16, activation='softmax'))
model.compile(loss='categorical_crossentropy',
  optimizer='adam', metrics=['accuracy'])
print("Starting training ")
num_epochs = 30
h = model.fit(X_train, y_train,
  batch_size=30, epochs=num_epochs, verbose=0)
print("Training finished \n")
for i in range(num_epochs):
  if i % 1 == 0:
    los = h.history['loss'][i]
    acc = h.history['acc'][i] * 100
    print("epoch: %5d loss = %0.4f acc = %0.2f%%" \
      % (i, los, acc))
eval = model.evaluate(X_test, y_test, verbose=0)
print("\nEvaluation on test data: \nloss = %0.4f \
  accuracy = %0.2f%%" % (eval[0], eval[1]*100) ) 

