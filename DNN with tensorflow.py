import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
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

## DNN
def buildDNN(inputs):
    
    with tf.variable_scope('hidden_layer1'):
        weights = tf.get_variable('weights',[INPUT_SIZE, HIDDEN_LAYER1_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        print(weights)
        biases = tf.get_variable('biases', [HIDDEN_LAYER1_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        layer1 = tf.nn.elu(tf.matmul(inputs, weights) + biases)
        
    with tf.variable_scope('hidden_layer2'):
        weights = tf.get_variable('weights',[HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        print(weights)
        biases = tf.get_variable('biases', [HIDDEN_LAYER2_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        layer2 = tf.nn.elu(tf.matmul(layer1, weights) + biases)
        
    with tf.variable_scope('hidden_layer3'):
        weights = tf.get_variable('weights',[HIDDEN_LAYER2_SIZE, HIDDEN_LAYER3_SIZE],
                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        print(weights)
        biases = tf.get_variable('biases', [HIDDEN_LAYER3_SIZE],
                                 initializer = tf.constant_initializer(0.1))
        layer3 = tf.nn.elu(tf.matmul(layer2, weights) + biases)
    return layer3

## define the training function
def trainNN(X_train, X_test, y_train, y_test, X_val, y_val):
    t = time.time()
    with tf.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.float32, [None, INPUT_SIZE], name = 'xInput')
        yTrue = tf.compat.v1.placeholder(tf.int32, [None, OUTPUT_SIZE], name = 'yInput')
    # generate the predicted labels from DNN
    yPrediction = buildDNN(x)
    # build the lost functions and I use crossentropy
    with tf.name_scope('lossDNN'):
        crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yPrediction, labels=tf.argmax(yTrue, 1))
        loss = tf.reduce_mean(crossEntropy)
    # choose the Adam optimizer and the learning rate is chosen by trial and error
    with tf.name_scope('trainDNN'):
         train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    # define the evaluation standard
    correctPrediction = tf.equal(tf.argmax(yPrediction,1), tf.argmax(yTrue,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    # START training ... ...
    with tf.Session() as sess:
        # use Tensorboard to visualize
        writer = tf.summary.FileWriter('graphDNN',sess.graph)
        # initialize 
        sess.run(tf.global_variables_initializer())
        # feed the validation and test data
        validationFeed = {x: X_val, yTrue: y_val}
        testFeed = {x: X_test, yTrue: y_test}
        tStartDNN = time.time() - t
        for i in range(TRAINING_STEPS+1):
            # record the accuracies every INTERVAL_SHOW iterations
            if i % INTERVAL_SHOW == 0:
                validationAcc = sess.run(accuracy, feed_dict = validationFeed)
                validationAccDNN.append(validationAcc)
                testAcc = sess.run(accuracy, feed_dict = testFeed)
                testAccDNN.append(testAcc)
                print("DNN: %d rounds, validationAcc = %g, testAcc=%g" % (i, validationAcc, testAcc))
            # keep training
            sess.run(train, feed_dict = {x: X_train, yTrue: y_train})
        tEndDNN = time.time() - t
        timeCostDNN = tEndDNN - tStartDNN
        print(timeCostDNN)
    writer.close()


INPUT_SIZE = 1892
OUTPUT_SIZE = 3

HIDDEN_LAYER1_SIZE = 512
HIDDEN_LAYER2_SIZE = 256
HIDDEN_LAYER3_SIZE = 128
BATCH_SIZE = 200
LEARNING_RATE = 0.001
TRAINING_STEPS = 500
INTERVAL_SHOW = 50
validationAccDNN = []
testAccDNN = []
numParameterDNN = (1892*512+512) + (512*256+256) + (256*128+128) + (128*3+3)
tf.compat.v1.reset_default_graph()

X_1, y_1=get_data('Dataset/HH1.csv', 1)
X_2, y_2=get_data('Dataset/HH2.csv', 2)
X_3, y_3=get_data('Dataset/HH3.csv', 3)
#X_4, y_4=get_data('Dataset/HH4.csv', 4)
#X_5, y_5=get_data('Dataset/nep10.csv', 5)
#X_6, y_6=get_data('Dataset/nep20.csv', 6)
#X_7, y_7=get_data('Dataset/nep30.csv', 7)
#X_8, y_8=get_data('Dataset/nep40.csv', 8)
#X_9, y_9=get_data('Dataset/nep50.csv', 9)
#X_10, y_10=get_data('Dataset/nep60.csv', 10)
#X_11, y_11=get_data('Dataset/Vodka15.csv', 11)
#X_12, y_12=get_data('Dataset/Vodka20.csv', 12)
#X_13, y_13=get_data('Dataset/Vodka30.csv', 13)
#X_14, y_14=get_data('Dataset/Vodka40.csv', 14)
#X_15, y_15=get_data('Dataset/Vodka50.csv', 15)
#X_16, y_16=get_data('Dataset/Vodka60.csv', 16)
X=np.vstack([X_1, X_2, X_3])#, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14, X_15, X_16])
X = preprocessing.scale(X)
y = np.append(y_1, y_2)
y = np.append(y, y_3)
#y = np.append(y, y_4)
#y = np.append(y, y_5)
#y = np.append(y, y_6)
#y = np.append(y, y_7)
#y = np.append(y, y_8)
#y = np.append(y, y_9)
#y = np.append(y, y_10)
#y = np.append(y, y_11)
#y = np.append(y, y_12)
#y = np.append(y, y_13)
#y = np.append(y, y_14)
#y = np.append(y, y_15)
#y = np.append(y, y_16)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
y_val = lb.transform(y_val)
trainNN(X_train, X_test, y_train, y_test, X_val, y_val)













