import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
def get_data2(filename):
    file = pd.read_csv(filename)
    row=1
    X=file.iloc[1, 54:587].values
    while row < 10:    
        u = file.iloc[row, 54:587].values
        X=np.vstack([X, u])
        row+=1  
    return X
def get_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[1:101, 54:587].values
    y = np.full((1, 100), label)
    Z = data.iloc[200, 54:587].values
    X = Z/X
    return X, y
def variableszerovariance(X):
    Var0Variable = np.where( X.var(axis=0) == 0 )
    if len(Var0Variable[0]) == 0:
        print( "No variables with zero variance" )
    else:
        print( "{0} variable(s) with zero variance".format(len(Var0Variable[0])))
        print( "Variable number: {0}".format(Var0Variable[0]+1) )
        print( "The variable(s) is(are) deleted." )
    return Var0Variable

X_1, y_1=get_data('Dataset/nep10.csv', 1)
X_2, y_2=get_data('Dataset/nep40.csv', 2)
X_3, y_3=get_data('Dataset/nep60.csv', 2)
#variableszerovariance(X_1)

c_1 = []
c_2 = []
c_3 = []
choose = []
for i in np.arange(0, 533):
    for j in np.arange(10):
        c_1.append(X_1[j][i])
        c_2.append(X_2[j][i])
        c_3.append(X_3[j][i])
    corr = np.corrcoef(c_1, c_2, c_3)
    c_1 = []
    c_2 = []
    c_3 = []
    if corr[0][1] < -0.4:
        choose.append(i)
print(choose)

